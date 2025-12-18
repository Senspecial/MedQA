import os
import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定 HF 缓存目录（可选）
os.environ["HF_HOME"] = "/sakura/sjs/hf_cache"

# ========== 配置 ==========
# 你的基座模型（用 Instruct 版是对的）
base_model_name = "Qwen/Qwen2.5-14B-Instruct"

# 输入：SFT 清洗后的 QA 数据
input_path = "/home/mry/sjs/MedQA/dpo.json"

# 输出：用于 RL/DPO 的三元组
output_jsonl = "/home/mry/sjs/MedQA/src/data/med_triples.jsonl"
output_json = "/home/mry/sjs/MedQA/src/data/med_triples.json"

# 限制最多处理多少条（调试时可以先设小一点，比如 1000）
max_samples = None  # 或者 10000，比如 10000


# ========== 加载模型 tokenizer ==========
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 如果没 chat_template，就补一份 Qwen 风格的
if not getattr(tokenizer, "chat_template", None):
    tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' -%}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' -%}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}"""

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)
model.eval()
model.config.use_cache = False

SYSTEM_PROMPT = (
    "你是一名谨慎的中文医学问答助手，请根据医学知识简要回答用户问题，"
    "不需要特别详细，不给出确切诊断和处方，并提醒不能替代线下就医。"
)

max_new_tokens = 256
temperature = 0.7   # 略降一点，减少发疯
top_p = 0.9

# 把 eos_token 和 <|im_end|> 都作为停止符号
eos_token_ids = [tokenizer.eos_token_id]
try:
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id not in eos_token_ids:
        eos_token_ids.append(im_end_id)
except Exception:
    pass


def build_chat_prompt(query: str) -> str:
    """
    用 Qwen 的 chat_template 构造真正喂给模型的 prompt。
    这里只用于生成 rejected，不写入三元组文件。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 结尾会加上 assistant 的起始标记
    )
    return prompt


def is_bad_text(text: str) -> bool:
    """
    简单过滤一下明显异常/垃圾的生成：
    - 含黄暴/奇怪关键词
    - 中文比例太低（基本全是乱码）
    - 同一字符长时间重复
    """
    if not text:
        return True

    lower = text.lower()

    # 1) 黑名单关键词（可以自己继续加）
    blacklist = ["creampie", "porn", "性交", "强奸"]
    for bad in blacklist:
        if bad in lower:
            return True

    # 2) 中文比例
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    zh_ratio = len(zh_chars) / max(len(text), 1)
    if zh_ratio < 0.3:
        return True

    # 3) 检查是否存在超长重复字符
    max_run = 0
    last_ch = ""
    run = 0
    for ch in text:
        if ch == last_ch:
            run += 1
        else:
            last_ch = ch
            run = 1
        if run > max_run:
            max_run = run
    if max_run > 20:
        return True

    return False


@torch.no_grad()
def generate_rejected(query: str) -> str:
    """
    用 Qwen2.5-14B-Instruct 生成一条回答，作为 rejected。
    返回纯回答文本。
    """
    prompt_text = build_chat_prompt(query)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_token_ids,   # 加上这一行，遇到 <|im_end|> 也停
    )

    # 截掉输入部分，只保留模型新生成的内容
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    # 如果里面还有 <|im_end|>，截掉后面的
    end_tag = "<|im_end|>"
    if end_tag in full_text:
        full_text = full_text.split(end_tag)[0]

    # 去掉残留 special token
    clean = full_text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    clean = clean.strip()

    # 控制在一两句话内（避免越写越跑偏）
    parts = re.split(r"[。！？\n]", clean)
    parts = [p.strip() for p in parts if p.strip()]
    if parts:
        clean = parts[0] + "。"

    return clean


# ========== 读取 clean_datasets.json ==========
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)   # list[{"query": str, "response": str}, ...]

if max_samples is not None:
    data = data[:max_samples]

print(f"载入 {len(data)} 条 QA 样本，开始生成 rejected...")

triples = []

for item in tqdm(data, desc="构造三元组"):
    query = (item.get("query") or "").strip()
    chosen = (item.get("response") or "").strip()
    if not query or not chosen:
        continue

    rejected = generate_rejected(query)

    # 简单过滤一下质量很差的（太短、空、完全相同、判定为垃圾）
    if not rejected:
        continue
    if len(rejected) < 10:
        continue
    if rejected == chosen:
        continue
    if is_bad_text(rejected):
        continue

    triples.append({
        "prompt": query,     # 注意：这里只存“纯问题文本”，模板留给训练脚本去套
        "chosen": chosen,    # 人工/百科答案
        "rejected": rejected # 基座模型生成的答案
    })

print(f"最终得到 {len(triples)} 条三元组")

# ========== 保存为 jsonl 和 json ==========
with open(output_jsonl, "w", encoding="utf-8") as f:
    for ex in triples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(triples, f, ensure_ascii=False, indent=2)

print("已保存到：")
print("  jsonl:", output_jsonl)
print("  json :", output_json)
