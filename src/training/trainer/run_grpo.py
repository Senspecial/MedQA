import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import torch
import re
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
import numpy as np
# ========== 可选：HF 缓存路径 ==========
# os.environ["HF_HOME"] = "/sakura/sjs/hf_cache"

# ========== 1. 路径配置 ==========
base_model_name = "/sakura/sjs/models/Qwen2.5-1.5B/qwen/Qwen2___5-1___5B"
sft_lora_path = "/home/mry/sjs/MedQA/save/medical_sft_qwen2_5_1_5b"

dataset_path = "/home/mry/sjs/MedQA/src/data/med_triples.json"  # [{prompt, chosen, rejected}, ...]

# ========== 2. 4bit 配置（QLoRA） ==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ========== 3. tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    padding_side="right",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

SYSTEM_PROMPT = (
    "你是一名谨慎的中文医学问答助手，请用中文简要回答问题，"
    "不做准确诊断和处方，并提醒不能替代线下就医。"
)

def format_prompt(query: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 让模型从 assistant 开始生成
    )

# ========== 4. 加载 base + QLoRA-SFT ==========

# 4.1 加载 4bit base 模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model = prepare_model_for_kbit_training(base_model)
base_model.config.use_cache = False

# 4.2 挂载你 SFT 训练好的 LoRA adapter
# 注意：这个 checkpoint 目录里应该有 adapter_config.json / adapter_model.bin
model = PeftModel.from_pretrained(base_model, sft_lora_path, is_trainable=True, )
model.print_trainable_parameters()  # 可以看一下目前哪些是 trainable

# ========== 5. 加载三元组数据 ==========
raw_dataset = load_dataset(
    "json",
    data_files={"train": dataset_path}
)["train"]

# 这里只用 prompt 做 RL，chosen / rejected 暂时不用
def preprocess_function(examples):
    prompts = [format_prompt(p) for p in examples["prompt"]]
    return {
        "prompt": prompts,
        "chosen": examples["chosen"],
        "rejected": examples["rejected"],
    }

dataset = raw_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=[],  # 只保留 prompt
)

# 调试时可以先只用一部分
# dataset = dataset.select(range(1000))

# ========== 6. GRPO 配置 ==========
grpo_config = GRPOConfig(
    output_dir="save/Qwen2_5-1_5B-GRPO-med",
    logging_steps=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-6,          # RL 一般再保守一点
    fp16=True,                   # 你的卡支持的话；否则用 fp16=True
    max_prompt_length=512,
    max_completion_length=256,
    remove_unused_columns=False,
    seed=42,
)
def unwrap_completions(completions):
    """
    completions: list[list[{"role": "...", "content": "..."}]]
    返回: list[str]
    """
    texts = []
    for c in completions:
        # GRPO 默认每个 completion 是一个长度为 1 的消息列表
        if isinstance(c, list) and len(c) > 0 and isinstance(c[0], dict):
            texts.append(c[0].get("content", ""))
        elif isinstance(c, dict) and "content" in c:
            texts.append(c["content"])
        else:
            # 万一已经是字符串，兜底一下
            texts.append(str(c))
    return texts

# ========== 7. reward 函数 ==========
def reward_len(completions, target_min=80, target_max=200):
    texts = unwrap_completions(completions)
    rewards = []
    for text in texts:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        L = len(tokens)
        mid = (target_min + target_max) / 2
        rewards.append(-abs(L - mid) / 20.0)
    return rewards

def extract_keywords(text, min_len=2):
    # 用中文标点和换行分割
    pieces = re.split(r"[，。？！、；：\n\r\s]", text)
    kws = set()
    for p in pieces:
        p = p.strip()
        # 只要长度>=2的纯中文串，避免很多垃圾碎片
        if len(p) >= min_len and re.search(r"[\u4e00-\u9fff]", p):
            kws.add(p)
    return kws

def overlap_score(gen, ref):
    ref_kws = extract_keywords(ref)
    if not ref_kws:
        return 0.0
    count = sum(1 for kw in ref_kws if kw in gen)
    return count / len(ref_kws)  # 0~1 之间
def similarity_with_rejected(gen, rej):
    return overlap_score(gen, rej)  # 重叠越多越不好
# ===== 症状 (Symptoms) =====
MED_SYMPTOMS = [
    # 原有
    "头晕", "胸闷", "腹痛", "乏力", "咳嗽", "发热", "恶心", "呕吐", "腹泻",
    # 新增（来自文本）
    "便秘", "便血", "粘液便", "贫血", "消瘦", "腹胀", "食欲不振", "失眠", "多梦",
    "心率减慢", "心动过缓", "水肿", "黄疸", "尿色加深", "皮肤瘙痒", "夜盲",
    "牙龈出血", "鼻出血", "蜘蛛痣", "肝掌", "脸色黝黑", "浮肿", "骨质疏松",
    "性欲减退", "月经失调", "易倦", "思睡", "厌油"
]

# ===== 疾病 (Diseases) =====
MED_DISEASES = [
    # 原有
    "直肠炎", "结肠炎", "甲减", "甲状腺功能减退", "冠心病", "心衰", "心力衰竭",
    # 新增（来自文本）
    "慢性直肠炎", "溃疡性结肠炎", "甲减性心脏病", "肝硬化", "胃底息肉", "脂肪肝",
    "分泌性中耳炎", "甲状腺炎", "结节性甲状腺肿", "高脂血症", "房室传导阻滞",
    "扭转型室速", "心房颤动", "食道静脉曲张", "腹水", "胸水"
]

# ===== 并发症 / 严重后果 (Complications) =====
MED_COMPS = [
    # 原有
    "癌变", "心包积液", "大出血", "动脉硬化", "高血压", "心律失常",
    # 新增（来自文本）
    "心脏扩大", "心肌病变", "心包填塞", "冠心病", "肝功能异常", "凝血障碍",
    "激素代谢异常", "维生素代谢障碍", "白蛋白合成障碍", "门脉高压",
    "肝性脑病", "肾功能损害", "电解质紊乱"
]

# ===== 病理/生理机制 (Mechanisms) =====
MED_MECH = [
    # 原有
    "基础代谢", "脂肪代谢", "水钠潴留", "毛细血管通透性", "心排量", "外周阻力",
    # 新增（来自文本）
    "胆固醇半寿期延长", "心肌酶活性抑制", "肌浆网功能降低", "ATP酶活性降低",
    "儿茶酚胺敏感性降低", "Q-T间期延长", "有效循环血量不足", "代偿性血管收缩",
    "黏多糖堆积", "淋巴回流减慢", "胆色素代谢异常", "糖代谢障碍",
    "凝血因子合成障碍", "肝细胞损害"
]

# ===== 治疗与生活建议 (Treatments & Lifestyle) =====
MED_TREAT = [
    # 原有
    "定期复查", "饮食控制", "戒烟", "忌酒", "适当运动", "遵医嘱", "及时就医",
    # 新增（来自文本）
    "低脂饮食", "软食", "半流质饮食", "低盐饮食", "避免粗食", "补充维生素",
    "补充铁剂", "细嚼慢咽", "多吃新鲜蔬菜水果", "避免生冷硬食物",
    "避免油煎酸辣食物", "保持良好作息", "不熬夜", "提高身体素质",
    "及早治疗原发病", "加强体育锻炼", "注意清洁卫生", "调适温暖"
]
def extract_medical_points(text: str):
    pts = {"disease": set(), "symptom": set(), "comp": set(), "mech": set(), "treat": set()}
    for w in MED_DISEASES:
        if w in text:
            pts["disease"].add(w)
    for w in MED_SYMPTOMS:
        if w in text:
            pts["symptom"].add(w)
    for w in MED_COMPS:
        if w in text:
            pts["comp"].add(w)
    for w in MED_MECH:
        if w in text:
            pts["mech"].add(w)
    for w in MED_TREAT:
        if w in text:
            pts["treat"].add(w)
    return pts

def medical_coverage(gen: str, ref: str) -> float:
    ref_pts = extract_medical_points(ref)
    if not any(ref_pts.values()):
        return 0.0

    gen_pts = extract_medical_points(gen)
    score = 0.0
    total_w = 0.0

    weights = {
        "disease": 3.0,   # 疾病名
        "symptom": 2.0,   # 症状
        "comp": 2.5,      # 并发症
        "mech": 1.5,      # 机制
        "treat": 2.0,     # 处理/建议
    }

    for k, w in weights.items():
        if len(ref_pts[k]) == 0:
            continue
        cover = len(ref_pts[k] & gen_pts[k]) / len(ref_pts[k])
        score += w * cover
        total_w += w

    return score / total_w if total_w > 0 else 0.0

def safety_reward(gen):
    text = gen.lower()

    reward = 0.0

    # 正向：鼓励就医 & 声明不能替代线下医生
    positive_patterns = [
        "不能替代线下就医",
        "建议尽快到医院",
        "建议到医院就诊",
        "建议咨询医生",
        "需由医生评估",
        "及时就医",
        "进一步检查",
        "请在医生指导下",
    ]

    for pat in positive_patterns:
        if pat in text:
            reward += 1.5

    # 负向：处罚擅自给药、调整药量、劝不就医
    negative_patterns = [
        "自行停药",
        "自己停药",
        "自己增减药量",
        "自行增减药量",
        "不用去医院",
        "不必去医院",
        "可以自行用药",
        "可以随意用药",
    ]

    for pat in negative_patterns:
        if pat in text:
            reward -= 2.0

    overclaim = [
        "一定能治好",
        "保证治愈",
        "百分之百治好",
    ]
    for p in overclaim:
        if p in gen:
            reward -= 3.0

    return reward

def med_reward(completions, **kwargs):
    gen_texts = unwrap_completions(completions)

    chosen_list   = kwargs.get("chosen", [])
    rejected_list = kwargs.get("rejected", [])

    n = min(len(gen_texts), len(chosen_list), len(rejected_list))
    gen_texts = gen_texts[:n]
    chosen_list = chosen_list[:n]
    rejected_list = rejected_list[:n]

    len_scores     = reward_len(gen_texts)
    cov_scores     = []
    rej_sim_scores = []
    safe_scores    = []

    for c, ch, rj in zip(completions, chosen_list, rejected_list):
        cov     = medical_coverage(c, ch)         # 内容覆盖
        rej_sim = medical_coverage(c, rj)         # 跟废话/简略答案的相似度
        safe    = safety_reward(c)                # 安全性

        cov_scores.append(cov)
        rej_sim_scores.append(rej_sim)
        safe_scores.append(safe)

    rewards = []
    for i in range(len(completions)):
        r_len = len_scores[i]
        r_cov = cov_scores[i]
        r_rej = rej_sim_scores[i]
        r_safe = safe_scores[i]

        # 权重可以根据实验改，这里给你一个适合“知识型问答”的起始值
        R = (
                0.5 * r_len +  # 长度适中
                3.0 * r_cov +  # 医学要点覆盖
                -2.5 * r_rej +  # 越像 rejected 越扣
                2.0 * r_safe  # 安全性最重要
        )
        rewards.append(R)

        # 做个裁剪和标准化，训练更稳
    rewards = np.array(rewards, dtype=np.float32)
    rewards = np.clip(rewards, -10.0, 10.0)
    if rewards.std() > 1e-6:
        rewards = (rewards - rewards.mean()) / rewards.std()

    return rewards.tolist()

# ========== 9. 创建 GRPOTrainer ==========
trainer = GRPOTrainer(
    model=model,                    # 这里传的是已经挂好 LoRA 的模型
    reward_funcs=med_reward,
    args=grpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    #tokenizer=tokenizer,
    # 注意：这里 *不要* 再传 peft_config，否则会再包一层新的 LoRA
)

# ========== 10. 开始训练 ==========
trainer.train()

# ========== 11. 保存 RL 后的 LoRA ==========
save_dir = "save/Qwen2_5-1_5B-GRPO-med-final"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print("GRPO 训练完成，模型保存在:", save_dir)

#python -m src.training.trainer.run_grpo.py