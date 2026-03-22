"""
使用 DeepSeek API 批量生成"直接回答（不搜索）"的 Agent SFT 补充数据。

策略：
  1. 先用 DeepSeek 对种子问题做 N 倍扩展，生成多样化的用户问法
  2. 再用 DeepSeek 对每个问题生成 <think>+<answer> 格式的直接回答
  3. 校验格式后输出为 sft_agent_r1.jsonl 兼容格式

用法：
    python src/training/scripts/run_direct_answer_construction.py \
        --config config/sft_direct_answer_config.yaml

    # 快速测试
    python src/training/scripts/run_direct_answer_construction.py \
        --config config/sft_direct_answer_config.yaml --max_samples 5

    # 跳过扩展，直接用种子问题生成回答
    python src/training/scripts/run_direct_answer_construction.py \
        --config config/sft_direct_answer_config.yaml --skip_expand
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import requests
import yaml
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────── Agent 系统提示（从统一 yaml 加载，与推理时一致） ──

def _load_agent_system_prompt():
    """从 config/agent_system_prompt.yaml 加载统一 Agent 提示词"""
    yaml_path = os.path.join(_PROJECT_ROOT, "config", "agent_system_prompt.yaml")
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return (cfg.get("agent_system_prompt") or "").strip() or None
    except Exception:
        return None

AGENT_SYSTEM_PROMPT = _load_agent_system_prompt() or """\
你是一个专业的医疗健康信息助手，具备搜索医学知识的能力，也可以直接根据已有知识给出回答。

**何时搜索、何时直接回答**：
- 日常问候（你好、谢谢等）或闲聊（医学以外知识、没有说明具体病因的问题等）：直接回答，不要搜索
- 你已经有把握的常识性医学问题：直接回答，不要搜索
- 需要精确数据、指南或不确定的医学问题：先搜索再回答

**搜索时严格遵守以下格式**：

需要搜索时：
<think>分析问题，明确需要搜索哪些医学知识</think>
<tool_call>{"name":"search","arguments":{"query":"搜索关键词"}}</tool_call>

收到搜索结果后，可继续搜索或给出最终回答：
<think>基于已有信息进行分析</think>
<answer>完整、安全、专业的医疗回答</answer>

不需要搜索时，直接回答：
<think>简要分析</think>
<answer>回答内容</answer>

**医疗回答原则**：
- 使用不确定表述：「可能是」「考虑是」「建议检查」
- 不做明确诊断，不开具处方，不给出具体用药剂量
- 有危及生命的症状时，明确建议立即就医"""


# ─────────────────────────── DeepSeek API 客户端 ──────────────────────────────

class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-chat", temperature: float = 0.7,
                 max_tokens: int = 2048, max_retries: int = 3,
                 retry_delay: float = 2.0, request_interval: float = 0.5):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_interval = request_interval
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_tokens,
        }
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(self.request_interval)
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers, json=payload, timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                logger.warning(f"API 请求失败 (第 {attempt} 次): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        raise RuntimeError(f"API 连续失败 {self.max_retries} 次: {last_error}")


# ─────────────────────────── 问题扩展 ─────────────────────────────────────────

EXPAND_PROMPT_TEMPLATE = """\
请为以下{category_cn}类别的种子问题，每个生成 {n} 个不同的变体表述。
要求：
- 意思相似但表达方式不同（口语化、正式、简短、详细混合）
- 模拟真实用户的不同问法
- 每个变体单独一行，不要编号，不要引号
- 直接输出变体，不要加任何解释

种子问题列表：
{seeds}"""

CATEGORY_CN = {
    "greetings": "日常对话/问候",
    "common_medical": "常识性医学问题",
    "clarification": "模糊/需要澄清的问题",
    "off_topic": "非医学话题",
}


def expand_questions(client: DeepSeekClient, seeds: List[str],
                     category: str, n: int) -> List[Tuple[str, str]]:
    """用 DeepSeek 对种子问题做 N 倍扩展，返回 [(question, category), ...]"""
    if n <= 0:
        return [(s, category) for s in seeds]

    category_cn = CATEGORY_CN.get(category, category)
    prompt = EXPAND_PROMPT_TEMPLATE.format(
        category_cn=category_cn,
        n=n,
        seeds="\n".join(f"- {s}" for s in seeds),
    )

    logger.info(f"扩展 {category} ({len(seeds)} 种子 × {n} 倍)...")
    try:
        result = client.chat([
            {"role": "system", "content": "你是一个数据生成助手，帮助生成训练数据的问题变体。"},
            {"role": "user", "content": prompt},
        ], temperature=0.9)

        lines = [l.strip().lstrip("- ·•·").strip()
                 for l in result.strip().split("\n") if l.strip()]
        lines = [l for l in lines if len(l) >= 2 and len(l) <= 200]
        expanded = [(l, category) for l in lines]

        logger.info(f"  {category}: 种子 {len(seeds)} → 扩展得 {len(expanded)} 条")
        return [(s, category) for s in seeds] + expanded

    except Exception as e:
        logger.error(f"扩展 {category} 失败: {e}，仅使用种子问题")
        return [(s, category) for s in seeds]


# ─────────────────────────── 回答生成 ─────────────────────────────────────────

GENERATE_PROMPT_TEMPLATES = {
    "greetings": (
        "用户对你说：「{question}」\n\n"
        "这是日常对话/问候，请严格按以下格式直接回答（不要搜索）：\n"
        "<think>简要判断这是什么类型的对话</think>\n"
        "<answer>友好、简洁的回应，说明你是医疗健康助手，可以帮忙解答健康问题</answer>"
    ),
    "common_medical": (
        "用户问：「{question}」\n\n"
        "这是常见的健康常识问题，你已有足够知识直接回答，不需要搜索。\n"
        "请严格按以下格式回答：\n"
        "<think>简要分析问题，说明这是常识性问题可以直接回答</think>\n"
        "<answer>专业、完整的医疗健康回答，需包含：\n"
        "- 问题的解释/原因\n"
        "- 处理建议或注意事项\n"
        "- 需要就医的警示情况\n"
        "- 末尾加上免责提示</answer>"
    ),
    "clarification": (
        "用户说：「{question}」\n\n"
        "用户的描述太模糊，无法直接给出有用的回答，需要引导用户提供更多信息。\n"
        "请严格按以下格式回答：\n"
        "<think>分析用户的表述，指出缺少哪些关键信息</think>\n"
        "<answer>友好地请用户补充具体信息（症状部位、持续时间、具体指标等），\n"
        "同时提醒如果症状严重应及时就医</answer>"
    ),
    "off_topic": (
        "用户说：「{question}」\n\n"
        "这不是医学问题。你是医疗健康助手，不擅长回答非医学话题。\n"
        "请严格按以下格式回答：\n"
        "<think>判断这不属于医疗健康领域</think>\n"
        "<answer>礼貌地说明自己是医疗健康助手，这个问题超出了你的专业范围，\n"
        "引导用户提出健康相关的问题</answer>"
    ),
}

THINK_PAT = re.compile(r'<think>(.*?)</think>', re.DOTALL)
ANSWER_PAT = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)


def generate_one(idx: int, question: str, category: str,
                 client: DeepSeekClient) -> Optional[Dict]:
    """为一个问题生成直接回答样本"""
    template = GENERATE_PROMPT_TEMPLATES.get(category, GENERATE_PROMPT_TEMPLATES["common_medical"])
    user_prompt = template.format(question=question)

    try:
        raw = client.chat([
            {"role": "system", "content": "你是一个训练数据生成助手。请严格按照要求的格式输出。"},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        logger.error(f"[{idx}] API 失败: {e}")
        return None

    think_m = THINK_PAT.search(raw)
    answer_m = ANSWER_PAT.search(raw)

    if not think_m or not answer_m:
        logger.warning(f"[{idx}] 格式不完整，跳过 (question={question[:30]})")
        return None

    think_text = think_m.group(1).strip()
    answer_text = answer_m.group(1).strip()

    if len(answer_text) < 5:
        logger.warning(f"[{idx}] answer 太短 ({len(answer_text)} 字符)，跳过")
        return None

    assistant_content = f"<think>{think_text}</think>\n<answer>{answer_text}</answer>"

    return {
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ],
        "_meta": {
            "index": idx,
            "num_tool_calls": 0,
            "num_turns": 1,
            "type": "direct_answer",
            "category": category,
        },
    }


# ─────────────────────────── 线程安全写入器 ───────────────────────────────────

class ResultWriter:
    def __init__(self, output_path: str, save_every: int = 20):
        self.output_path = output_path
        self.save_every = save_every
        self._lock = Lock()
        self._buffer: List[Dict] = []
        self.total_ok = 0
        self.total_fail = 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: Dict):
        with self._lock:
            self._buffer.append(record)
            self.total_ok += 1
            if len(self._buffer) >= self.save_every:
                self._flush()

    def add_fail(self):
        with self._lock:
            self.total_fail += 1

    def _flush(self):
        if not self._buffer:
            return
        with open(self.output_path, "a", encoding="utf-8") as f:
            for r in self._buffer:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._buffer.clear()

    def finalize(self):
        with self._lock:
            self._flush()


# ─────────────────────────── 主流程 ───────────────────────────────────────────

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="批量生成直接回答 Agent SFT 数据")
    parser.add_argument("--config", default="config/sft_direct_answer_config.yaml")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最多生成样本数（覆盖配置）")
    parser.add_argument("--skip_expand", action="store_true",
                        help="跳过问题扩展，直接使用种子问题")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)
    cfg = load_config(config_path)

    # API 客户端
    ds_cfg = cfg["deepseek"]
    api_key = os.environ.get("DEEPSEEK_API_KEY") or ds_cfg.get("api_key", "")
    if not api_key or api_key.startswith("sk-xxx"):
        raise ValueError("请设置 DEEPSEEK_API_KEY 或在配置中填写 api_key")

    client = DeepSeekClient(
        api_key=api_key,
        base_url=ds_cfg.get("base_url", "https://api.deepseek.com/v1"),
        model=ds_cfg.get("model", "deepseek-chat"),
        temperature=ds_cfg.get("temperature", 0.85),
        max_tokens=ds_cfg.get("max_tokens", 2048),
        max_retries=ds_cfg.get("max_retries", 3),
        retry_delay=ds_cfg.get("retry_delay", 2.0),
        request_interval=ds_cfg.get("request_interval", 0.5),
    )

    # 收集种子问题
    seed_cfg = cfg.get("seed_questions", {})
    expand_cfg = cfg.get("expansion", {})
    target = args.max_samples or cfg.get("output", {}).get("target_samples", 300)

    all_questions: List[Tuple[str, str]] = []  # (question, category)

    for category, seeds in seed_cfg.items():
        if not isinstance(seeds, list):
            continue
        n = 0 if args.skip_expand else expand_cfg.get(category, 2)
        expanded = expand_questions(client, seeds, category, n)
        all_questions.extend(expanded)

    # 去重
    seen = set()
    unique = []
    for q, cat in all_questions:
        key = q.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append((q, cat))
    all_questions = unique

    if len(all_questions) > target:
        all_questions = all_questions[:target]

    logger.info(f"共 {len(all_questions)} 个问题待生成回答 (目标 {target} 条)")

    # 输出
    output_path = cfg.get("output", {}).get("output_path", "data/SFT_Agent/sft_direct_answer_deepseek.jsonl")
    if not os.path.isabs(output_path):
        output_path = os.path.join(_PROJECT_ROOT, output_path)

    conc_cfg = cfg.get("concurrency", {})
    num_workers = args.num_workers or conc_cfg.get("num_workers", 3)
    save_every = conc_cfg.get("save_every", 20)

    writer = ResultWriter(output_path, save_every)

    # 清空已有文件
    open(output_path, "w").close()

    print("=" * 60)
    print("直接回答样本构造")
    print("=" * 60)
    print(f"  问题数: {len(all_questions)}")
    print(f"  输出  : {output_path}")
    print(f"  并发  : {num_workers}")
    print(f"  目标  : {target} 条")
    print("=" * 60)

    # 并发生成
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(generate_one, i, q, cat, client): i
            for i, (q, cat) in enumerate(all_questions)
        }

        pbar = tqdm(total=len(futures), desc="生成进度")
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    writer.add(result)
                    if writer.total_ok >= target:
                        logger.info(f"已达目标 {target} 条，停止")
                        for f in futures:
                            f.cancel()
                        pbar.update(1)
                        break
                else:
                    writer.add_fail()
            except Exception as e:
                logger.error(f"未捕获异常: {e}")
                writer.add_fail()
            pbar.update(1)
        pbar.close()

    writer.finalize()

    print()
    print("=" * 60)
    print("构造完成！")
    print("=" * 60)
    print(f"  成功: {writer.total_ok} 条")
    print(f"  失败: {writer.total_fail} 条")
    print(f"  输出: {output_path}")
    print()
    print("下一步：将补充数据追加到训练集：")
    print(f"  cat {output_path} >> {_PROJECT_ROOT / 'data/SFT_Agent/sft_agent_r1.jsonl'}")
    print()
    print("然后重新训练 Agent SFT 模型。")


if __name__ == "__main__":
    main()
