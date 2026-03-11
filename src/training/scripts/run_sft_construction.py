"""
SFT 数据构造脚本 (run_sft_construction.py)

使用 DeepSeek API 对原始医疗问答进行增强，生成带推理链 + 模拟搜索的高质量 SFT 数据。

输出格式（assistant 回复）:
    <think>推理过程</think>
    <tool_call>{"name":"search","arguments":{"query":"关键词"}}</tool_call>
    <search_result>模拟检索到的知识</search_result>
    <think>继续推理</think>        ← 可多轮
    <tool_call>...</tool_call>    ← 可多次搜索
    <search_result>...</search_result>
    <answer>最终医疗回答</answer>

使用方式:
    python src/training/scripts/run_sft_construction.py \\
        --config config/sft_construction_config.yaml

    # 快速测试（只处理前10条）
    python src/training/scripts/run_sft_construction.py \\
        --config config/sft_construction_config.yaml --max_samples 10

    # 从第100条继续处理（断点续处理）
    python src/training/scripts/run_sft_construction.py \\
        --config config/sft_construction_config.yaml --start_index 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SFTSample:
    """一条 SFT 训练样本"""
    question: str                   # 原始问题
    raw_answer: str                 # 原始参考答案
    constructed_response: str       # DeepSeek 生成的完整回复（含所有标签）
    num_tool_calls: int             # 搜索次数
    think_segments: List[str]       # 每段 <think> 内容
    search_queries: List[str]       # 每次搜索的 query
    search_results: List[str]       # 每次搜索的 result
    final_answer: str               # <answer> 内容
    index: int                      # 在原始数据集中的下标

    def to_chat_dict(self, system_prompt: str) -> Dict:
        """转为 chat 格式（LLaMA-Factory / TRL 兼容）"""
        return {
            "messages": [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": self.question},
                {"role": "assistant", "content": self.constructed_response},
            ],
            "_meta": {
                "index": self.index,
                "num_tool_calls": self.num_tool_calls,
                "raw_answer": self.raw_answer,
            },
        }

    def to_alpaca_dict(self) -> Dict:
        """转为 alpaca 格式"""
        return {
            "instruction": self.question,
            "input": "",
            "output": self.constructed_response,
            "_meta": {
                "index": self.index,
                "num_tool_calls": self.num_tool_calls,
                "raw_answer": self.raw_answer,
            },
        }


# ---------------------------------------------------------------------------
# DeepSeek API 客户端
# ---------------------------------------------------------------------------

class DeepSeekClient:
    """封装 DeepSeek API 调用"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        request_interval: float = 0.5,
    ):
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

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """调用 chat API，返回 assistant 文本内容。失败抛出 RuntimeError。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(self.request_interval)
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                logger.warning(f"API 请求失败 (第 {attempt} 次): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"DeepSeek API 连续失败 {self.max_retries} 次: {last_error}")


# ---------------------------------------------------------------------------
# 格式解析
# ---------------------------------------------------------------------------

_TAG_THINK   = re.compile(r"<think>(.*?)</think>",             re.DOTALL)
_TAG_CALL    = re.compile(r"<tool_call>(.*?)</tool_call>",     re.DOTALL)
_TAG_RESULT  = re.compile(r"<search_result>(.*?)</search_result>", re.DOTALL)
_TAG_ANSWER  = re.compile(r"<answer>(.*?)</answer>",           re.DOTALL)
_QUERY_RE    = re.compile(r'"query"\s*:\s*"([^"]*)"')


def parse_response(text: str) -> Optional[SFTSample]:
    """
    解析 DeepSeek 生成的文本，提取各标签内容。
    任意必要标签缺失则返回 None。
    """
    think_segs   = [s.strip() for s in _TAG_THINK.findall(text)]
    call_segs    = [s.strip() for s in _TAG_CALL.findall(text)]
    result_segs  = [s.strip() for s in _TAG_RESULT.findall(text)]
    answer_segs  = [s.strip() for s in _TAG_ANSWER.findall(text)]

    if not think_segs or not call_segs or not result_segs or not answer_segs:
        return None

    # 提取每次搜索的 query
    queries = []
    for call_text in call_segs:
        m = _QUERY_RE.search(call_text)
        queries.append(m.group(1) if m else call_text[:50])

    return SFTSample(
        question="",            # 由调用方填充
        raw_answer="",          # 由调用方填充
        constructed_response=text.strip(),
        num_tool_calls=len(call_segs),
        think_segments=think_segs,
        search_queries=queries,
        search_results=result_segs,
        final_answer=answer_segs[-1],
        index=-1,               # 由调用方填充
    )


def validate_sample(
    sample: SFTSample,
    min_answer_length: int = 20,
    max_tool_calls: int = 5,
) -> Tuple[bool, str]:
    """校验解析后的样本，返回 (ok, reason)。"""
    if len(sample.final_answer) < min_answer_length:
        return False, f"answer 长度不足 {min_answer_length} 字符"
    if sample.num_tool_calls > max_tool_calls:
        return False, f"tool_call 次数超过上限 {max_tool_calls}"
    if sample.num_tool_calls != len(sample.search_results):
        return False, "tool_call 与 search_result 数量不匹配"
    return True, ""


# ---------------------------------------------------------------------------
# 单条处理
# ---------------------------------------------------------------------------

def process_one(
    idx: int,
    question: str,
    raw_answer: str,
    client: DeepSeekClient,
    system_prompt: str,
    validation_cfg: Dict,
) -> Tuple[Optional[SFTSample], None]:
    """
    处理单条问答，返回 (sample, None)。
    成功则 sample 非 None；任何失败（API/解析/校验）均静默跳过，返回 (None, None)。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{question}\n\n"
                f"（参考信息：{raw_answer}）"
            ),
        },
    ]

    try:
        raw_text = client.chat(messages)
    except Exception as e:
        logger.error(f"[{idx}] API 调用失败: {e}")
        return None, None

    sample = parse_response(raw_text)
    if sample is None:
        logger.warning(f"[{idx}] 标签解析失败，跳过")
        return None, None   # 不存，直接跳过

    sample.question   = question
    sample.raw_answer = raw_answer
    sample.index      = idx

    ok, reason = validate_sample(
        sample,
        min_answer_length=validation_cfg.get("min_answer_length", 20),
        max_tool_calls=validation_cfg.get("max_tool_calls", 5),
    )
    if not ok:
        logger.warning(f"[{idx}] 校验失败: {reason}，跳过")
        return None, None   # 不存，直接跳过

    logger.info(
        f"[{idx}] OK | 搜索 {sample.num_tool_calls} 次 | "
        f"answer {len(sample.final_answer)} 字符"
    )
    return sample, None


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_input_data(
    input_path: str,
    dedup: bool = True,
    start_index: int = 0,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
) -> List[Tuple[int, str, str]]:
    """
    加载原始问答数据，返回 [(global_idx, question, answer), ...]。
    支持 JSON 数组（含 question/answer 字段）和 JSONL 格式。

    Args:
        input_path  : 输入文件路径
        dedup       : 是否对 question 去重
        start_index : 跳过前 N 条（顺序模式下的断点续处理）
        max_samples : 最多返回多少条输入（None = 不限）
        shuffle     : 是否随机打乱后再取样
        shuffle_seed: 随机种子（保证可复现）
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 自动识别 JSON 数组 / JSONL
    with path.open(encoding="utf-8") as f:
        first_char = f.read(1)

    if first_char == "[":
        with path.open(encoding="utf-8") as f:
            raw_list = json.load(f)
    else:
        raw_list = []
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_list.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {lineno} 行解析失败，已跳过: {e}")

    if not isinstance(raw_list, list):
        raise ValueError("输入数据须为 JSON 数组或 JSONL")

    # 去重
    seen_questions: set = set()
    candidates: List[Tuple[int, str, str]] = []
    for i, item in enumerate(raw_list):
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        if dedup:
            if q in seen_questions:
                continue
            seen_questions.add(q)
        candidates.append((i, q, a))

    # 随机打乱
    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(candidates)
        logger.info(f"已随机打乱（seed={shuffle_seed}）")
    else:
        # 顺序模式：跳过前 start_index 条
        candidates = candidates[start_index:]

    # 截断到 max_samples
    if max_samples:
        candidates = candidates[:max_samples]

    logger.info(
        f"加载 {len(candidates)} 条问答"
        f"（shuffle={shuffle}, dedup={dedup}, max_samples={max_samples}）"
    )
    return candidates


# ---------------------------------------------------------------------------
# 结果写入（线程安全）
# ---------------------------------------------------------------------------

class ResultWriter:
    """线程安全的结果写入器，支持定期 flush。"""

    def __init__(
        self,
        output_path: str,
        failed_path: str,
        system_prompt: str,
        output_format: str = "chat",
        save_every: int = 50,
        keep_original: bool = True,
    ):
        self.output_path   = output_path
        self.failed_path   = failed_path
        self.system_prompt = system_prompt
        self.output_format = output_format
        self.save_every    = save_every
        self.keep_original = keep_original

        self._lock      = Lock()
        self._success   : List[Dict] = []
        self._failed    : List[Dict] = []
        self._total_ok  = 0
        self._total_fail= 0

        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def add_success(self, sample: SFTSample) -> None:
        if self.output_format == "chat":
            record = sample.to_chat_dict(self.system_prompt)
        else:
            record = sample.to_alpaca_dict()

        if not self.keep_original:
            record.pop("_meta", None)

        with self._lock:
            self._success.append(record)
            self._total_ok += 1
            if len(self._success) % self.save_every == 0:
                self._flush_success()

    def add_failed(self, item: Dict) -> None:
        with self._lock:
            self._failed.append(item)
            self._total_fail += 1

    def _flush_success(self) -> None:
        """将 _success 追加写入输出文件（持有锁时调用）。"""
        if not self._success:
            return
        with open(self.output_path, "a", encoding="utf-8") as f:
            for record in self._success:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._success.clear()

    def _flush_failed(self) -> None:
        if not self._failed:
            return
        with open(self.failed_path, "a", encoding="utf-8") as f:
            for item in self._failed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._failed.clear()

    def finalize(self) -> Tuple[int, int]:
        """训练结束时调用，写入所有剩余数据，返回 (ok_count, fail_count)。"""
        with self._lock:
            self._flush_success()
            self._flush_failed()
        return self._total_ok, self._total_fail


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT 数据构造（DeepSeek 模拟搜索）")
    parser.add_argument("--config",         default="config/sft_construction_config.yaml")
    parser.add_argument("--target_samples", type=int, default=None,
                        help="目标成功条数：构造到 N 条成功为止（覆盖 config）")
    parser.add_argument("--max_samples",    type=int, default=None,
                        help="最多处理的输入条数上限（覆盖 config，优先级低于 target_samples）")
    parser.add_argument("--start_index",    type=int, default=None,
                        help="从第几条开始处理（断点续处理，shuffle=true 时无效）")
    parser.add_argument("--num_workers",    type=int, default=None, help="并发线程数（覆盖 config）")
    parser.add_argument("--shuffle",        action="store_true", default=None,
                        help="随机打乱输入数据后取样（覆盖 config）")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # ── API 配置 ──────────────────────────────────────────────────────────
    ds_cfg = cfg["deepseek"]
    api_key = os.environ.get("DEEPSEEK_API_KEY") or ds_cfg.get("api_key", "")
    if not api_key or api_key.startswith("sk-xxx"):
        raise ValueError(
            "请设置 DEEPSEEK_API_KEY 环境变量，或在 config 的 deepseek.api_key 填写真实 Key"
        )

    client = DeepSeekClient(
        api_key=api_key,
        base_url=ds_cfg.get("base_url", "https://api.deepseek.com/v1"),
        model=ds_cfg.get("model", "deepseek-chat"),
        temperature=ds_cfg.get("temperature", 0.7),
        max_tokens=ds_cfg.get("max_tokens", 2048),
        max_retries=ds_cfg.get("max_retries", 3),
        retry_delay=ds_cfg.get("retry_delay", 2.0),
        request_interval=ds_cfg.get("request_interval", 0.5),
    )

    # ── 数据配置 ─────────────────────────────────────────────────────────
    data_cfg       = cfg["data"]
    input_path     = data_cfg["input_path"]
    output_path    = data_cfg["output_path"]
    failed_path    = data_cfg.get("failed_path", output_path.replace(".jsonl", "_failed.jsonl"))
    start_index    = args.start_index    if args.start_index    is not None else data_cfg.get("start_index", 0)
    target_samples = args.target_samples if args.target_samples is not None else data_cfg.get("target_samples")
    max_samples    = args.max_samples    if args.max_samples    is not None else data_cfg.get("max_samples")
    shuffle        = args.shuffle        if args.shuffle        is not None else data_cfg.get("shuffle", False)
    shuffle_seed   = data_cfg.get("shuffle_seed", 42)
    dedup          = data_cfg.get("dedup_questions", True)

    # target_samples：加载全部可用数据，依赖取消机制在达标后停止
    # 不再预估成功率（实际成功率因模型/格式而异，难以预判）
    effective_max = max_samples   # None = 全量；有 max_samples 则以它为上限
    if target_samples:
        logger.info(
            f"target_samples={target_samples}，将加载全部可用输入"
            f"{'（上限 ' + str(max_samples) + ' 条）' if max_samples else ''}，"
            f"达标后自动取消剩余任务"
        )

    # ── 并发配置 ─────────────────────────────────────────────────────────
    conc_cfg    = cfg.get("concurrency", {})
    num_workers = args.num_workers if args.num_workers is not None else conc_cfg.get("num_workers", 3)
    save_every  = conc_cfg.get("save_every", 50)

    # ── 其他配置 ─────────────────────────────────────────────────────────
    system_prompt  = cfg["system_prompt"]
    validation_cfg = cfg.get("validation", {})
    output_cfg     = cfg.get("output", {})
    output_format  = output_cfg.get("format", "chat")
    keep_original  = output_cfg.get("keep_original", True)

    # ── 加载数据 ─────────────────────────────────────────────────────────
    items = load_input_data(
        input_path,
        dedup=dedup,
        start_index=start_index,
        max_samples=effective_max,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )
    if not items:
        logger.warning("没有可处理的数据，退出")
        return

    writer = ResultWriter(
        output_path=output_path,
        failed_path=failed_path,
        system_prompt=system_prompt,
        output_format=output_format,
        save_every=save_every,
        keep_original=keep_original,
    )

    logger.info("=" * 60)
    logger.info("SFT 数据构造开始")
    logger.info(f"  输入 : {input_path} ({len(items)} 条)")
    logger.info(f"  输出 : {output_path}")
    logger.info(f"  模型 : {client.model}  并发: {num_workers}")
    if target_samples:
        logger.info(f"  目标 : 成功构造 {target_samples} 条（达成后自动停止）")
    logger.info("=" * 60)

    # ── 并发处理 ─────────────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                process_one, idx, q, a, client, system_prompt, validation_cfg
            ): idx
            for idx, q, a in items
        }

        pbar = tqdm(total=len(futures), desc="构造进度")
        for future in as_completed(futures):
            try:
                sample, _ = future.result()
            except Exception as e:
                idx = futures[future]
                logger.error(f"[{idx}] 未捕获异常: {e}")
                pbar.update(1)
                continue

            if sample is not None:
                writer.add_success(sample)
                # target_samples 达成：取消剩余任务并退出
                if target_samples and writer._total_ok >= target_samples:
                    logger.info(f"已达目标成功条数 {target_samples}，取消剩余任务")
                    for f in futures:
                        f.cancel()
                    pbar.update(1)
                    break
            pbar.update(1)
        pbar.close()

    # ── 收尾 ─────────────────────────────────────────────────────────────
    ok_count, fail_count = writer.finalize()

    total_processed = ok_count + fail_count
    success_rate = ok_count / max(total_processed, 1) * 100
    logger.info("=" * 60)
    logger.info("SFT 数据构造完成！")
    logger.info(f"  成功: {ok_count} 条  →  {output_path}")
    logger.info(f"  处理: {total_processed} 条输入（成功率 {success_rate:.1f}%）")
    if target_samples and ok_count < target_samples:
        logger.warning(
            f"  未达目标 {target_samples} 条（仅成功 {ok_count} 条），"
            f"可用数据已耗尽。建议检查格式校验规则或放宽 validation 参数。"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

#python src/training/scripts/run_sft_construction.py --config config/sft_construction_config.yaml