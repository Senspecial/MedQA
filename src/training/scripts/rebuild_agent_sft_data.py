#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent SFT 数据重构脚本

解决的问题:
  1. 直接回答样本占比过低 (7.7% → 目标 20-25%)
  2. 单次搜索样本过少 (4% → 目标 15%)
  3. 已有的 208 条直接回答数据未合并进训练集
  4. 总样本数偏少 (2296 → 目标 4000+)

执行步骤:
  Step 1: 合并 — 将现有数据 + 直接回答补充数据合并
  Step 2: 分析 — 统计当前分布，计算各类别需要补充的数量
  Step 3: 构造 — 调用 DeepSeek API 补充直接回答 + 单次搜索样本
  Step 4: 拆分 — 按 85/10/5 比例重新拆分 train/valid/test

用法:
    # 仅合并 + 拆分（不调 API，不构造新数据）
    python src/training/scripts/rebuild_agent_sft_data.py --merge-only

    # 完整重构（合并 + 构造 + 拆分）
    python src/training/scripts/rebuild_agent_sft_data.py \
        --config config/sft_direct_answer_config.yaml

    # 指定目标总量
    python src/training/scripts/rebuild_agent_sft_data.py \
        --config config/sft_direct_answer_config.yaml --target-total 4000

    # 试运行（不写文件，仅打印统计）
    python src/training/scripts/rebuild_agent_sft_data.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

THINK_PAT = re.compile(r'<think>(.*?)</think>', re.DOTALL)
TOOL_CALL_PAT = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
ANSWER_PAT = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)


def count_tool_calls(item: dict) -> int:
    """统计一条样本中的搜索次数"""
    meta = item.get("_meta", {})
    if "num_tool_calls" in meta:
        return meta["num_tool_calls"]
    count = 0
    for msg in item.get("messages", []):
        if msg.get("role") == "assistant":
            count += len(TOOL_CALL_PAT.findall(msg.get("content", "")))
    return count


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[dict], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_question(item: dict) -> str:
    """从样本中提取问题文本"""
    for msg in item.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if not content.startswith("<tool_response>"):
                return content.strip()
    return ""


def print_distribution(data: List[dict], title: str = ""):
    """打印数据集的搜索次数分布"""
    dist = Counter(count_tool_calls(item) for item in data)
    total = len(data)
    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
    print(f"  总样本数: {total}")
    for k in sorted(dist.keys()):
        pct = dist[k] / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {k} 次搜索: {dist[k]:5d} ({pct:5.1f}%)  {bar}")
    direct = dist.get(0, 0)
    print(f"  直接回答占比: {direct/total*100:.1f}%")
    single = dist.get(1, 0)
    print(f"  单次搜索占比: {single/total*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: 合并现有数据
# ═══════════════════════════════════════════════════════════════════════════════

def merge_existing_data(
    r1_path: str,
    direct_answer_path: str,
    extra_paths: Optional[List[str]] = None,
) -> List[dict]:
    """合并所有现有 Agent SFT 数据，按 question 去重"""
    all_data: List[dict] = []
    seen_questions: set = set()

    def add_file(path: str, label: str):
        if not os.path.exists(path):
            print(f"  [跳过] {label}: {path} 不存在")
            return 0
        data = load_jsonl(path)
        added = 0
        for item in data:
            q = get_question(item)
            if q and q not in seen_questions:
                seen_questions.add(q)
                all_data.append(item)
                added += 1
        print(f"  [合并] {label}: 加载 {len(data)}, 去重后新增 {added}")
        return added

    print("\n" + "=" * 60)
    print("Step 1: 合并现有数据")
    print("=" * 60)

    add_file(r1_path, "Agent R1 主数据")
    add_file(direct_answer_path, "直接回答补充数据")

    if extra_paths:
        for p in extra_paths:
            add_file(p, f"额外数据 ({os.path.basename(p)})")

    print(f"\n  合并后总计: {len(all_data)} 条（去重）")
    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: 分析并计算补充需求
# ═══════════════════════════════════════════════════════════════════════════════

def compute_supplement_plan(
    data: List[dict],
    target_total: int = 4000,
    target_direct_ratio: float = 0.22,
    target_single_ratio: float = 0.13,
) -> Dict[str, int]:
    """
    计算需要补充的各类别数量。

    Returns:
        {"direct_answer": N, "single_search": M}
    """
    current_total = len(data)
    dist = Counter(count_tool_calls(item) for item in data)

    current_direct = dist.get(0, 0)
    current_single = dist.get(1, 0)

    print("\n" + "=" * 60)
    print("Step 2: 分析补充需求")
    print("=" * 60)
    print_distribution(data, "当前分布")

    need_direct = max(0, int(target_total * target_direct_ratio) - current_direct)
    need_single = max(0, int(target_total * target_single_ratio) - current_single)

    remaining = target_total - current_total - need_direct - need_single
    if remaining < 0:
        scale = (target_total - current_total) / max(need_direct + need_single, 1)
        need_direct = int(need_direct * scale)
        need_single = int(need_single * scale)

    plan = {"direct_answer": need_direct, "single_search": need_single}

    print(f"\n  目标总量: {target_total}")
    print(f"  目标直接回答占比: {target_direct_ratio:.0%}")
    print(f"  目标单次搜索占比: {target_single_ratio:.0%}")
    print(f"  需补充直接回答: {need_direct} 条")
    print(f"  需补充单次搜索: {need_single} 条")
    print(f"  补充后预计总量: {current_total + need_direct + need_single}")

    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: 构造补充数据
# ═══════════════════════════════════════════════════════════════════════════════

def _load_agent_system_prompt() -> str:
    yaml_path = os.path.join(_PROJECT_ROOT, "config", "agent_system_prompt.yaml")
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return (cfg.get("agent_system_prompt") or "").strip()
    except Exception:
        return ""


AGENT_SYSTEM_PROMPT = _load_agent_system_prompt() or (
    "你是一个专业的医疗健康信息助手，具备搜索医学知识的能力，也可以直接根据已有知识给出回答。"
)


# ── 直接回答构造 ──

DIRECT_ANSWER_SYSTEM = """\
你是一个训练数据生成助手。用户会给你一个医学问题和参考答案。
请你用以下格式生成一个"不需要搜索、直接回答"的样本：

<think>简要分析问题，说明这是可以直接根据已有知识回答的常见问题</think>
<answer>专业、完整、安全的医疗回答（参考但不要照搬参考答案，需重新组织语言，加入安全提示）</answer>

要求：
- <think> 中说明为什么可以直接回答（不需要搜索）
- <answer> 不少于 50 字，要专业但通俗
- 使用不确定表述（可能、考虑、建议）
- 不做明确诊断，不开处方"""


def _validate_direct_answer(raw: str) -> Optional[Tuple[str, str]]:
    """
    校验直接回答格式，返回 (think_text, answer_text) 或 None。

    检查项:
      1. 必须有且仅有一对 <think>...</think>
      2. 必须有且仅有一对 <answer>...</answer>
      3. 不得包含 <tool_call> （直接回答不能搜索）
      4. think 和 answer 内容不能为空
      5. answer 长度 >= 20 字符
    """
    if "<tool_call>" in raw:
        return None

    think_matches = THINK_PAT.findall(raw)
    answer_matches = ANSWER_PAT.findall(raw)

    if len(think_matches) != 1 or len(answer_matches) != 1:
        return None

    think_text = think_matches[0].strip()
    answer_text = answer_matches[0].strip()

    if not think_text or len(answer_text) < 20:
        return None

    return think_text, answer_text


def construct_direct_answer(
    idx: int, question: str, raw_answer: str, client: Any,
) -> Optional[dict]:
    """构造一条直接回答样本"""
    try:
        raw = client.chat([
            {"role": "system", "content": DIRECT_ANSWER_SYSTEM},
            {"role": "user", "content": f"问题：{question}\n参考答案：{raw_answer[:300]}"},
        ])
    except Exception as e:
        return None

    result = _validate_direct_answer(raw)
    if result is None:
        return None

    think_text, answer_text = result

    return {
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"<think>{think_text}</think>\n<answer>{answer_text}</answer>"},
        ],
        "_meta": {"index": idx, "num_tool_calls": 0, "num_turns": 1, "type": "direct_answer_补充"},
    }


# ── 单次搜索构造 ──

SINGLE_SEARCH_SYSTEM = """\
你是一个训练数据生成助手。用户会给你一个医学问题和参考答案。
请你用以下格式生成一个"搜索 1 次就给出回答"的样本：

<think>分析问题，明确需要搜索什么</think>
<tool_call>{{"name":"search","arguments":{{"query":"精准的搜索关键词"}}}}</tool_call>
<search_result>基于参考答案模拟的搜索结果（100-300字，要像真实检索结果）</search_result>
<think>根据搜索结果进行分析，综合给出回答</think>
<answer>专业、完整、安全的医疗回答</answer>

要求：
- 只搜索 1 次，搜索关键词要精准
- search_result 要像真实知识库返回的内容
- answer 不少于 80 字
- 使用不确定表述"""


def _validate_tool_call_json(tc_str: str) -> bool:
    """校验 tool_call 内容是合法的 search JSON"""
    try:
        obj = json.loads(tc_str)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(obj, dict):
        return False
    if obj.get("name") != "search":
        return False
    args = obj.get("arguments")
    if not isinstance(args, dict):
        return False
    query = args.get("query", "")
    return isinstance(query, str) and len(query.strip()) > 0


def construct_single_search(
    idx: int, question: str, raw_answer: str, client: Any,
) -> Optional[dict]:
    """构造一条单次搜索样本"""
    try:
        raw = client.chat([
            {"role": "system", "content": SINGLE_SEARCH_SYSTEM},
            {"role": "user", "content": f"问题：{question}\n参考答案：{raw_answer[:500]}"},
        ])
    except Exception as e:
        return None

    from src.training.scripts.run_sft_format_convert import parse_assistant_content, validate_steps, build_chatml_messages

    steps = parse_assistant_content(raw)
    valid, _ = validate_steps(steps)
    if not valid:
        return None

    tool_count = sum(1 for s in steps if not s.is_final)
    if tool_count != 1:
        return None

    tool_step = next(s for s in steps if not s.is_final)

    if not _validate_tool_call_json(tool_step.tool_call):
        return None

    if not tool_step.tool_result or len(tool_step.tool_result.strip()) < 20:
        return None

    final_step = steps[-1]
    if not final_step.think or not final_step.answer or len(final_step.answer.strip()) < 20:
        return None

    messages = build_chatml_messages(AGENT_SYSTEM_PROMPT, question, steps)

    return {
        "messages": messages,
        "_meta": {"index": idx, "num_tool_calls": 1, "num_turns": 2, "type": "single_search_补充"},
    }


def construct_supplement(
    plan: Dict[str, int],
    source_data_path: str,
    existing_questions: set,
    api_config: dict,
    num_workers: int = 3,
) -> List[dict]:
    """调用 DeepSeek API 批量构造补充数据"""
    import requests

    class SimpleClient:
        def __init__(self, cfg):
            self.api_key = cfg.get("api_key", "")
            self.base_url = cfg.get("base_url", "https://api.deepseek.com/v1").rstrip("/")
            self.model = cfg.get("model", "deepseek-chat")
            self.temperature = cfg.get("temperature", 0.8)
            self.max_tokens = cfg.get("max_tokens", 4096)
            self.max_retries = cfg.get("max_retries", 3)
            self.interval = cfg.get("request_interval", 0.5)
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

        def chat(self, messages):
            for attempt in range(1, self.max_retries + 1):
                try:
                    time.sleep(self.interval)
                    resp = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json={"model": self.model, "messages": messages,
                              "temperature": self.temperature, "max_tokens": self.max_tokens},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    if attempt == self.max_retries:
                        raise
                    time.sleep(2 ** attempt)

    client = SimpleClient(api_config)

    # 加载源数据并去重
    print(f"\n  加载源数据: {source_data_path}")
    with open(source_data_path, "r", encoding="utf-8") as f:
        source = json.load(f)

    seen_in_source: set = set()
    candidates = []
    dup_in_source = 0
    dup_with_existing = 0

    for i, item in enumerate(source):
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if not q or not a:
            continue
        if q in existing_questions:
            dup_with_existing += 1
            continue
        if q in seen_in_source:
            dup_in_source += 1
            continue
        seen_in_source.add(q)
        candidates.append((i, q, a))

    print(f"  源数据总条数: {len(source)}")
    print(f"  与训练集重复（跳过）: {dup_with_existing}")
    print(f"  源数据内部重复（跳过）: {dup_in_source}")
    print(f"  可用候选: {len(candidates)}")

    random.shuffle(candidates)

    results: List[dict] = []
    need_direct = plan.get("direct_answer", 0)
    need_single = plan.get("single_search", 0)

    print(f"\n" + "=" * 60)
    print("Step 3: 构造补充数据")
    print("=" * 60)

    # ── 构造直接回答 ──
    if need_direct > 0:
        # 选择简短问题（更适合直接回答）
        short_candidates = sorted(candidates, key=lambda x: len(x[1]) + len(x[2]))
        direct_pool = short_candidates[:need_direct * 3]

        print(f"\n  构造直接回答: 目标 {need_direct} 条 (候选池 {len(direct_pool)} 条)")
        direct_ok = 0

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(construct_direct_answer, idx, q, a, client): idx
                for idx, q, a in direct_pool
            }
            pbar = tqdm(total=len(futures), desc="  直接回答")
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        direct_ok += 1
                        if direct_ok >= need_direct:
                            for f in futures:
                                f.cancel()
                            pbar.update(1)
                            break
                except Exception:
                    pass
                pbar.update(1)
            pbar.close()

        print(f"  直接回答: 成功 {direct_ok} / 目标 {need_direct}")
        used_indices = {r["_meta"]["index"] for r in results}
        candidates = [(i, q, a) for i, q, a in candidates if i not in used_indices]

    # ── 构造单次搜索 ──
    if need_single > 0:
        single_pool = candidates[:need_single * 3]

        print(f"\n  构造单次搜索: 目标 {need_single} 条 (候选池 {len(single_pool)} 条)")
        single_ok = 0

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(construct_single_search, idx, q, a, client): idx
                for idx, q, a in single_pool
            }
            pbar = tqdm(total=len(futures), desc="  单次搜索")
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        single_ok += 1
                        if single_ok >= need_single:
                            for f in futures:
                                f.cancel()
                            pbar.update(1)
                            break
                except Exception:
                    pass
                pbar.update(1)
            pbar.close()

        print(f"  单次搜索: 成功 {single_ok} / 目标 {need_single}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: 拆分
# ═══════════════════════════════════════════════════════════════════════════════

_XML_TC_PAT = re.compile(
    r'<name>\s*search\s*</name>\s*<arguments>\s*<query"?>(.*?)</query>\s*</arguments>',
    re.DOTALL,
)

_NAME_PREFIX_PAT = re.compile(r'<name>\s*search\s*</name>\s*')


def _repair_content(content: str) -> str:
    """修复单条 assistant 消息中的常见格式问题"""

    # 1) 修复多层嵌套 <tool_call>（循环处理直到稳定）
    while "<tool_call><tool_call>" in content or "<tool_call>\n<tool_call>" in content:
        content = content.replace("<tool_call><tool_call>", "<tool_call>")
        content = content.replace("<tool_call>\n<tool_call>", "<tool_call>")
        content = content.replace("</tool_call></tool_call>", "</tool_call>")
        content = content.replace("</tool_call>\n</tool_call>", "</tool_call>")

    # 2) 修复纯 XML 风格 tool_call → JSON（含 <query"> 拼写错误）
    m = _XML_TC_PAT.search(content)
    if m:
        query = m.group(1).strip()
        json_tc = json.dumps({"name": "search", "arguments": {"query": query}}, ensure_ascii=False)
        content = content[:m.start()] + json_tc + content[m.end():]

    # 3) 修复混合格式: <tool_call> 内有 <name>search</name> 前缀 + JSON
    #    e.g. <tool_call><name>search</name>\n{"name":"search",...}</tool_call>
    tc_tag_pat = re.compile(r'(<tool_call>)(.*?)(</tool_call>)', re.DOTALL)
    def _strip_name_prefix(m: re.Match) -> str:
        inner = m.group(2)
        if '<name>' in inner and '{"name"' in inner:
            inner = _NAME_PREFIX_PAT.sub("", inner).strip()
        return f"{m.group(1)}{inner}{m.group(3)}"
    content = tc_tag_pat.sub(_strip_name_prefix, content)

    return content


def repair_sample(item: dict) -> dict:
    """
    尝试修复常见格式问题:
      - 多层嵌套 <tool_call>
      - XML 风格 tool_call → JSON
    返回修复后的样本（原样本不变）。
    """
    import copy
    item = copy.deepcopy(item)
    for msg in item.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        repaired = _repair_content(content)
        if repaired != content:
            msg["content"] = repaired
    return item


def validate_sample(item: dict) -> Tuple[bool, str]:
    """
    最终写入前的格式校验，确保每条样本符合 Agent SFT ChatML 规范。

    检查项:
      1. messages 非空，且至少有 system + user + assistant 三条消息
      2. 第一条必须是 system
      3. 第二条必须是 user（问题）
      4. 最后一条必须是 assistant 且包含 <answer>
      5. assistant 消息中的 <tool_call> JSON 格式合法
      6. 工具调用轮次: assistant(tool_call) 后必须跟 user(tool_response)
      7. 没有连续的同 role 消息
    """
    msgs = item.get("messages", [])
    if len(msgs) < 3:
        return False, f"消息数不足: {len(msgs)}"

    if msgs[0].get("role") != "system":
        return False, "首条消息不是 system"

    if msgs[1].get("role") != "user":
        return False, "第二条消息不是 user"

    last_asst = None
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            last_asst = m
            break
    if last_asst is None:
        return False, "没有 assistant 消息"

    if "<answer>" not in last_asst.get("content", ""):
        return False, "最后一条 assistant 消息缺少 <answer>"

    prev_role = None
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = m.get("content", "")

        if not content.strip():
            return False, f"第 {i} 条消息内容为空 (role={role})"

        if role == prev_role and role != "user":
            return False, f"第 {i} 条与前一条 role 连续重复: {role}"
        prev_role = role

        if role == "assistant" and "<tool_call>" in content:
            tc_match = TOOL_CALL_PAT.search(content)
            if tc_match and not _validate_tool_call_json(tc_match.group(1)):
                return False, f"第 {i} 条 assistant 的 tool_call JSON 格式错误"

            if i + 1 < len(msgs):
                next_msg = msgs[i + 1]
                if next_msg.get("role") != "user" or "<tool_response>" not in next_msg.get("content", ""):
                    return False, f"第 {i} 条 tool_call 后未跟 tool_response"

    return True, ""


def stratified_split(
    data: List[dict],
    train_ratio: float = 0.85,
    valid_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    按搜索次数分层拆分，确保各类别在 train/valid/test 中分布均匀。
    """
    rng = random.Random(seed)

    by_tc: Dict[int, List[dict]] = {}
    for item in data:
        tc = count_tool_calls(item)
        by_tc.setdefault(tc, []).append(item)

    train, valid, test = [], [], []

    for tc, items in sorted(by_tc.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_valid = max(1, int(n * valid_ratio)) if n > 2 else 0
        n_test = n - n_train - n_valid

        if n_test < 0:
            n_valid = n - n_train
            n_test = 0

        train.extend(items[:n_train])
        valid.extend(items[n_train:n_train + n_valid])
        test.extend(items[n_train + n_valid:])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)

    return train, valid, test


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Agent SFT 数据重构")
    parser.add_argument("--config", type=str, default=None,
                        help="DeepSeek API 配置 (sft_direct_answer_config.yaml)，"
                             "不提供则跳过构造步骤")
    parser.add_argument("--merge-only", action="store_true",
                        help="仅合并 + 拆分，不调 API 构造新数据")
    parser.add_argument("--dry-run", action="store_true",
                        help="试运行，仅打印统计，不写文件")
    parser.add_argument("--target-total", type=int, default=4000,
                        help="目标总样本数 (默认 4000)")
    parser.add_argument("--target-direct-ratio", type=float, default=0.22,
                        help="目标直接回答占比 (默认 0.22)")
    parser.add_argument("--target-single-ratio", type=float, default=0.13,
                        help="目标单次搜索占比 (默认 0.13)")
    parser.add_argument("--num-workers", type=int, default=3,
                        help="API 并发线程数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.join(_PROJECT_ROOT, "data", "SFT_Agent")
    r1_path = os.path.join(data_dir, "sft_agent_r1.jsonl")
    direct_path = os.path.join(data_dir, "sft_direct_answer_deepseek.jsonl")
    source_path = os.path.join(data_dir, "merged_data.json")

    output_train = os.path.join(data_dir, "sft_agent_r1_train.jsonl")
    output_valid = os.path.join(data_dir, "sft_agent_r1_valid.jsonl")
    output_test = os.path.join(data_dir, "sft_agent_r1_test.jsonl")

    # ── Step 1: 合并 ──
    all_data = merge_existing_data(r1_path, direct_path)

    if args.dry_run or args.merge_only:
        # 仅分析，不构造
        plan = compute_supplement_plan(
            all_data,
            target_total=args.target_total,
            target_direct_ratio=args.target_direct_ratio,
            target_single_ratio=args.target_single_ratio,
        )

        if args.dry_run:
            print("\n[Dry Run] 不写入文件")
            return

    else:
        # ── Step 2: 分析 ──
        plan = compute_supplement_plan(
            all_data,
            target_total=args.target_total,
            target_direct_ratio=args.target_direct_ratio,
            target_single_ratio=args.target_single_ratio,
        )

        total_needed = plan["direct_answer"] + plan["single_search"]
        if total_needed > 0:
            # ── Step 3: 构造 ──
            api_cfg_path = args.config
            if not api_cfg_path:
                api_cfg_path = os.path.join(_PROJECT_ROOT, "config", "sft_direct_answer_config.yaml")

            if not os.path.isabs(api_cfg_path):
                api_cfg_path = os.path.join(_PROJECT_ROOT, api_cfg_path)

            if not os.path.exists(api_cfg_path):
                print(f"\n[Error] API 配置文件不存在: {api_cfg_path}")
                print("请提供 --config 参数，或使用 --merge-only 仅合并")
                sys.exit(1)

            with open(api_cfg_path, "r", encoding="utf-8") as f:
                api_full_cfg = yaml.safe_load(f)

            api_cfg = api_full_cfg.get("deepseek", {})
            api_key = os.environ.get("DEEPSEEK_API_KEY") or api_cfg.get("api_key", "")
            if not api_key or api_key.startswith("sk-xxx"):
                print("\n[Error] 未配置 API key，请设置 DEEPSEEK_API_KEY 或在配置文件中填写")
                sys.exit(1)
            api_cfg["api_key"] = api_key

            existing_questions = {get_question(item) for item in all_data}

            supplement = construct_supplement(
                plan=plan,
                source_data_path=source_path,
                existing_questions=existing_questions,
                api_config=api_cfg,
                num_workers=args.num_workers,
            )
            all_data.extend(supplement)
        else:
            print("\n  当前数据已满足目标分布，无需补充")

    # ── Step 3.5: 修复 + 格式校验 ──
    print("\n" + "=" * 60)
    print("Step 3.5: 修复 + 格式校验")
    print("=" * 60)

    repaired_count = 0
    valid_data = []
    invalid_count = 0
    invalid_reasons: Counter = Counter()

    for item in all_data:
        repaired = repair_sample(item)
        if repaired != item:
            repaired_count += 1
        ok, reason = validate_sample(repaired)
        if ok:
            valid_data.append(repaired)
        else:
            invalid_count += 1
            invalid_reasons[reason] += 1

    if repaired_count > 0:
        print(f"  自动修复: {repaired_count} 条（双层嵌套 tool_call 等）")
    if invalid_count > 0:
        print(f"  过滤不合格样本: {invalid_count} 条")
        for reason, cnt in invalid_reasons.most_common(10):
            print(f"    - {reason}: {cnt}")
    print(f"  通过校验: {len(valid_data)} / {len(all_data)}")

    all_data = valid_data

    # ── Step 4: 拆分 ──
    print("\n" + "=" * 60)
    print("Step 4: 分层拆分 (85/10/5)")
    print("=" * 60)

    train, valid, test = stratified_split(all_data, seed=args.seed)

    print(f"  Train: {len(train)} 条")
    print(f"  Valid: {len(valid)} 条")
    print(f"  Test : {len(test)} 条")

    print_distribution(train, "Train 分布")
    print_distribution(valid, "Valid 分布")
    print_distribution(test, "Test 分布")

    # ── 写入 ──
    # 先备份原文件
    import shutil
    backup_dir = os.path.join(data_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    for p in [output_train, output_valid, output_test]:
        if os.path.exists(p):
            backup_name = os.path.basename(p) + ".bak"
            shutil.copy2(p, os.path.join(backup_dir, backup_name))

    save_jsonl(train, output_train)
    save_jsonl(valid, output_valid)
    save_jsonl(test, output_test)

    print(f"\n" + "=" * 60)
    print("重构完成！")
    print("=" * 60)
    print(f"  Train: {output_train} ({len(train)} 条)")
    print(f"  Valid: {output_valid} ({len(valid)} 条)")
    print(f"  Test : {output_test}  ({len(test)} 条)")
    print(f"  备份: {backup_dir}/")
    print(f"\n下一步:")
    print(f"  python src/training/scripts/run_sft.py --config config/agent_sft_config.yaml")
    print()


if __name__ == "__main__":
    main()
