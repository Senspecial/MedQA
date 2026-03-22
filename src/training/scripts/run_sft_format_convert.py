"""
将 sft_constructed.jsonl（单轮嵌入格式）转换为 Agent-R1 兼容的多轮 ChatML 格式。

原始格式（单条 assistant 消息中嵌套所有步骤）：
    <think>...</think>
    <tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>
    <search_result>...</search_result>
    <think>...</think>
    ...
    <think>...</think>
    <answer>...</answer>

转换后格式（多轮 ChatML，与 NousToolEnv 完全兼容）：
    system: <重构的系统提示>
    user:   <原始问题>
    assistant: <think>...</think>\n<tool_call>...</tool_call>
    user:   <tool_response>...</tool_response>
    assistant: <think>...</think>\n<tool_call>...</tool_call>
    user:   <tool_response>...</tool_response>
    ...
    assistant: <think>...</think>\n<answer>...</answer>

用法：
    python run_sft_format_convert.py [--config ../../config/sft_convert_config.yaml]
    python run_sft_format_convert.py --input ../../data/SFT_Agent/sft_constructed.jsonl \\
                                     --output ../../data/SFT_Agent/sft_agent_r1.jsonl \\
                                     --system-prompt-style agent_r1
"""

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

import yaml


# ─────────────────────────────── 项目根路径 ─────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))


# ─────────────────────────────── 统一提示词加载 ─────────────────────────────

def _load_agent_prompts():
    """从 config/agent_system_prompt.yaml 加载统一 Agent 提示词"""
    yaml_path = os.path.join(_PROJECT_ROOT, "config", "agent_system_prompt.yaml")
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sys_prompt = (cfg.get("agent_system_prompt") or "").strip() or None
        tool_prompt = (cfg.get("agent_tool_format_prompt") or "").strip() or None
        return sys_prompt, tool_prompt
    except Exception:
        return None, None

_loaded_sys, _loaded_tool = _load_agent_prompts()


# ─────────────────────────────── 系统提示模板 ─────────────────────────────────

SYSTEM_PROMPTS = {
    # 最小化：只描述工具调用格式，不含医疗安全规范
    "minimal": """\
你是一个医疗健康助手，可以使用搜索工具查询医学知识，也可以直接根据已有知识给出回答。

**工具调用格式**：
<think>分析问题，决定如何搜索</think>
<tool_call>{"name":"search","arguments":{"query":"搜索关键词"}}</tool_call>

收到搜索结果后，可以继续搜索或给出最终回答：
<think>根据结果分析</think>
<answer>最终回答</answer>
""",

    # Agent-R1 风格：与推理时一致的完整提示（从统一 yaml 加载）
    "agent_r1": _loaded_sys or """\
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
- 有危及生命的症状时，明确建议立即就医
""",

    # 完整版：包含项目 system_prompt.yaml 中的全套医疗安全规范 + 工具调用格式
    "full": None,  # 从 config/system_prompt.yaml 动态加载后追加工具说明
}

TOOL_FORMAT_SUFFIX = _loaded_tool or """\
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


# ─────────────────────────────── 解析逻辑 ─────────────────────────────────────

@dataclass
class Step:
    """一个推理步骤：think + (tool_call + result) 或 (answer)"""
    think: str = ""
    tool_call: Optional[str] = None   # JSON 字符串
    tool_result: Optional[str] = None # 搜索结果内容
    answer: Optional[str] = None      # 最终答案（只有最后一步有）

    @property
    def is_final(self) -> bool:
        return self.answer is not None


def parse_assistant_content(content: str) -> List[Step]:
    """
    将单条 assistant 消息拆解为有序的 Step 列表。

    处理模式：
        <think>T</think> <tool_call>TC</tool_call> <search_result>SR</search_result>
        <think>T</think> <tool_call>TC</tool_call> <search_result>SR</search_result>
        ...
        <think>T</think> <answer>A</answer>
    """
    steps: List[Step] = []
    pos = 0
    text = content.strip()

    # 提取各类标签的正则（允许标签间有任意空白）
    THINK_PAT    = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    TOOL_PAT     = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    RESULT_PAT   = re.compile(r'<search_result>(.*?)</search_result>', re.DOTALL)
    ANSWER_PAT   = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

    while pos < len(text):
        # 1. 找 <think>
        m_think = THINK_PAT.search(text, pos)
        if not m_think:
            break
        think_content = m_think.group(1).strip()
        pos = m_think.end()

        step = Step(think=think_content)

        # 2. 下一个有意义的标签是 <tool_call> 还是 <answer>？
        m_tool   = TOOL_PAT.search(text, pos)
        m_answer = ANSWER_PAT.search(text, pos)

        # 哪个先出现
        tool_start   = m_tool.start()   if m_tool   else len(text)
        answer_start = m_answer.start() if m_answer else len(text)

        if tool_start <= answer_start and m_tool:
            step.tool_call = m_tool.group(1).strip()
            pos = m_tool.end()

            # 3. 紧随其后找 <search_result>
            m_result = RESULT_PAT.search(text, pos)
            if m_result:
                step.tool_result = m_result.group(1).strip()
                pos = m_result.end()

        elif m_answer:
            step.answer = m_answer.group(1).strip()
            pos = m_answer.end()
            steps.append(step)
            break  # 最终回答，结束

        else:
            break  # 无法解析，退出

        steps.append(step)

    return steps


def validate_steps(steps: List[Step]) -> Tuple[bool, str]:
    """检查解析结果是否完整有效。"""
    if not steps:
        return False, "未解析到任何步骤"
    if not steps[-1].is_final:
        return False, "最后一步缺少 <answer>"
    for i, s in enumerate(steps[:-1]):
        if s.tool_call is None:
            return False, f"步骤 {i} 缺少 <tool_call>"
    return True, ""


# ─────────────────────────────── 格式转换 ─────────────────────────────────────

def build_chatml_messages(system_prompt: str, user_question: str,
                          steps: List[Step]) -> List[dict]:
    """
    将解析后的步骤列表转换为 Agent-R1 多轮 ChatML messages。

    每个工具调用步骤拆成：
      assistant: <think>T</think>\n<tool_call>TC</tool_call>
      user:      <tool_response>\nSR\n</tool_response>

    最终步骤：
      assistant: <think>T</think>\n<answer>A</answer>
    """
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_question},
    ]

    for step in steps:
        if step.is_final:
            assistant_content = f"<think>{step.think}</think>\n<answer>{step.answer}</answer>"
            messages.append({"role": "assistant", "content": assistant_content})
        else:
            # 工具调用 turn
            assistant_content = f"<think>{step.think}</think>\n<tool_call>{step.tool_call}</tool_call>"
            messages.append({"role": "assistant", "content": assistant_content})

            # 工具响应 turn（以 user 身份注入，与 NousToolEnv.format_tool_response 一致）
            result_text = step.tool_result or ""
            user_content = f"<tool_response>\n{result_text}\n</tool_response>"
            messages.append({"role": "user", "content": user_content})

    return messages


# ─────────────────────────────── 系统提示构建 ─────────────────────────────────

def build_system_prompt(style: str, base_prompt_path: Optional[str],
                        custom_prompt: Optional[str]) -> str:
    """
    根据 style 构建系统提示。

    style:
        minimal    - 最简洁版本
        agent_r1   - Agent-R1 标准格式说明（推荐 SFT 冷启动）
        full       - 从 system_prompt.yaml 加载完整医疗规范 + 追加工具格式说明
        custom     - 完全使用 custom_prompt 参数
    """
    if style == "custom":
        if not custom_prompt:
            raise ValueError("style=custom 时必须提供 custom_prompt")
        return custom_prompt.strip()

    if style == "full":
        # 从项目配置文件加载完整系统提示
        if base_prompt_path and os.path.exists(base_prompt_path):
            with open(base_prompt_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            base = cfg.get("system_prompt", "").strip()
        else:
            print(f"[WARN] system_prompt.yaml 未找到: {base_prompt_path}，退回 agent_r1 风格")
            return SYSTEM_PROMPTS["agent_r1"].strip()
        return base + "\n\n" + TOOL_FORMAT_SUFFIX.strip()

    prompt = SYSTEM_PROMPTS.get(style)
    if prompt is None:
        raise ValueError(f"未知 style: {style}，可选: {list(SYSTEM_PROMPTS.keys()) + ['custom']}")
    return prompt.strip()


# ─────────────────────────────── 主流程 ──────────────────────────────────────

def convert(input_path: str, output_path: str, system_prompt: str,
            skip_invalid: bool, max_samples: Optional[int]) -> dict:
    """执行格式转换，返回统计信息。"""
    stats = {"total": 0, "success": 0, "skipped_invalid": 0, "skipped_no_steps": 0}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if max_samples and stats["total"] >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_invalid"] += 1
                continue

            messages = record.get("messages", [])
            meta     = record.get("_meta", {})

            # 提取 user 问题（第二条消息）
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            if not user_msg:
                stats["skipped_invalid"] += 1
                continue

            # 提取 assistant 消息（第三条消息）
            asst_msg = next((m for m in messages if m["role"] == "assistant"), None)
            if not asst_msg:
                stats["skipped_invalid"] += 1
                continue

            # 解析 assistant 内容
            steps = parse_assistant_content(asst_msg["content"])

            if not steps:
                stats["skipped_no_steps"] += 1
                if not skip_invalid:
                    print(f"[WARN] 第 {stats['total']} 条未解析到步骤，已跳过")
                continue

            valid, reason = validate_steps(steps)
            if not valid:
                stats["skipped_invalid"] += 1
                if not skip_invalid:
                    print(f"[WARN] 第 {stats['total']} 条格式无效（{reason}），已跳过")
                continue

            # 构建多轮 ChatML
            new_messages = build_chatml_messages(
                system_prompt=system_prompt,
                user_question=user_msg["content"],
                steps=steps,
            )

            out_record = {
                "messages": new_messages,
                "_meta": {
                    **meta,
                    "num_turns": len(steps),
                    "num_tool_calls": sum(1 for s in steps if not s.is_final),
                    "converted": True,
                },
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            stats["success"] += 1

    return stats


# ─────────────────────────────── 入口 ────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../.."))

    parser = argparse.ArgumentParser(description="将 SFT Agent 数据转换为 Agent-R1 多轮 ChatML 格式")
    parser.add_argument(
        "--input", default=os.path.join(project_root, "data/SFT_Agent/sft_constructed.jsonl"),
        help="输入文件路径")
    parser.add_argument(
        "--output", default=os.path.join(project_root, "data/SFT_Agent/sft_agent_r1.jsonl"),
        help="输出文件路径")
    parser.add_argument(
        "--system-prompt-style", default="agent_r1",
        choices=["minimal", "agent_r1", "full", "custom"],
        help="系统提示风格（minimal/agent_r1/full/custom）")
    parser.add_argument(
        "--system-prompt-file",
        default=os.path.join(project_root, "config/system_prompt.yaml"),
        help="system_prompt.yaml 路径（style=full 时使用）")
    parser.add_argument(
        "--custom-prompt", type=str, default=None,
        help="自定义系统提示内容（style=custom 时使用）")
    parser.add_argument(
        "--skip-invalid", action="store_true", default=True,
        help="静默跳过无法解析的条目（默认开启）")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="最多处理的条目数（用于测试）")
    args = parser.parse_args()

    system_prompt = build_system_prompt(
        style=args.system_prompt_style,
        base_prompt_path=args.system_prompt_file,
        custom_prompt=args.custom_prompt,
    )

    print("=" * 60)
    print("SFT 格式转换：单轮嵌入 → Agent-R1 多轮 ChatML")
    print("=" * 60)
    print(f"输入文件:     {args.input}")
    print(f"输出文件:     {args.output}")
    print(f"系统提示风格: {args.system_prompt_style}")
    print(f"最大条目数:   {args.max_samples or '全量'}")
    print("=" * 60)
    print("\n[系统提示预览]")
    print(system_prompt[:300] + ("..." if len(system_prompt) > 300 else ""))
    print()

    stats = convert(
        input_path=args.input,
        output_path=args.output,
        system_prompt=system_prompt,
        skip_invalid=args.skip_invalid,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 60)
    print("转换完成")
    print("=" * 60)
    print(f"总条目数:     {stats['total']}")
    print(f"成功转换:     {stats['success']}")
    print(f"格式无效跳过: {stats['skipped_invalid']}")
    print(f"无步骤跳过:   {stats['skipped_no_steps']}")
    print(f"输出文件:     {args.output}")


if __name__ == "__main__":
    main()

