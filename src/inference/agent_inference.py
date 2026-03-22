#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
医学 Agent 推理：模型 + 本地知识库搜索 + 多轮工具调用循环

Agent 交互范式（与 Agent SFT 训练数据一致）：
    system:    系统提示（含工具调用格式说明）
    user:      用户问题
    assistant: <think>...</think>\n<tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>
    user:      <tool_response>\n...\n</tool_response>
    assistant: <think>...</think>\n<tool_call>...</tool_call>
    user:      <tool_response>...\n</tool_response>
    ...
    assistant: <think>...</think>\n<answer>最终回答</answer>

用法：
    # 交互式
    python src/inference/agent_inference.py --config config/agent_inference_config.yaml

    # 单问题
    python src/inference/agent_inference.py --config config/agent_inference_config.yaml \
        --mode single --question "糖尿病的早期症状有哪些？"
"""

import json
import os
import re
import sys
import time
import yaml
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ─────────────────────────── Agent 系统提示 ───────────────────────────────────
# 从 config/agent_system_prompt.yaml 统一加载，确保训练-推理一致

def _load_agent_prompts():
    """从统一配置加载 Agent 提示词，返回 (system_prompt, tool_format_prompt)"""
    yaml_path = os.path.join(project_root, "config", "agent_system_prompt.yaml")
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sys_prompt = (cfg.get("agent_system_prompt") or "").strip() or None
        tool_prompt = (cfg.get("agent_tool_format_prompt") or "").strip() or None
        return sys_prompt, tool_prompt
    except Exception:
        return None, None

_loaded_sys, _loaded_tool = _load_agent_prompts()

AGENT_TOOL_FORMAT_PROMPT = _loaded_tool or """\
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

AGENT_SYSTEM_PROMPT = _loaded_sys or (
    "你是一个专业的医疗健康信息助手，具备搜索医学知识的能力，也可以直接根据已有知识给出回答。\n\n"
    + AGENT_TOOL_FORMAT_PROMPT
)


# ─────────────────────────── 解析工具 ─────────────────────────────────────────

TOOL_CALL_PAT = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
ANSWER_PAT = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
THINK_PAT = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def parse_response(text: str) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    """
    解析模型单轮输出，返回 (tool_call_dict, answer_str, think_str)。
    优先级：tool_call > <answer>标签 > </think>后的裸文本。
    """
    think_match = THINK_PAT.search(text)
    think_str = think_match.group(1).strip() if think_match else None

    tool_match = TOOL_CALL_PAT.search(text)
    if tool_match:
        try:
            tc = json.loads(tool_match.group(1).strip())
            return tc, None, think_str
        except json.JSONDecodeError:
            pass

    answer_match = ANSWER_PAT.search(text)
    if answer_match:
        return None, answer_match.group(1).strip(), think_str

    if '</think>' in text:
        after_think = text.split('</think>', 1)[-1].strip()
        if after_think:
            return None, after_think, think_str

    return None, None, think_str


# ─────────────────────────── 知识库搜索工具 ───────────────────────────────────

def load_search_tool(kb_dir: str, device: str = "cpu", top_k: int = 5):
    """加载本地医学知识库搜索工具"""
    sys.path.insert(0, str(project_root))
    from kb.kb_tool import MedSearchTool
    return MedSearchTool(
        config_path=os.path.join(kb_dir, "config.yaml"),
        device=device,
        top_k=top_k,
    )


def execute_tool(tool_call: dict, search_tool,
                 seen_titles: Optional[set] = None,
                 think_text: Optional[str] = None) -> str:
    """执行工具调用，返回格式化结果字符串。支持跨轮去重和 HyDE。"""
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})

    if think_text and name == "search":
        args = dict(args)
        args["hyde_text"] = think_text

    if name == "search" and search_tool is not None:
        result = search_tool.execute(args)
        content = result["content"]
        try:
            data = json.loads(content)
            results = data.get("results", [])
            if not results:
                return "未找到相关医学信息。请尝试换用不同的关键词重新搜索。"

            if seen_titles is not None:
                results = [r for r in results
                           if r.get("title", "") not in seen_titles]
                if not results:
                    return ("搜索结果与之前的搜索重复，未找到新信息。"
                            "请尝试使用不同角度的关键词搜索。")

            lines = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                body = r.get("content", "")[:400]
                score = r.get("score", 0)

                if seen_titles is not None:
                    seen_titles.add(title)

                score_tag = f" (相关度: {score:.2f})" if score else ""
                lines.append(f"[{i}]{score_tag} {title}\n{body}")
            return "\n\n".join(lines)
        except json.JSONDecodeError:
            return content
    return f"Error: 未知工具 '{name}'"


# ─────────────────────────── Agent 推理核心 ───────────────────────────────────

class MedicalAgentInference:
    """医学 Agent 推理器：模型 + 知识库 + 多轮工具调用"""

    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        is_lora: bool = False,
        system_prompt: Optional[str] = None,
        device: str = "cuda",
        load_in_4bit: bool = False,
        kb_dir: Optional[str] = None,
        kb_device: str = "cpu",
        kb_top_k: int = 5,
        max_turns: int = 5,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.system_prompt = system_prompt or AGENT_SYSTEM_PROMPT

        self._load_model(model_path, base_model_path, is_lora, load_in_4bit)
        self._load_kb(kb_dir, kb_device, kb_top_k)

    def _load_model(self, model_path: str, base_model_path: Optional[str],
                    is_lora: bool, load_in_4bit: bool):
        """加载模型和 tokenizer"""
        print("=" * 60)
        print("加载 Agent 模型")
        print("=" * 60)

        tokenizer_path = base_model_path if is_lora else model_path
        print(f"  Tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, padding_side='left',
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
        }
        if load_in_4bit:
            load_kwargs['load_in_4bit'] = True
        else:
            load_kwargs['torch_dtype'] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            if self.device == "cuda":
                load_kwargs['device_map'] = 'auto'

        if is_lora:
            print(f"  Base model: {base_model_path}")
            base = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
            print(f"  LoRA adapter: {model_path}")
            model = PeftModel.from_pretrained(base, model_path)
            print("  Merging LoRA weights...")
            self.model = model.merge_and_unload()
        else:
            print(f"  Model: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        self.model.eval()
        print("  Model loaded.\n")

    def _load_kb(self, kb_dir: Optional[str], kb_device: str, top_k: int):
        """加载知识库搜索工具"""
        if kb_dir is None:
            kb_dir = os.path.join(project_root, "kb")

        if not os.path.isdir(kb_dir):
            print(f"[WARN] 知识库目录不存在: {kb_dir}，Agent 将不使用搜索工具")
            self.search_tool = None
            return

        index_bin = os.path.join(kb_dir, "index", "index.bin")
        if not os.path.exists(index_bin):
            print(f"[WARN] FAISS 索引不存在: {index_bin}，请先运行 python kb/build_kb.py")
            self.search_tool = None
            return

        print("=" * 60)
        print("加载医学知识库")
        print("=" * 60)
        self.search_tool = load_search_tool(kb_dir, device=kb_device, top_k=top_k)
        print()

    # ─────────────────── 生成 ────────────────────

    def _generate_one_turn(self, messages: List[dict]) -> str:
        """基于当前 messages 生成一轮 assistant 回复"""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=False).strip()
        for tag in ('<|im_end|>', '<|endoftext|>', '<|im_start|>'):
            response = response.replace(tag, '')
        return response.strip()

    # ─────────────────── Agent 循环 ────────────────────

    def run(self, question: str, verbose: bool = True) -> dict:
        """
        运行完整的 Agent 推理循环。

        Returns:
            {
                "question": str,
                "answer": str,
                "turns": [{"role": ..., "content": ..., "tool_call": ..., "tool_response": ...}, ...],
                "num_tool_calls": int,
                "elapsed_sec": float,
            }
        """
        t0 = time.time()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        turns = []
        num_tool_calls = 0
        final_answer = None
        seen_titles: set = set()

        for turn_i in range(self.max_turns):
            response = self._generate_one_turn(messages)
            tool_call, answer, think = parse_response(response)

            if verbose:
                self._print_turn(turn_i, think, tool_call, answer, response)

            if tool_call is not None:
                num_tool_calls += 1
                tool_result = execute_tool(
                    tool_call, self.search_tool, seen_titles, think)

                if verbose:
                    self._print_tool_result(tool_result)

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{tool_result}\n</tool_response>",
                })
                turns.append({
                    "role": "assistant", "content": response,
                    "tool_call": tool_call, "tool_response": tool_result,
                })

            elif answer is not None:
                final_answer = answer
                messages.append({"role": "assistant", "content": response})
                turns.append({"role": "assistant", "content": response, "answer": answer})
                break

            else:
                final_answer = response
                messages.append({"role": "assistant", "content": response})
                turns.append({"role": "assistant", "content": response, "answer": response})
                break

        if final_answer is None:
            final_answer = "抱歉，经过多轮搜索仍未能得出明确结论，建议您前往医院就诊获取专业意见。"

        elapsed = time.time() - t0
        return {
            "question": question,
            "answer": final_answer,
            "turns": turns,
            "num_tool_calls": num_tool_calls,
            "elapsed_sec": round(elapsed, 2),
        }

    # ─────────────────── 打印 ────────────────────

    @staticmethod
    def _print_turn(turn_i, think, tool_call, answer, raw):
        C_THINK = "\033[90m"
        C_TOOL = "\033[1;35m"
        C_ANSWER = "\033[1;32m"
        C_RESET = "\033[0m"
        C_DIM = "\033[2m"

        print(f"\n{'─' * 50} Turn {turn_i + 1} {'─' * 50}")
        if think:
            print(f"{C_THINK}[Think] {think}{C_RESET}")
        if tool_call:
            query = tool_call.get("arguments", {}).get("query", "")
            print(f"{C_TOOL}[Search] {query}{C_RESET}")
        elif answer:
            print(f"{C_ANSWER}[Answer]{C_RESET}")
            print(answer)
        else:
            print(f"{C_DIM}[Raw] {raw[:200]}{C_RESET}")

    @staticmethod
    def _print_tool_result(result: str):
        C_RESULT = "\033[33m"
        C_RESET = "\033[0m"
        preview = result[:300] + ("..." if len(result) > 300 else "")
        print(f"{C_RESULT}[Result] {preview}{C_RESET}")

    # ─────────────────── 交互式 ────────────────────

    def interactive_chat(self):
        """交互式 Agent 对话"""
        print("\n" + "=" * 60)
        print("  医学 Agent 交互式对话")
        print("  输入问题开始对话，输入 quit / exit 退出")
        print("=" * 60)

        while True:
            try:
                question = input("\n\033[1;34m[User]\033[0m ").strip()
                if question.lower() in ('quit', 'exit', 'q', '退出'):
                    print("再见！")
                    break
                if not question:
                    continue

                result = self.run(question, verbose=True)
                print(f"\n\033[2m({result['num_tool_calls']} searches, "
                      f"{result['elapsed_sec']}s)\033[0m")

            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"\n\033[31m[Error] {e}\033[0m")

    # ─────────────────── 批量推理 ────────────────────

    def batch_run(self, questions: List[str], verbose: bool = False) -> List[dict]:
        """批量运行 Agent"""
        from tqdm import tqdm
        results = []
        for q in tqdm(questions, desc="Agent 推理"):
            results.append(self.run(q, verbose=verbose))
        return results


# ─────────────────────────── 配置加载 ─────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_agent_from_config(config: dict) -> MedicalAgentInference:
    """根据配置字典构建 Agent 推理器"""
    mc = config['model']
    gc = config['generation']
    ac = config.get('agent', {})
    kc = config.get('knowledge_base', {})

    system_prompt = config.get('system_prompt')
    if isinstance(system_prompt, str) and system_prompt.endswith('.yaml'):
        sp_path = system_prompt
        if not os.path.isabs(sp_path):
            sp_path = os.path.join(project_root, sp_path)
        if os.path.exists(sp_path):
            with open(sp_path, 'r', encoding='utf-8') as f:
                sp_cfg = yaml.safe_load(f)
            base_prompt = sp_cfg.get('system_prompt', '')
            system_prompt = base_prompt.rstrip() + "\n\n" + AGENT_TOOL_FORMAT_PROMPT
        else:
            system_prompt = None

    kb_dir = kc.get('kb_dir')
    if kb_dir and not os.path.isabs(kb_dir):
        kb_dir = os.path.join(project_root, kb_dir)

    model_path = mc['model_path']
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    base_model_path = mc.get('base_model_path')
    if base_model_path and not os.path.isabs(base_model_path):
        base_model_path = os.path.join(project_root, base_model_path)

    return MedicalAgentInference(
        model_path=model_path,
        base_model_path=base_model_path,
        is_lora=mc.get('is_lora', False),
        system_prompt=system_prompt,
        device=mc.get('device', 'cuda'),
        load_in_4bit=mc.get('load_in_4bit', False),
        kb_dir=kb_dir,
        kb_device=kc.get('device', 'cpu'),
        kb_top_k=kc.get('top_k', 5),
        max_turns=ac.get('max_turns', 5),
        max_new_tokens=gc.get('max_new_tokens', 1024),
        temperature=gc.get('temperature', 0.7),
        top_p=gc.get('top_p', 0.9),
        repetition_penalty=gc.get('repetition_penalty', 1.0),
    )


# ─────────────────────────── CLI 入口 ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="医学 Agent 推理")
    parser.add_argument("--config", type=str, default="config/agent_inference_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=['interactive', 'single', 'batch'],
                        default=None, help="推理模式（覆盖配置文件）")
    parser.add_argument("--question", type=str, default=None, help="单问题模式的问题")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        print(f"[Error] 配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    mode = args.mode or config.get('inference', {}).get('mode', 'interactive')

    agent = build_agent_from_config(config)

    if mode == 'interactive':
        agent.interactive_chat()

    elif mode == 'single':
        question = args.question or config.get('inference', {}).get('single', {}).get('question', '')
        if not question:
            print("[Error] 单问题模式需要提供 --question 或在配置中设置 inference.single.question")
            sys.exit(1)
        result = agent.run(question, verbose=True)
        print(f"\n{'=' * 60}")
        print(f"最终回答: {result['answer']}")
        print(f"搜索次数: {result['num_tool_calls']}  耗时: {result['elapsed_sec']}s")
        print(f"{'=' * 60}")

    elif mode == 'batch':
        batch_cfg = config.get('inference', {}).get('batch', {})
        input_file = batch_cfg.get('input_file', '')
        output_file = batch_cfg.get('output_file', 'output/agent_results.json')
        max_samples = batch_cfg.get('max_samples')

        if not os.path.isabs(input_file):
            input_file = os.path.join(project_root, input_file)
        if not os.path.isabs(output_file):
            output_file = os.path.join(project_root, output_file)

        if input_file.endswith('.jsonl'):
            data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'data' in data:
                data = data['data']

        questions = []
        for item in data:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict):
                q = item.get('question') or item.get('query') or item.get('instruction') or ''
                if not q and 'messages' in item:
                    for msg in item['messages']:
                        if msg.get('role') == 'user' and not msg.get('content', '').startswith('<tool_response>'):
                            q = msg['content']
                            break
                questions.append(q)

        if max_samples:
            questions = questions[:max_samples]

        print(f"\n批量推理: {len(questions)} 个问题\n")
        results = agent.batch_run(questions, verbose=False)

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {output_file}")
        avg_tc = sum(r['num_tool_calls'] for r in results) / len(results)
        avg_t = sum(r['elapsed_sec'] for r in results) / len(results)
        print(f"平均搜索次数: {avg_tc:.1f}  平均耗时: {avg_t:.1f}s")


if __name__ == "__main__":
    main()
