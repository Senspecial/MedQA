#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
医疗QA模型推理工具
支持SFT和DPO模型的推理，支持LoRA模型加载和合并
支持 HuggingFace Transformers 和 vLLM 两种推理后端
"""

import os
import sys
import json
import yaml
import torch
from pathlib import Path
from typing import List, Dict, Optional, Union

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

# 尝试导入 vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def load_system_prompt(config_path: str = "config/system_prompt.yaml") -> str:
    """
    加载系统提示
    
    Args:
        config_path: 系统提示配置文件路径
        
    Returns:
        系统提示字符串
    """
    try:
        # 尝试多个可能的路径
        possible_paths = [
            config_path,
            os.path.join(project_root, config_path),
            os.path.join(os.getcwd(), config_path),
        ]
        
        for full_path in possible_paths:
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config and 'system_prompt' in config:
                        return config['system_prompt']
    except Exception as e:
        print(f"警告: 无法加载系统提示配置 {config_path}: {e}")
    
    # 默认系统提示
    return """你是一个专业的医疗健康信息助手，具备全科医学基础知识。
请遵循以下原则：
1. 使用不确定性表述（"可能是"、"考虑"、"常见原因包括"）
2. 建议检查项目和就医科室，但不做明确诊断
3. 严重症状必须建议就医
4. 不编造信息，不确定时引导专业就医"""


class MedicalQAInference:
    """医疗QA推理类，支持 HuggingFace Transformers 和 vLLM 两种后端"""
    
    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        is_lora: bool = False,
        merge_lora: bool = True,
        system_prompt: Optional[str] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_vllm: bool = False,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_tensor_parallel_size: int = 1,
        vllm_max_model_len: int = 4096,
        vllm_dtype: str = "auto",
    ):
        """
        初始化推理器

        Args:
            model_path: 模型路径（可以是完整模型或LoRA适配器）
            base_model_path: 基础模型路径（仅当is_lora=True时需要）
            is_lora: 是否是LoRA模型
            merge_lora: 是否合并LoRA权重（推荐True以加速推理）
            system_prompt: 系统提示（如果为None，从配置文件加载）
            device: 设备（cuda/cpu），仅对HuggingFace后端有效
            load_in_8bit: 是否以8bit加载（节省显存），仅对HuggingFace后端有效
            load_in_4bit: 是否以4bit加载（更省显存），仅对HuggingFace后端有效
            use_vllm: 是否使用vLLM作为推理后端（高吞吐量场景推荐）
            vllm_gpu_memory_utilization: vLLM GPU显存占用比例（0.0-1.0）
            vllm_tensor_parallel_size: vLLM张量并行GPU数量
            vllm_max_model_len: vLLM最大序列长度
            vllm_dtype: vLLM数据类型（auto/float16/bfloat16）
        """
        self.use_vllm = use_vllm
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.is_lora = is_lora

        if use_vllm and not VLLM_AVAILABLE:
            raise ImportError(
                "use_vllm=True 但 vLLM 未安装，请执行: pip install vllm"
            )

        # 加载系统提示
        if system_prompt is None:
            self.system_prompt = load_system_prompt()
            print("✓ 从配置文件加载系统提示")
        else:
            self.system_prompt = system_prompt
            print("✓ 使用传入的系统提示")

        print(f"\n{'='*60}")
        print("初始化医疗QA推理器")
        print(f"{'='*60}")
        print(f"推理后端: {'vLLM' if use_vllm else 'HuggingFace Transformers'}")
        print(f"模型路径: {model_path}")
        print(f"是否LoRA: {is_lora}")
        if is_lora:
            print(f"基础模型: {base_model_path}")
            print(f"合并LoRA: {merge_lora}")
        if not use_vllm:
            print(f"设备: {self.device}")
        print(f"{'='*60}\n")

        if use_vllm:
            self._init_vllm(
                model_path=model_path,
                dtype=vllm_dtype,
                tensor_parallel_size=vllm_tensor_parallel_size,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                max_model_len=vllm_max_model_len,
            )
        else:
            self._init_hf(
                model_path=model_path,
                base_model_path=base_model_path,
                is_lora=is_lora,
                merge_lora=merge_lora,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )

    def _init_vllm(
        self,
        model_path: str,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
    ):
        """使用 vLLM 初始化推理引擎"""
        print("📥 使用 vLLM 加载模型...")
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count < tensor_parallel_size:
            print(f"  警告: 可用GPU数({gpu_count}) < 请求并行数({tensor_parallel_size})，已自动调整")
            tensor_parallel_size = max(1, gpu_count)

        self.vllm_engine = LLM(
            model=model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.tokenizer = self.vllm_engine.get_tokenizer()
        print("✓ vLLM 模型加载完成\n")

    def _init_hf(
        self,
        model_path: str,
        base_model_path: Optional[str],
        is_lora: bool,
        merge_lora: bool,
        load_in_8bit: bool,
        load_in_4bit: bool,
    ):
        """使用 HuggingFace Transformers 初始化推理引擎"""
        print("📥 加载Tokenizer...")
        tokenizer_path = base_model_path if is_lora else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✓ Tokenizer加载完成")

        print("\n📥 加载模型...")
        load_kwargs = {
            'trust_remote_code': True,
            'device_map': 'auto' if self.device == 'cuda' else None,
        }
        if not (load_in_8bit or load_in_4bit):
            load_kwargs['torch_dtype'] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if load_in_8bit:
            load_kwargs['load_in_8bit'] = True
            print("  使用8bit量化加载")
        elif load_in_4bit:
            load_kwargs['load_in_4bit'] = True
            print("  使用4bit量化加载")

        if is_lora:
            print(f"  加载基础模型: {base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
            print(f"  加载LoRA适配器: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)
            if merge_lora:
                print("  合并LoRA权重...")
                model = model.merge_and_unload()
                print("  ✓ LoRA权重已合并")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        model.eval()
        self.model = model
        print("✓ 模型加载完成\n")
    
    def _build_prompt(self, question: str) -> str:
        """将问题构建为带 chat template 的完整 prompt 字符串"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # 手动构建（Qwen 格式兜底）
        text = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        text += f"<|im_start|>user\n{question}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        return text

    def generate(
        self,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """
        生成单个问题的回答

        Args:
            question: 问题
            max_new_tokens: 最大生成token数
            temperature: 温度（越高越随机）
            top_p: nucleus sampling
            top_k: top-k sampling
            repetition_penalty: 重复惩罚
            do_sample: 是否采样

        Returns:
            生成的回答
        """
        if self.use_vllm:
            return self._generate_vllm(
                question, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, do_sample
            )
        return self._generate_hf(
            question, max_new_tokens, temperature, top_p, top_k,
            repetition_penalty, do_sample
        )

    def _generate_vllm(
        self,
        question: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
    ) -> str:
        """使用 vLLM 生成回答"""
        prompt = self._build_prompt(question)
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0 if not do_sample else temperature,
            top_p=1.0 if not do_sample else top_p,
            top_k=-1 if not do_sample else top_k,
            repetition_penalty=repetition_penalty,
            stop=["<|im_end|>"],
        )
        outputs = self.vllm_engine.generate(prompt, sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _generate_hf(
        self,
        question: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
    ) -> str:
        """使用 HuggingFace Transformers 生成回答"""
        text = self._build_prompt(question)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
    
    def batch_generate(
        self,
        questions: List[str],
        batch_size: int = 4,
        **generate_kwargs
    ) -> List[str]:
        """
        批量生成回答。vLLM 后端会一次性提交所有 prompt，充分发挥并行吞吐优势；
        HuggingFace 后端按 batch_size 分批顺序推理。

        Args:
            questions: 问题列表
            batch_size: 批次大小（仅对 HuggingFace 后端有效）
            **generate_kwargs: 传递给 generate 的其他参数

        Returns:
            回答列表
        """
        if self.use_vllm:
            return self._batch_generate_vllm(questions, **generate_kwargs)
        return self._batch_generate_hf(questions, batch_size, **generate_kwargs)

    def _batch_generate_vllm(self, questions: List[str], **generate_kwargs) -> List[str]:
        """vLLM 批量推理：一次提交全部 prompt"""
        max_new_tokens = generate_kwargs.get("max_new_tokens", 512)
        temperature = generate_kwargs.get("temperature", 0.7)
        top_p = generate_kwargs.get("top_p", 0.9)
        top_k = generate_kwargs.get("top_k", 50)
        repetition_penalty = generate_kwargs.get("repetition_penalty", 1.0)
        do_sample = generate_kwargs.get("do_sample", True)

        prompts = [self._build_prompt(q) for q in questions]
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0 if not do_sample else temperature,
            top_p=1.0 if not do_sample else top_p,
            top_k=-1 if not do_sample else top_k,
            repetition_penalty=repetition_penalty,
            stop=["<|im_end|>"],
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def _batch_generate_hf(
        self,
        questions: List[str],
        batch_size: int,
        **generate_kwargs
    ) -> List[str]:
        """HuggingFace 分批推理"""
        from tqdm import tqdm

        answers = []
        for i in tqdm(range(0, len(questions), batch_size), desc="批量推理"):
            batch = questions[i:i + batch_size]
            for question in batch:
                answer = self.generate(question, **generate_kwargs)
                answers.append(answer)
        return answers
    
    def interactive_chat(self):
        """交互式对话模式"""
        print("\n" + "="*60)
        print("医疗QA交互式对话")
        print("="*60)
        print("输入问题开始对话，输入 'quit' 或 'exit' 退出")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("\n👤 用户: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("\n再见！")
                    break
                
                if not question:
                    continue
                
                print("\n🤖 助手: ", end="", flush=True)
                answer = self.generate(question)
                print(answer)
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="医疗QA模型推理")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="基础模型路径（仅当使用LoRA时需要）"
    )
    parser.add_argument(
        "--is_lora",
        action="store_true",
        help="是否是LoRA模型"
    )
    parser.add_argument(
        "--no_merge",
        action="store_true",
        help="不合并LoRA（默认合并）"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="单个问题（如果不提供，进入交互模式）"
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=None,
        help="问题文件（JSON格式，每行一个问题）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（批量推理时）"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="top_p"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="以8bit加载模型（节省显存）"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="以4bit加载模型（更省显存）"
    )
    
    args = parser.parse_args()
    
    # 初始化推理器
    inferencer = MedicalQAInference(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        is_lora=args.is_lora,
        merge_lora=not args.no_merge,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    
    # 单个问题
    if args.question:
        print(f"\n问题: {args.question}\n")
        answer = inferencer.generate(
            args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"回答: {answer}\n")
    
    # 批量问题
    elif args.questions_file:
        print(f"\n从文件加载问题: {args.questions_file}")
        
        with open(args.questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持多种格式
        if isinstance(data, list):
            if isinstance(data[0], str):
                questions = data
            elif isinstance(data[0], dict):
                questions = [item.get('question') or item.get('query') or '' for item in data]
        else:
            raise ValueError("不支持的问题文件格式")
        
        print(f"共 {len(questions)} 个问题\n")
        
        # 批量生成
        answers = inferencer.batch_generate(
            questions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        # 保存结果
        results = [
            {"question": q, "answer": a}
            for q, a in zip(questions, answers)
        ]
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 结果已保存到: {args.output}")
        else:
            # 打印前3个
            print("\n" + "="*60)
            print("示例结果（前3个）:")
            print("="*60)
            for i, result in enumerate(results[:3], 1):
                print(f"\n[{i}] 问题: {result['question']}")
                print(f"    回答: {result['answer'][:200]}...")
    
    # 交互模式
    else:
        inferencer.interactive_chat()


if __name__ == "__main__":
    main()
