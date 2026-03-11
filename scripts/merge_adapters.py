#!/usr/bin/env python3
"""
多 LoRA Adapter 顺序合并脚本

将多个 LoRA adapter 依次合并到基础模型，输出完整权重。
合并顺序：base → CPT adapter → SFT adapter → DPO adapter → 保存

用法：
    python scripts/merge_adapters.py
    python scripts/merge_adapters.py --output model_output/qwen3_5_0_8b_merged
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def abs_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return str(PROJECT_ROOT / p)


def merge_adapters(
    base_model_path: str,
    adapter_paths: list[str],       # 按合并顺序排列
    output_path: str,
    torch_dtype=torch.bfloat16,
):
    base_model_path = abs_path(base_model_path)
    adapter_paths   = [abs_path(p) for p in adapter_paths]
    output_path     = abs_path(output_path)

    print("=" * 65)
    print("多 Adapter 顺序合并")
    print("=" * 65)
    print(f"基础模型:  {base_model_path}")
    for i, p in enumerate(adapter_paths, 1):
        print(f"Adapter {i}: {p}")
    print(f"输出路径:  {output_path}")
    print("=" * 65)

    # 检查路径
    for p in [base_model_path] + adapter_paths:
        if not os.path.exists(p):
            print(f"[ERROR] 路径不存在: {p}")
            sys.exit(1)

    # 1. 加载 tokenizer（从 base model）
    print("\n[1/3] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 2. 加载 base model
    print("[2/3] 加载基础模型 ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    print(f"      参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # 3. 逐个加载并合并 adapter
    print("[3/3] 逐层合并 adapter ...")
    for i, adapter_path in enumerate(adapter_paths, 1):
        adapter_name = Path(adapter_path).name
        print(f"\n  [{i}/{len(adapter_paths)}] 加载 {adapter_name} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"         合并中 ...")
        model = model.merge_and_unload()
        print(f"         ✓ 合并完成")

    # 4. 保存
    print(f"\n保存合并模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    # 统计文件
    total_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if os.path.isfile(os.path.join(output_path, f))
    )
    print(f"总大小: {total_size / 1024**3:.2f} GB")
    print("\n✓ 合并完成！")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="多 LoRA adapter 顺序合并")
    parser.add_argument(
        "--base-model",
        default="Qwen3.5-0.8B-Base/Qwen/Qwen3___5-0___8B-Base",
        help="基础模型路径（相对于项目根目录或绝对路径）",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=[
            "model_output/qwen3_5_0_8b_cpt",
            "model_output/qwen3_5_0_8b_sft",
            "model_output/qwen3_5_0_8b_dpo",
        ],
        help="adapter 路径列表，按合并顺序排列",
    )
    parser.add_argument(
        "--output",
        default="model_output/qwen3_5_0_8b_merged",
        help="合并后模型的保存路径",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="模型权重精度",
    )
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }

    merge_adapters(
        base_model_path=args.base_model,
        adapter_paths=args.adapters,
        output_path=args.output,
        torch_dtype=dtype_map[args.dtype],
    )


if __name__ == "__main__":
    main()
