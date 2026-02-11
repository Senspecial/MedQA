#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并LoRA适配器到基础模型
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent
training_dir = script_dir.parent
src_dir = training_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_model(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    device: str = "cuda",
    max_shard_size: str = "5GB"
):
    """
    合并LoRA适配器到基础模型
    
    Args:
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA适配器路径
        output_path: 输出路径
        device: 设备
        max_shard_size: 最大分片大小
    """
    
    print("=" * 70)
    print("LoRA 模型合并工具")
    print("=" * 70)
    print(f"基础模型: {base_model_path}")
    print(f"LoRA适配器: {lora_adapter_path}")
    print(f"输出路径: {output_path}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 1. 加载tokenizer
    print("\n📥 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    print("✓ tokenizer加载完成")
    
    # 2. 加载基础模型
    print("\n📥 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() and device == "cuda" else torch.float32,
        device_map="auto" if torch.cuda.is_available() and device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    print("✓ 基础模型加载完成")
    
    # 3. 加载LoRA适配器
    print("\n📥 加载LoRA适配器...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() and device == "cuda" else torch.float32
    )
    print("✓ LoRA适配器加载完成")
    
    # 4. 合并权重
    print("\n🔀 合并LoRA权重到基础模型...")
    merged_model = model.merge_and_unload()
    print("✓ 权重合并完成")
    
    # 5. 保存合并后的模型
    print(f"\n💾 保存合并后的模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    merged_model.save_pretrained(
        output_path,
        max_shard_size=max_shard_size,
        safe_serialization=True
    )
    print("✓ 模型已保存")
    
    # 6. 保存tokenizer
    print(f"\n💾 保存tokenizer...")
    tokenizer.save_pretrained(output_path)
    print("✓ tokenizer已保存")
    
    # 7. 保存配置信息
    print(f"\n💾 保存合并信息...")
    merge_info = {
        "base_model": base_model_path,
        "lora_adapter": lora_adapter_path,
        "merged_at": str(Path(output_path).absolute()),
        "device": device,
        "dtype": str(merged_model.dtype)
    }
    
    import json
    with open(os.path.join(output_path, "merge_info.json"), 'w', encoding='utf-8') as f:
        json.dump(merge_info, f, ensure_ascii=False, indent=2)
    print("✓ 合并信息已保存")
    
    print("\n" + "=" * 70)
    print("✅ LoRA模型合并完成！")
    print("=" * 70)
    print(f"\n合并后的模型保存在: {output_path}")
    print("\n使用方法:")
    print("```python")
    print("from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f'model = AutoModelForCausalLM.from_pretrained("{output_path}")')
    print(f'tokenizer = AutoTokenizer.from_pretrained("{output_path}")')
    print("```")
    
    # 清理内存
    del merged_model
    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="合并LoRA适配器到基础模型")
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="基础模型路径"
    )
    
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="LoRA适配器路径"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出路径"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="设备"
    )
    
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="最大分片大小（如：5GB, 2GB）"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    base_model_path = args.base_model
    if not os.path.isabs(base_model_path):
        base_model_path = os.path.join(project_root, base_model_path)
    
    lora_adapter_path = args.lora_adapter
    if not os.path.isabs(lora_adapter_path):
        lora_adapter_path = os.path.join(project_root, lora_adapter_path)
    
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    
    # 检查路径
    if not os.path.exists(base_model_path):
        print(f"❌ 错误: 基础模型路径不存在: {base_model_path}")
        sys.exit(1)
    
    if not os.path.exists(lora_adapter_path):
        print(f"❌ 错误: LoRA适配器路径不存在: {lora_adapter_path}")
        sys.exit(1)
    
    # 执行合并
    try:
        merge_lora_model(
            base_model_path=base_model_path,
            lora_adapter_path=lora_adapter_path,
            output_path=output_path,
            device=args.device,
            max_shard_size=args.max_shard_size
        )
    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
