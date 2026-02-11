#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并SFT LoRA模型
将基础模型 + SFT LoRA 合并为完整的SFT模型
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def merge_sft_model(
    base_model_path: str = "Qwen2.5-1.5B-Instruct/qwen/Qwen2___5-1___5B-Instruct",
    sft_lora_path: str = "model_output/qwen2_5_1_5b_instruct_sft",
    output_path: str = "model_output/qwen2_5_1_5b_instruct_sft_merged"
):
    """
    合并SFT LoRA到基础模型
    
    Args:
        base_model_path: 基础模型路径
        sft_lora_path: SFT LoRA适配器路径
        output_path: 输出完整模型路径
    """
    
    print("=" * 70)
    print("SFT模型合并工具")
    print("=" * 70)
    
    # 转换为绝对路径
    if not os.path.isabs(base_model_path):
        base_model_path = os.path.join(project_root, base_model_path)
    if not os.path.isabs(sft_lora_path):
        sft_lora_path = os.path.join(project_root, sft_lora_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    
    print(f"\n📂 路径配置:")
    print(f"  基础模型: {base_model_path}")
    print(f"  SFT LoRA: {sft_lora_path}")
    print(f"  输出路径: {output_path}")
    
    # 检查路径是否存在
    if not os.path.exists(base_model_path):
        print(f"\n❌ 错误: 基础模型路径不存在: {base_model_path}")
        return None
    
    if not os.path.exists(sft_lora_path):
        print(f"\n❌ 错误: SFT LoRA路径不存在: {sft_lora_path}")
        return None
    
    # 步骤1: 加载基础模型
    print(f"\n步骤1: 加载基础模型...")
    print(f"  (这可能需要几分钟...)")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✓ 基础模型加载完成")
    
    # 步骤2: 加载SFT LoRA
    print(f"\n步骤2: 加载SFT LoRA适配器...")
    model_with_lora = PeftModel.from_pretrained(base_model, sft_lora_path)
    print("✓ SFT LoRA加载完成")
    
    # 显示可训练参数
    trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_with_lora.parameters())
    print(f"\n📊 模型参数:")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  总参数: {total_params:,}")
    print(f"  LoRA参数占比: {100 * trainable_params / total_params:.2f}%")
    
    # 步骤3: 合并LoRA
    print(f"\n步骤3: 合并SFT LoRA到基础模型...")
    print(f"  (这可能需要几分钟...)")
    merged_model = model_with_lora.merge_and_unload()
    print("✓ LoRA已合并到基础模型")
    
    # 步骤4: 保存完整模型
    print(f"\n步骤4: 保存完整模型...")
    os.makedirs(output_path, exist_ok=True)
    
    print(f"  保存模型权重...")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True  # 使用safetensors格式
    )
    
    print(f"  保存tokenizer...")
    tokenizer.save_pretrained(output_path)
    
    print("✓ 模型已保存")
    
    # 步骤5: 验证保存的文件
    print(f"\n步骤5: 验证保存的文件...")
    saved_files = []
    total_size = 0
    
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            size_mb = size / (1024 * 1024)
            saved_files.append((file, size_mb))
    
    print(f"\n📊 保存的文件:")
    for file, size_mb in sorted(saved_files, key=lambda x: -x[1]):
        if size_mb > 1:  # 只显示大于1MB的文件
            print(f"  {file}: {size_mb:.1f} MB")
    
    total_size_gb = total_size / (1024 * 1024 * 1024)
    print(f"\n  总大小: {total_size_gb:.2f} GB")
    
    # 检查是否是完整模型（不是LoRA）
    has_adapter_config = os.path.exists(os.path.join(output_path, "adapter_config.json"))
    has_model_weights = any(
        f.endswith(('.safetensors', '.bin')) and 'adapter' not in f
        for f in os.listdir(output_path)
    )
    
    if has_adapter_config:
        print("\n⚠️  警告: 发现adapter_config.json，这可能仍然是LoRA适配器")
    
    if not has_model_weights:
        print("\n⚠️  警告: 未找到完整模型权重文件")
    
    if has_model_weights and not has_adapter_config:
        print("\n✅ 验证通过: 这是一个完整的合并模型")
    
    # 完成
    print("\n" + "=" * 70)
    print("✅ SFT模型合并完成！")
    print("=" * 70)
    print(f"\n完整SFT模型保存在: {output_path}")
    print("\n现在你可以:")
    print("1. 评估SFT模型:")
    print(f"   model_path: {output_path}")
    print(f"   is_lora: false")
    print("\n2. 在此基础上训练DPO:")
    print(f"   base_model_path: {output_path}")
    print(f"   is_lora: false")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="合并SFT LoRA模型")
    parser.add_argument(
        "--base_model",
        default="Qwen2.5-1.5B-Instruct/qwen/Qwen2___5-1___5B-Instruct",
        help="基础模型路径"
    )
    parser.add_argument(
        "--sft_lora",
        default="model_output/qwen2_5_1_5b_instruct_sft",
        help="SFT LoRA路径"
    )
    parser.add_argument(
        "--output",
        default="model_output/qwen2_5_1_5b_instruct_sft_merged",
        help="输出路径"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = merge_sft_model(
            base_model_path=args.base_model,
            sft_lora_path=args.sft_lora,
            output_path=args.output
        )
        
        if output_path:
            print(f"\n✅ 成功！")
            sys.exit(0)
        else:
            print(f"\n❌ 失败")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
# python scripts/merge_sft_model.py 