#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DPO训练脚本 - 使用TRL库
支持从SFT LoRA模型开始训练
"""

import os
import sys
import json
import yaml
import torch
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent
training_dir = script_dir.parent
src_dir = training_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(project_root))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import Dataset, load_dataset
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """脚本参数"""
    config_path: str = field(
        default="config/dpo_training_config.yaml",
        metadata={"help": "配置文件路径"}
    )


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_and_merge_lora_model(
    base_model_path: str,
    lora_checkpoint_path: str,
    merge_lora: bool = True
):
    """
    加载基础模型和LoRA检查点，可选择是否合并
    
    Args:
        base_model_path: 基础模型路径
        lora_checkpoint_path: LoRA检查点路径
        merge_lora: 是否合并LoRA权重
        
    Returns:
        model, tokenizer
    """
    logger.info("=" * 60)
    logger.info("加载SFT模型")
    logger.info("=" * 60)
    logger.info(f"基础模型: {base_model_path}")
    logger.info(f"LoRA检查点: {lora_checkpoint_path}")
    logger.info(f"合并LoRA: {merge_lora}")
    
    # 加载tokenizer
    logger.info("\n加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("✓ Tokenizer加载完成")
    
    # 加载基础模型
    logger.info("\n加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    logger.info("✓ 基础模型加载完成")
    
    # 加载LoRA适配器
    logger.info("\n加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
    logger.info("✓ LoRA适配器加载完成")
    
    if merge_lora:
        # 合并LoRA权重
        logger.info("\n🔀 合并LoRA权重到基础模型...")
        model = model.merge_and_unload()
        logger.info("✓ LoRA权重已合并")
    else:
        logger.info("\n⚡ 保持LoRA适配器（未合并）")
    
    logger.info("=" * 60)
    
    return model, tokenizer


def load_dpo_dataset(
    data_path: str,
    max_length: int = 512,
    max_prompt_length: int = 256,
    system_prompt: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    加载DPO数据集（直接加载，不使用MedicalDataset的tokenization）

    TRL DPOTrainer会自己处理tokenization，我们只需提供原始文本。
    自动识别 JSON 数组 / JSONL 格式，流式读取。

    Args:
        data_path: 数据文件路径
        max_length: 最大长度（传给DPOTrainer配置）
        max_prompt_length: 最大prompt长度（传给DPOTrainer配置）
        system_prompt: 系统提示
        max_samples: 最大样本数（None = 全量）

    Returns:
        Dataset对象（TRL DPO格式：prompt, chosen, rejected）
    """
    logger.info(f"\n加载DPO数据: {data_path}")

    # 自动识别格式，流式读取，避免大文件一次性加载
    with open(data_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)

    if first_char == "[":
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            raw_data = [raw_data]
        if max_samples:
            raw_data = raw_data[:max_samples]
    else:
        raw_data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if max_samples and len(raw_data) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {lineno} 行解析失败，已跳过: {e}")

    logger.info(f"原始数据样本数: {len(raw_data)}")
    
    # 格式化为TRL格式
    formatted_data = []
    system_msg = system_prompt or "你是一个专业的医疗助手。"
    
    for item in raw_data:
        # 兼容多种字段命名风格
        prompt = (item.get('prompt') or item.get('question') or item.get('instruction') or '').strip()
        chosen = (item.get('chosen') or item.get('response_chosen') or item.get('accepted') or '').strip()
        rejected = (item.get('rejected') or item.get('response_rejected') or item.get('declined') or '').strip()

        if not prompt or not chosen or not rejected:
            logger.warning(f"跳过不完整的样本: {list(item.keys())}")
            continue
        
        # 构建完整的prompt（TRL格式）
        # 包含系统提示 + 用户问题 + assistant开始标记
        full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        formatted_data.append({
            'prompt': full_prompt,
            'chosen': chosen,  # 只包含回答内容
            'rejected': rejected  # 只包含回答内容
        })
    
    logger.info(f"✓ 有效DPO样本数: {len(formatted_data)}")
    
    # 创建Dataset（TRL会自动处理tokenization）
    dataset = Dataset.from_list(formatted_data)
    
    return dataset


def _resolve_path(path: Optional[str], root: Path) -> Optional[str]:
    if not path:
        return None
    return path if os.path.isabs(path) else str(root / path)


def load_model_and_tokenizer(model_config: Dict, project_root: Path):
    """加载模型和分词器，支持 CPT + SFT 双阶段动态挂载。

    加载顺序（按需）：
      1. Base Model
      2. mount + merge CPT LoRA  （cpt_adapter_path 不为 null 时）
      3. mount + merge SFT LoRA  （sft_checkpoint_path 不为 null 时）
      4. DPO LoRA 将由 DPOTrainer 负责套上

    精度优先级：bf16 > fp16 > fp32（从 model_config.dtype 或 CUDA 可用性推断）
    """
    use_cuda = torch.cuda.is_available()

    # 精度解析
    dtype_str = model_config.get("dtype", "bf16" if use_cuda else "fp32")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    logger.info(f"模型精度: {dtype_str}")

    base_model_path = _resolve_path(model_config["base_model_path"], project_root)
    cpt_adapter_path = _resolve_path(model_config.get("cpt_adapter_path"), project_root)
    sft_adapter_path = _resolve_path(model_config.get("sft_checkpoint_path"), project_root)

    logger.info("=" * 60)
    logger.info(f"Base model  : {base_model_path}")
    logger.info(f"CPT adapter : {cpt_adapter_path or '(跳过)'}")
    logger.info(f"SFT adapter : {sft_adapter_path or '(跳过)'}")
    logger.info("=" * 60)

    # ── 1. 加载 tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 2. 加载 base model ───────────────────────────────────────────────
    logger.info("加载 Base Model ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype if use_cuda else torch.float32,
        device_map={"": torch.cuda.current_device()} if use_cuda else None,
    )
    model.config.use_cache = False

    # ── 3. 动态挂载 CPT LoRA（如有）────────────────────────────────────
    if cpt_adapter_path:
        logger.info(f"挂载 CPT adapter: {cpt_adapter_path}")
        model = PeftModel.from_pretrained(model, cpt_adapter_path, is_trainable=False)
        model = model.merge_and_unload()
        logger.info("CPT adapter 已 merge")

    # ── 4. 动态挂载 SFT LoRA（如有）────────────────────────────────────
    if sft_adapter_path:
        logger.info(f"挂载 SFT adapter: {sft_adapter_path}")
        model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=False)
        model = model.merge_and_unload()
        logger.info("SFT adapter 已 merge，DPO LoRA 将在此基础上训练")

    model.config.use_cache = False
    return model, tokenizer


def setup_lora_config(lora_config: Dict) -> Optional[LoraConfig]:
    """配置LoRA"""
    if not lora_config.get('enabled', False):
        return None
    
    logger.info("\n⚙️ 配置LoRA...")
    
    config = LoraConfig(
        r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('lora_alpha', 32),
        target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        bias=lora_config.get('bias', "none"),
        task_type="CAUSAL_LM"
    )
    
    logger.info(f"  LoRA rank: {config.r}")
    logger.info(f"  LoRA alpha: {config.lora_alpha}")
    logger.info(f"  Target modules: {config.target_modules}")
    logger.info("✓ LoRA配置完成")
    
    return config


def train_dpo(config_path: str):
    """执行DPO训练"""
    
    # 加载配置
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 提取配置
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    lora_config = config.get('lora', {})
    output_config = config['output']
    
    # 路径处理
    train_data_path = data_config['train_data_path']
    if not os.path.isabs(train_data_path):
        train_data_path = os.path.join(project_root, train_data_path)
    
    eval_data_path = data_config.get('eval_data_path')
    if eval_data_path and not os.path.isabs(eval_data_path):
        eval_data_path = os.path.join(project_root, eval_data_path)
    
    output_dir = output_config['output_dir']
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    
    print("\n" + "=" * 70)
    print("DPO训练配置")
    print("=" * 70)
    print(f"基础模型: {model_config['base_model_path']}")
    if model_config.get('sft_checkpoint_path'):
        print(f"SFT检查点: {model_config['sft_checkpoint_path']}")
    print(f"训练数据: {train_data_path}")
    if eval_data_path:
        print(f"评估数据: {eval_data_path}")
    print(f"输出目录: {output_dir}")
    print(f"使用LoRA: {lora_config.get('enabled', False)}")
    print(f"训练轮数: {training_config['num_train_epochs']}")
    print(f"批次大小: {training_config['per_device_train_batch_size']}")
    print(f"学习率: {training_config['learning_rate']}")
    print(f"Beta (DPO): {training_config.get('beta', 0.1)}")
    print("=" * 70)
    
    # 1. 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_config, project_root)
    
    # 2. 配置LoRA（如果需要）
    peft_config = setup_lora_config(lora_config)
    model_is_peft = False  # 跟踪模型是否已经是PeftModel
    
    if peft_config:
        if not isinstance(model, PeftModel):
            # 模型还不是PeftModel，DPOTrainer会应用peft_config
            logger.info("✓ 将在DPO训练中应用新的LoRA配置")
            model_is_peft = False
        else:
            # 模型已经是PeftModel（不应该发生，因为我们合并了）
            logger.warning("⚠️ 模型已经是PeftModel，这不应该发生")
            model_is_peft = True
    
    # 3. 加载数据集
    logger.info("\n📂 加载数据集...")
    system_prompt = config.get('system_prompt')
    
    train_dataset = load_dpo_dataset(
        train_data_path,
        max_length=data_config.get('max_length', 512),
        max_prompt_length=data_config.get('max_prompt_length', 256),
        system_prompt=system_prompt,
        max_samples=data_config.get('max_train_samples'),
    )

    eval_dataset = None
    if eval_data_path:
        eval_dataset = load_dpo_dataset(
            eval_data_path,
            max_length=data_config.get('max_length', 512),
            max_prompt_length=data_config.get('max_prompt_length', 256),
            system_prompt=system_prompt,
            max_samples=data_config.get('max_eval_samples'),
        )
        logger.info(f"验证集样本数: {len(eval_dataset)}")
    
    # 4. 配置训练参数
    logger.info("\n⚙️ 配置训练参数...")
    
    training_args = DPOConfig(
        output_dir=output_dir,
        
        # 基础训练参数
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        
        # 优化器参数
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        
        # DPO特定参数
        beta=training_config.get('beta', 0.1),
        loss_type=training_config.get('loss_type', 'sigmoid'),  # 'sigmoid' or 'hinge' or 'ipo'
        
        # 长度限制参数
        max_length=data_config.get('max_length', 512),
        max_prompt_length=data_config.get('max_prompt_length', 256),
        
        # 评估参数：有验证集时自动开启，无验证集时强制关闭
        eval_strategy=training_config.get('eval_strategy', 'steps') if eval_dataset else 'no',
        eval_steps=training_config.get('eval_steps', 100) if eval_dataset else None,
        load_best_model_at_end=training_config.get('load_best_model_at_end', True) if eval_dataset else False,
        metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss') if eval_dataset else None,

        # 保存参数
        save_strategy=training_config.get('save_strategy', 'steps'),
        save_steps=training_config.get('save_steps', 200),
        save_total_limit=training_config.get('save_total_limit', 3),
        
        # 日志参数
        logging_steps=training_config.get('logging_steps', 10),
        logging_dir=os.path.join(output_dir, 'logs'),
        report_to=training_config.get('report_to', ['tensorboard']),
        
        # 精度：从 model_config.dtype 读取，与模型加载保持一致
        bf16=model_config.get("dtype", "bf16") == "bf16" and torch.cuda.is_available(),
        fp16=model_config.get("dtype", "bf16") == "fp16" and torch.cuda.is_available(),
        remove_unused_columns=False,
        gradient_checkpointing=training_config.get('gradient_checkpointing', False),
        
        # 数据加载
        dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
        dataloader_pin_memory=True,
    )
    
    # 5. 创建DPO Trainer
    logger.info("\n🚀 创建DPO Trainer...")
    
    # 只有当模型不是PeftModel时，才传递peft_config
    # DPOTrainer会自动应用peft_config到基础模型
    trainer_peft_config = peft_config if not model_is_peft else None
    
    if trainer_peft_config:
        logger.info("  使用LoRA训练（DPOTrainer将应用peft_config）")
    else:
        logger.info("  使用全参数训练")
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # None表示使用frozen copy作为reference model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # 新版TRL使用processing_class而不是tokenizer
        peft_config=trainer_peft_config,  # 只有当模型不是PeftModel时才传递
    )
    
    logger.info("✓ DPO Trainer创建完成")
    
    # 6. 开始训练
    logger.info("\n" + "=" * 70)
    logger.info("开始DPO训练...")
    logger.info("=" * 70 + "\n")
    
    train_result = trainer.train()
    
    # 7. 保存模型
    logger.info("\n💾 保存模型...")
    
    # 检查是否需要保存完整模型（包含SFT+DPO）
    save_merged = model_config.get('save_merged_dpo', True)
    
    if save_merged and isinstance(model, PeftModel):
        # 情况1: 模型是PeftModel（有DPO LoRA）
        logger.info("合并并保存完整模型（基础+SFT+DPO）...")
        
        # 合并DPO LoRA到已包含SFT的模型
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✓ 完整合并模型已保存到: {output_dir}")
        
        # 另外保存LoRA适配器到子目录（可选，用于继续训练）
        lora_dir = os.path.join(output_dir, "dpo_lora_adapter")
        os.makedirs(lora_dir, exist_ok=True)
        model.save_pretrained(lora_dir)
        logger.info(f"✓ DPO LoRA适配器已保存到: {lora_dir}")
        
    else:
        # 情况2: 使用全参数训练或已经是完整模型
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"✓ 模型已保存到: {output_dir}")
    
    # 保存训练状态
    trainer.save_state()
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logger.info(f"✓ 模型已保存到: {output_dir}")
    
    # 8. 打印训练统计
    logger.info("\n" + "=" * 70)
    logger.info("训练完成统计")
    logger.info("=" * 70)
    logger.info(f"训练样本数: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"评估样本数: {len(eval_dataset)}")
    logger.info(f"训练轮数: {training_config['num_train_epochs']}")
    logger.info(f"总步数: {train_result.global_step}")
    logger.info(f"训练损失: {metrics.get('train_loss', 'N/A')}")
    logger.info("=" * 70)
    
    logger.info("\n✅ DPO训练完成！")


def main():
    """主函数"""
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # 配置文件路径
    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    # 检查配置文件
    if not os.path.exists(config_path):
        logger.error(f"❌ 错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 运行训练
    try:
        train_dpo(config_path)
    except Exception as e:
        logger.error(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
#python src/training/scripts/run_dpo_training.py --config_path config/dpo_training_config.yaml