"""
DPO Trainer (Direct Preference Optimization Trainer)

基于 TRL DPOTrainer，对医疗问答偏好数据进行对齐训练。
支持 LoRA / 全参数训练，可从 CPT + SFT 阶段的 adapter 链式加载。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOConfig, DPOTrainer as _TRLDPOTrainer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DPOTrainer:
    """医疗问答 DPO 训练器。

    训练模式（由配置自动选择）：
      - LoRA  : use_lora=True  → 仅训练 LoRA 参数，节省显存
      - 全参数: use_lora=False → 训练全部参数
    """

    DEFAULT_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    def __init__(
        self,
        base_model_path: str,
        output_dir: str,
        # 预训练 adapter 路径（按合并顺序）
        cpt_adapter_path: Optional[str] = None,
        sft_adapter_path: Optional[str] = None,
        # LoRA
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # DPO
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        # 数据长度
        max_length: int = 512,
        max_prompt_length: int = 256,
        # 训练超参（整体传入，优先级最高）
        training_args: Optional[Dict[str, Any]] = None,
    ):
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        self.cpt_adapter_path = cpt_adapter_path
        self.sft_adapter_path = sft_adapter_path

        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or self.DEFAULT_TARGET_MODULES

        self.beta = beta
        self.loss_type = loss_type
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.training_args = training_args or {}

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model = None
        self.trainer: Optional[_TRLDPOTrainer] = None

    # ------------------------------------------------------------------
    # 模型 & Tokenizer 加载
    # ------------------------------------------------------------------

    def _get_model_dtype(self) -> tuple[torch.dtype, str]:
        if self.training_args.get("bf16", False):
            return torch.bfloat16, "bf16"
        if self.training_args.get("fp16", False):
            return torch.float16, "fp16"
        return torch.float32, "fp32"

    def load_model_and_tokenizer(self):
        """加载 base model，按需合并 CPT / SFT adapter。"""
        use_cuda = torch.cuda.is_available()
        model_dtype, dtype_name = self._get_model_dtype()

        logger.info("=" * 60)
        logger.info(f"Base model   : {self.base_model_path}")
        logger.info(f"CPT adapter  : {self.cpt_adapter_path or '(跳过)'}")
        logger.info(f"SFT adapter  : {self.sft_adapter_path or '(跳过)'}")
        logger.info(f"精度: {dtype_name}  LoRA: {self.use_lora}")
        logger.info("=" * 60)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            torch_dtype=model_dtype if use_cuda else torch.float32,
            device_map={"": torch.cuda.current_device()} if use_cuda else None,
        )
        model.config.use_cache = False

        for tag, adapter_path in [("CPT", self.cpt_adapter_path),
                                  ("SFT", self.sft_adapter_path)]:
            if adapter_path:
                logger.info(f"挂载 {tag} adapter: {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
                model = model.merge_and_unload()
                logger.info(f"{tag} adapter 已 merge")

        model.config.use_cache = False
        self.model = model
        return self.model, self.tokenizer

    # ------------------------------------------------------------------
    # 训练主流程
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()

        use_cuda = torch.cuda.is_available()
        has_eval = eval_dataset is not None

        defaults: Dict[str, Any] = {
            "output_dir": self.output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "bf16": False,
            "fp16": False,
            "gradient_checkpointing": use_cuda,
            "logging_steps": 10,
            "save_steps": 200,
            "save_total_limit": 3,
            "eval_strategy": "steps" if has_eval else "no",
            "eval_steps": 100 if has_eval else None,
            "load_best_model_at_end": has_eval,
            "report_to": "tensorboard",
            "remove_unused_columns": False,
            "seed": 42,
            "beta": self.beta,
            "loss_type": self.loss_type,
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
        }
        defaults.update(self.training_args)
        if not has_eval:
            defaults.pop("eval_steps", None)
            defaults["eval_strategy"] = "no"
            defaults["load_best_model_at_end"] = False

        dpo_config = DPOConfig(**defaults)

        peft_config = None
        if self.use_lora:
            peft_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.lora_target_modules,
            )

        self.trainer = _TRLDPOTrainer(
            model=self.model,
            ref_model=None,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )

        logger.info("=" * 60)
        logger.info("开始 DPO 训练 ...")
        train_result = self.trainer.train()

        logger.info(f"保存模型到 {self.output_dir}")
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        logger.info("=" * 60)
        logger.info("DPO 训练完成！")
        logger.info(f"  训练损失: {metrics.get('train_loss', 'N/A')}")
        logger.info(f"  总步数: {train_result.global_step}")
        logger.info("=" * 60)

        return self.output_dir


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_dpo_trainer_from_config(config: Dict[str, Any]) -> DPOTrainer:
    """从 dpo_training_config.yaml 解析的字典构建 DPOTrainer。"""
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    return DPOTrainer(
        base_model_path=model_cfg.get("base_model_path", ""),
        output_dir=output_cfg.get("output_dir", "model_output/dpo"),
        cpt_adapter_path=model_cfg.get("cpt_adapter_path"),
        sft_adapter_path=model_cfg.get("sft_checkpoint_path"),
        use_lora=lora_cfg.get("enabled", True),
        lora_r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        lora_target_modules=lora_cfg.get("target_modules", None),
        beta=train_cfg.get("beta", 0.1),
        loss_type=train_cfg.get("loss_type", "sigmoid"),
        max_length=data_cfg.get("max_length", 512),
        max_prompt_length=data_cfg.get("max_prompt_length", 256),
        training_args=train_cfg,
    )
