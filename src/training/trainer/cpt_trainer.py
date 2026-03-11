"""
增量预训练 Trainer (Continual Pre-Training Trainer)

基于 Hugging Face Trainer，使用 CLM (因果语言建模) 目标对基础模型进行
医学领域增量预训练。支持 QLoRA / LoRA / 全参数三种模式。

训练完成后自动在验证集 / 测试集上评估 PPL 和 next-token Accuracy。
训练产出 LoRA adapter，可通过 merge_lora_model.py 合并后供 SFT 使用。
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src.utils.logger import setup_logger
from ..dataset.pretrain_dataset import PretrainDataCollator, build_split_dataset

logger = setup_logger(__name__)


class CPTTrainer:
    """增量预训练训练器。

    支持三种训练模式（由配置自动选择）：
      - QLoRA : use_lora=True  + use_4bit=True   → 省显存，速度较慢
      - LoRA  : use_lora=True  + use_4bit=False  → 速度快，需要更多显存
      - 全参数: use_lora=False                   → 最快，显存需求最高

    模型精度（bf16 / fp16 / fp32）统一从 training_args 读取，不硬编码。
    """

    DEFAULT_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        # LoRA
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_target_modules: Optional[List[str]] = None,
        # 量化（仅 use_lora=True 时生效）
        use_4bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = False,
        # 训练超参（YAML training 节整体传入，优先级最高）
        training_args: Optional[Dict[str, Any]] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir

        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.lora_target_modules = lora_target_modules or self.DEFAULT_TARGET_MODULES

        self.use_4bit = use_4bit and use_lora  # 4bit 只在 LoRA 模式下有意义
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        self.training_args = training_args or {}

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model = None
        self.trainer: Optional[Trainer] = None

    # ------------------------------------------------------------------
    # 工具：从 training_args 推断模型精度
    # ------------------------------------------------------------------

    def _get_model_dtype(self) -> tuple[torch.dtype, str]:
        """从 training_args 读取 bf16/fp16 配置，返回 (torch.dtype, 名称)。"""
        if self.training_args.get("bf16", False):
            return torch.bfloat16, "bf16"
        if self.training_args.get("fp16", False):
            return torch.float16, "fp16"
        return torch.float32, "fp32"

    # ------------------------------------------------------------------
    # 模型 & Tokenizer 加载
    # ------------------------------------------------------------------

    def load_model_and_tokenizer(self):
        """加载 tokenizer 和模型。

        精度由 training_args 中的 bf16 / fp16 字段决定，无硬编码。
        模式选择：
          use_lora=True  + use_4bit=True  → QLoRA (4bit 量化 + LoRA)
          use_lora=True  + use_4bit=False → LoRA  (原生精度)
          use_lora=False                  → 全参数微调
        """
        logger.info(f"加载 tokenizer: {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        use_cuda = torch.cuda.is_available()
        model_dtype, dtype_name = self._get_model_dtype()

        # ── 加载 base model ──────────────────────────────────────────────
        if self.use_4bit and use_cuda:
            logger.info(f"模式: QLoRA  精度: 4bit(NF4) compute={dtype_name}")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=model_dtype,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                quantization_config=bnb_config,
                device_map={"": torch.cuda.current_device()},
                trust_remote_code=True,
            )
            base_model = prepare_model_for_kbit_training(base_model)

        else:
            mode_name = f"LoRA ({dtype_name})" if self.use_lora else f"全参数 ({dtype_name})"
            logger.info(f"模式: {mode_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=model_dtype if use_cuda else torch.float32,
                device_map={"": torch.cuda.current_device()} if use_cuda else None,
                trust_remote_code=True,
            )

        base_model.config.use_cache = False

        # ── 挂载 LoRA ────────────────────────────────────────────────────
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                task_type="CAUSAL_LM",
                target_modules=self.lora_target_modules,
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()
        else:
            self.model = base_model

        return self.model, self.tokenizer

    # ------------------------------------------------------------------
    # TrainingArguments 构建
    # ------------------------------------------------------------------

    def _build_training_arguments(self, has_eval: bool) -> TrainingArguments:
        """构建 TrainingArguments，用户 training_args 优先级最高。

        默认值随 use_lora / use_4bit 自动调整，避免硬编码冲突。
        """
        use_cuda = torch.cuda.is_available()
        _, dtype_name = self._get_model_dtype()

        # 优化器：4bit 模式用 paged_adamw_8bit 省显存；否则用标准 adamw
        default_optim = "paged_adamw_8bit" if (self.use_4bit and use_cuda) else "adamw_torch"

        defaults: Dict[str, Any] = {
            "output_dir": self.output_dir,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "optim": default_optim,
            "fp16": False,
            "bf16": False,
            "gradient_checkpointing": use_cuda,  # 默认开启，可在 yaml 中覆盖
            "logging_steps": 20,
            "save_steps": 200,
            "save_total_limit": 3,
            "eval_strategy": "steps" if has_eval else "no",
            "eval_steps": 200 if has_eval else None,
            "load_best_model_at_end": False,
            "report_to": "tensorboard",
            "remove_unused_columns": False,
            "seed": 42,
        }

        # 用户配置整体覆盖默认值
        defaults.update(self.training_args)

        # eval_steps 只在有验证集时有效
        if not has_eval:
            defaults.pop("eval_steps", None)
            defaults["eval_strategy"] = "no"

        logger.info(
            f"训练精度: {'bf16' if defaults['bf16'] else 'fp16' if defaults['fp16'] else 'fp32'}  "
            f"优化器: {defaults['optim']}  "
            f"梯度检查点: {defaults['gradient_checkpointing']}"
        )
        return TrainingArguments(**defaults)

    # ------------------------------------------------------------------
    # PPL & Accuracy 评估
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_ppl_and_accuracy(
        self,
        dataset,
        split_name: str = "eval",
        batch_size: int = 4,
        pack_sequences: bool = True,
    ) -> Dict[str, float]:
        """在给定数据集上计算 PPL 和 next-token Accuracy。

        Returns:
            {"loss": float, "ppl": float, "accuracy": float}
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        collator = (
            default_data_collator if pack_sequences
            else PretrainDataCollator(pad_token_id=self.tokenizer.pad_token_id)
        )
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        for batch in loader:
            input_ids    = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels       = batch["labels"].to(device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            num_batches += 1

            valid_mask = labels != -100
            pred_ids   = outputs.logits.argmax(dim=-1)
            total_correct += ((pred_ids == labels) & valid_mask).sum().item()
            total_tokens  += valid_mask.sum().item()

        avg_loss = total_loss / max(num_batches, 1)
        ppl      = math.exp(min(avg_loss, 20))  # 防止 exp 溢出
        accuracy = total_correct / max(total_tokens, 1)

        logger.info(
            f"[{split_name}] loss={avg_loss:.4f}  PPL={ppl:.2f}  "
            f"Accuracy={accuracy * 100:.2f}%  "
            f"(correct={total_correct} / {total_tokens} tokens)"
        )
        self.model.train()
        return {"loss": avg_loss, "ppl": ppl, "accuracy": accuracy}

    # ------------------------------------------------------------------
    # 主训练流程
    # ------------------------------------------------------------------

    def train(
        self,
        train_path: str,
        valid_path: Optional[str] = None,
        test_path: Optional[str] = None,
        max_length: int = 1024,
        pack_sequences: bool = True,
        max_samples: Optional[int] = None,
        text_field: str = "text",
        resume_from_checkpoint: bool = False,
        eval_batch_size: int = 4,
    ) -> Dict[str, Any]:
        """执行增量预训练，训练结束后评估 PPL 和 Accuracy。

        Args:
            train_path: 训练集文件路径
            valid_path: 验证集文件路径（None 则训练时不做在线 eval）
            test_path:  测试集文件路径（None 则跳过最终测试评估）
            max_length: 最大序列长度
            pack_sequences: 是否使用 packed 模式（推荐）
            max_samples: 最大训练样本数，None 表示全量
            text_field: json/jsonl 中的文本字段名
            resume_from_checkpoint: 是否从最新 checkpoint 续训
            eval_batch_size: 最终评估时的 batch size

        Returns:
            包含 output_dir 和各阶段评估指标的字典
        """
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()

        # ── 构建数据集 ───────────────────────────────────────────────────
        logger.info("构建训练数据集 ...")
        train_dataset = build_split_dataset(
            file_path=train_path, tokenizer=self.tokenizer,
            max_length=max_length, pack_sequences=pack_sequences,
            max_samples=max_samples, text_field=text_field,
        )
        logger.info(f"训练集: {len(train_dataset)} 个样本/chunk")

        valid_dataset = None
        if valid_path:
            logger.info("构建验证数据集 ...")
            valid_dataset = build_split_dataset(
                file_path=valid_path, tokenizer=self.tokenizer,
                max_length=max_length, pack_sequences=pack_sequences,
                text_field=text_field,
            )
            logger.info(f"验证集: {len(valid_dataset)} 个样本/chunk")

        test_dataset = None
        if test_path:
            logger.info("构建测试数据集 ...")
            test_dataset = build_split_dataset(
                file_path=test_path, tokenizer=self.tokenizer,
                max_length=max_length, pack_sequences=pack_sequences,
                text_field=text_field,
            )
            logger.info(f"测试集: {len(test_dataset)} 个样本/chunk")

        # ── 构建 Trainer ─────────────────────────────────────────────────
        training_args = self._build_training_arguments(has_eval=valid_dataset is not None)
        collator = (
            default_data_collator if pack_sequences
            else PretrainDataCollator(pad_token_id=self.tokenizer.pad_token_id)
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=collator,
        )

        # ── 续训 checkpoint ──────────────────────────────────────────────
        checkpoint = None
        if resume_from_checkpoint and os.path.isdir(self.output_dir):
            ckpt_dirs = sorted(
                [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[-1]),
            )
            if ckpt_dirs:
                checkpoint = os.path.join(self.output_dir, ckpt_dirs[-1])
                logger.info(f"从 checkpoint 恢复: {checkpoint}")

        # ── 训练 ─────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("开始增量预训练 ...")
        self.trainer.train(resume_from_checkpoint=checkpoint)

        logger.info(f"保存模型到 {self.output_dir}")
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # ── 训练后评估 ───────────────────────────────────────────────────
        results: Dict[str, Any] = {"output_dir": self.output_dir}
        logger.info("=" * 60)
        logger.info("训练后评估阶段")

        for split_name, dataset in [("valid", valid_dataset), ("test", test_dataset)]:
            if dataset is not None:
                metrics = self.evaluate_ppl_and_accuracy(
                    dataset=dataset, split_name=split_name,
                    batch_size=eval_batch_size, pack_sequences=pack_sequences,
                )
                results[split_name] = metrics

        # 保存评估结果
        results_path = os.path.join(self.output_dir, "cpt_eval_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("=" * 60)
        logger.info("增量预训练完成！评估结果汇总:")
        for split, metrics in results.items():
            if isinstance(metrics, dict):
                logger.info(
                    f"  [{split}] PPL={metrics['ppl']:.2f}  "
                    f"Accuracy={metrics['accuracy'] * 100:.2f}%"
                )
        logger.info(f"评估结果已保存至: {results_path}")
        logger.info("=" * 60)

        return results


# ---------------------------------------------------------------------------
# 工厂函数：从 YAML 配置构建 CPTTrainer
# ---------------------------------------------------------------------------

def build_cpt_trainer_from_config(config: Dict[str, Any]) -> CPTTrainer:
    """从 cpt_config.yaml 解析的字典构建 CPTTrainer。"""
    model_cfg = config.get("model", {})
    lora_cfg  = config.get("lora", {})
    quant_cfg = config.get("quantization", {})
    train_cfg = config.get("training", {})

    return CPTTrainer(
        model_name_or_path=model_cfg.get("base_model_path", ""),
        output_dir=model_cfg.get("output_dir", "model_output/cpt"),
        use_lora=lora_cfg.get("enabled", True),
        lora_r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        lora_bias=lora_cfg.get("bias", "none"),
        lora_target_modules=lora_cfg.get("target_modules", None),
        use_4bit=quant_cfg.get("use_4bit", False),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", False),
        training_args=train_cfg,
    )
