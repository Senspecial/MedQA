"""
SFT Trainer (Supervised Fine-Tuning Trainer)

基于 Hugging Face Trainer，对医疗问答数据进行有监督微调。
支持 QLoRA / LoRA / 全参数三种训练模式。
精度（bf16/fp16/fp32）统一从 training_args 读取，不硬编码。
训练完成后可对测试集评估 PPL、ROUGE-L、BLEU-4、BERTScore。
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.utils.logger import setup_logger
from ..dataset.medical_dataset import MedicalDataset

logger = setup_logger(__name__)


class SFTTrainer:
    """医疗问答有监督微调训练器。

    训练模式（由配置自动选择）：
      - QLoRA : use_qlora=True  + use_4bit=True   → 省显存，速度较慢
      - LoRA  : use_qlora=True  + use_4bit=False  → 速度快，需更多显存
      - 全参数: use_qlora=False                   → 最快，显存需求最高

    模型精度由 training_args 中的 bf16/fp16 字段决定，不硬编码。
    """

    DEFAULT_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        training_args: Optional[Dict[str, Any]] = None,
        use_qlora: bool = True,
        use_4bit: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        cpt_adapter_path: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.training_args = training_args or {}
        self.cpt_adapter_path = cpt_adapter_path

        self.use_qlora = use_qlora
        self.use_4bit  = use_4bit and use_qlora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or self.DEFAULT_TARGET_MODULES

        self.tokenizer = None
        self.model = None
        self.trainer = None

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _get_model_dtype(self) -> tuple[torch.dtype, str]:
        if self.training_args.get("bf16", False):
            return torch.bfloat16, "bf16"
        if self.training_args.get("fp16", False):
            return torch.float16, "fp16"
        return torch.float32, "fp32"

    def _clean_training_args(self, args: Dict) -> Dict:
        """去掉内部私有键（以 _ 开头），避免传入 TrainingArguments。"""
        return {k: v for k, v in args.items() if not k.startswith("_")}

    # ------------------------------------------------------------------
    # 模型 & Tokenizer 加载
    # ------------------------------------------------------------------

    def load_model_and_tokenizer(self):
        logger.info(f"加载模型: {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        use_cuda = torch.cuda.is_available()
        model_dtype, dtype_name = self._get_model_dtype()

        # ── 1. 加载 base model ──────────────────────────────────────────────
        if self.use_4bit and use_cuda:
            logger.info(f"模式: QLoRA  精度: 4bit(NF4) compute={dtype_name}")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.training_args.get("_bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=model_dtype,
                bnb_4bit_use_double_quant=self.training_args.get("_bnb_4bit_use_double_quant", False),
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                quantization_config=bnb_config,
                device_map={"": torch.cuda.current_device()},
                trust_remote_code=True,
            )
            base_model = prepare_model_for_kbit_training(base_model)
        else:
            mode = f"LoRA ({dtype_name})" if self.use_qlora else f"全参数 ({dtype_name})"
            logger.info(f"模式: {mode}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=model_dtype if use_cuda else torch.float32,
                device_map={"": torch.cuda.current_device()} if use_cuda else None,
                trust_remote_code=True,
            )

        base_model.config.use_cache = False

        # ── 2. 动态挂载 CPT LoRA adapter（如有），merge 后继续 ──────────────
        if self.cpt_adapter_path:
            logger.info(f"动态加载 CPT adapter: {self.cpt_adapter_path}")
            base_model = PeftModel.from_pretrained(
                base_model,
                self.cpt_adapter_path,
                is_trainable=False,
            )
            logger.info("CPT adapter 加载完成，开始 merge_and_unload() ...")
            base_model = base_model.merge_and_unload()
            logger.info("CPT adapter 已 merge 进 base model，无需额外显存")

        # ── 3. 套 SFT LoRA ──────────────────────────────────────────────────
        if self.use_qlora:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.lora_target_modules,
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()
        else:
            self.model = base_model

        return self.model, self.tokenizer

    # ------------------------------------------------------------------
    # Trainer 创建
    # ------------------------------------------------------------------

    def create_trainer(self, train_dataset, eval_dataset=None):
        use_cuda = torch.cuda.is_available()
        default_optim = "paged_adamw_8bit" if (self.use_4bit and use_cuda) else "adamw_torch"

        defaults: Dict[str, Any] = {
            "output_dir": self.output_dir,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "optim": default_optim,
            "fp16": False,
            "bf16": False,
            "gradient_checkpointing": use_cuda,
            "logging_steps": 10,
            "save_steps": 200,
            "save_total_limit": 3,
            "eval_strategy": "steps" if eval_dataset else "no",
            "eval_steps": 200 if eval_dataset else None,
            "load_best_model_at_end": eval_dataset is not None,
            "report_to": "tensorboard",
            "remove_unused_columns": False,
            "seed": 42,
        }
        defaults.update(self._clean_training_args(self.training_args))
        if not eval_dataset:
            defaults.pop("eval_steps", None)
            defaults["eval_strategy"] = "no"
            defaults["load_best_model_at_end"] = False

        _, dtype_name = self._get_model_dtype()
        logger.info(
            f"训练精度: {dtype_name}  优化器: {defaults['optim']}  "
            f"梯度检查点: {defaults['gradient_checkpointing']}"
        )

        def data_collator(features):
            batch = {}
            for key in ["input_ids", "attention_mask", "labels"]:
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
            return batch

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(**defaults),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        return self.trainer

    # ------------------------------------------------------------------
    # 训练主流程
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Union[MedicalDataset, Dataset] = None,
        eval_dataset: Union[MedicalDataset, Dataset] = None,
        medical_dataset: MedicalDataset = None,
        eval_split: float = 0.1,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()

        if train_dataset is not None:
            logger.info("使用已划分数据集")
            if isinstance(train_dataset, MedicalDataset):
                train_dataset = train_dataset.get_sft_dataset(self.tokenizer)
            if eval_dataset is not None and isinstance(eval_dataset, MedicalDataset):
                eval_dataset = eval_dataset.get_sft_dataset(self.tokenizer)
            logger.info(f"训练集: {len(train_dataset)} 条  验证集: {len(eval_dataset) if eval_dataset else 0} 条")

        elif medical_dataset is not None:
            logger.warning("建议使用已划分数据集，避免测试集数据泄露")
            dataset = medical_dataset.get_sft_dataset(self.tokenizer)
            if eval_split > 0:
                dataset = dataset.train_test_split(test_size=eval_split, seed=42)
                train_dataset = dataset["train"]
                eval_dataset  = dataset["test"]
            else:
                train_dataset = dataset
                eval_dataset  = None
        else:
            raise ValueError("必须提供 train_dataset 或 medical_dataset")

        self.create_trainer(train_dataset, eval_dataset)

        logger.info("=" * 60)
        logger.info("开始 SFT 训练 ...")
        self.trainer.train()

        logger.info(f"保存模型到 {self.output_dir}")
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        return self.output_dir

    # ------------------------------------------------------------------
    # 测试集评估：PPL + ROUGE-L + BLEU-4 + BERTScore
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_ppl(self, test_dataset) -> float:
        """在 SFT tokenized 测试集上计算 PPL。"""
        self.model.eval()
        device = next(self.model.parameters()).device

        def collate_fn(features):
            return {k: torch.tensor([f[k] for f in features], dtype=torch.long)
                    for k in ["input_ids", "attention_mask", "labels"]}

        loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)
        total_loss, n = 0.0, 0
        for batch in loader:
            out = self.model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            total_loss += out.loss.item()
            n += 1

        ppl = math.exp(min(total_loss / max(n, 1), 20))
        self.model.train()
        return ppl

    @torch.no_grad()
    def generate_responses(
        self,
        questions: List[str],
        system_prompt: str,
        max_new_tokens: int = 256,
        batch_size: int = 4,
    ) -> List[str]:
        """批量生成模型回复。"""
        self.model.eval()
        device = next(self.model.parameters()).device
        generated = []

        for i in range(0, len(questions), batch_size):
            batch_qs = questions[i: i + batch_size]
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user",   "content": q}],
                    tokenize=False, add_generation_prompt=True,
                )
                for q in batch_qs
            ]
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=1024,
            ).to(device)
            out_ids = self.model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
            )
            for j, ids in enumerate(out_ids):
                new_ids = ids[enc["input_ids"].shape[1]:]
                generated.append(self.tokenizer.decode(new_ids, skip_special_tokens=True).strip())

        self.model.train()
        return generated

    def compute_generation_metrics(
        self,
        generated: List[str],
        references: List[str],
        bertscore_model_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """计算 ROUGE-L、BLEU-4、BERTScore。"""
        metrics: Dict[str, float] = {}

        # ROUGE
        try:
            from rouge_chinese import Rouge
            rouge = Rouge()
            hyps = [" ".join(g) if g else "无" for g in generated]
            refs = [" ".join(r) if r else "无" for r in references]
            s = rouge.get_scores(hyps, refs, avg=True)
            metrics["rouge_1"] = s["rouge-1"]["f"]
            metrics["rouge_2"] = s["rouge-2"]["f"]
            metrics["rouge_l"] = s["rouge-l"]["f"]
        except Exception as e:
            logger.warning(f"ROUGE 计算失败: {e}")
            metrics.update({"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0})

        # BLEU-4
        try:
            import sacrebleu
            hyps_chr = [" ".join(list(g)) for g in generated]
            refs_chr = [[" ".join(list(r)) for r in references]]
            bleu = sacrebleu.corpus_bleu(hyps_chr, refs_chr, tokenize="char")
            metrics["bleu_4"] = bleu.score / 100.0
        except Exception as e:
            logger.warning(f"BLEU 计算失败: {e}")
            metrics["bleu_4"] = 0.0

        # BERTScore
        try:
            from bert_score import score as bert_score_fn
            kwargs: Dict = {"lang": "zh", "verbose": False, "rescale_with_baseline": False}
            if bertscore_model_path:
                kwargs["model_type"] = bertscore_model_path
            P, R, F1 = bert_score_fn(generated, references, **kwargs)
            metrics["bertscore_p"]  = P.mean().item()
            metrics["bertscore_r"]  = R.mean().item()
            metrics["bertscore_f1"] = F1.mean().item()
        except Exception as e:
            logger.warning(f"BERTScore 计算失败: {e}")
            metrics["bertscore_f1"] = None

        return metrics

    def evaluate_test_set(
        self,
        test_dataset,
        raw_test_path: str,
        system_prompt: str,
        max_new_tokens: int = 256,
        num_samples: Optional[int] = 200,
        bertscore_model_path: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """SFT 训练后的完整测试集评估（PPL + ROUGE + BLEU + BERTScore）。"""
        if self.model is None:
            raise RuntimeError("请先调用 train() 或 load_model_and_tokenizer()")

        results: Dict[str, Any] = {}
        logger.info("=" * 60)
        logger.info("SFT 测试集评估开始")

        # 1. PPL
        logger.info("[1/3] 计算 PPL ...")
        results["ppl"] = self.evaluate_ppl(test_dataset)
        logger.info(f"  PPL = {results['ppl']:.4f}")

        # 2. 加载原始测试数据（自动识别 JSON 数组 / JSONL，流式读取）
        logger.info("[2/3] 加载原始测试数据，准备生成评估 ...")
        with open(raw_test_path, encoding="utf-8") as f:
            first_char = f.read(1)
        if first_char == "[":
            with open(raw_test_path, encoding="utf-8") as f:
                raw_data = json.load(f)
            if not isinstance(raw_data, list):
                raw_data = [raw_data]
        else:
            raw_data = []
            with open(raw_test_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            raw_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        seen, pairs = set(), []
        for item in raw_data:
            # 兼容 instruction/input/output 与 question/answer 两种字段名
            q = str(item.get("question") or item.get("instruction") or "").strip()
            a = str(item.get("answer") or item.get("output") or "").strip()
            if q and a and q not in seen:
                seen.add(q)
                pairs.append({"question": q, "answer": a})

        if num_samples and len(pairs) > num_samples:
            import random
            random.seed(42)
            pairs = random.sample(pairs, num_samples)

        questions  = [p["question"] for p in pairs]
        references = [p["answer"]   for p in pairs]
        logger.info(f"  评估样本数: {len(pairs)}")

        # 3. 生成 + 指标
        logger.info("[3/3] 生成回答并计算 ROUGE / BLEU / BERTScore ...")
        generated = self.generate_responses(
            questions=questions, system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        results.update(self.compute_generation_metrics(
            generated=generated, references=references,
            bertscore_model_path=bertscore_model_path,
        ))

        # 汇总
        logger.info("=" * 60)
        logger.info("测试集评估结果汇总:")
        logger.info(f"  PPL          : {results.get('ppl', 0):.4f}")
        logger.info(f"  ROUGE-1      : {results.get('rouge_1', 0):.4f}")
        logger.info(f"  ROUGE-2      : {results.get('rouge_2', 0):.4f}")
        logger.info(f"  ROUGE-L      : {results.get('rouge_l', 0):.4f}")
        logger.info(f"  BLEU-4       : {results.get('bleu_4', 0):.4f}")
        bs = results.get("bertscore_f1")
        logger.info(f"  BERTScore-F1 : {f'{bs:.4f}' if bs is not None else '未计算'}")
        logger.info("=" * 60)

        if save_path is None:
            save_path = os.path.join(self.output_dir, "sft_eval_results.json")
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存至: {save_path}")

        return results


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_sft_trainer_from_config(config: Dict[str, Any]) -> SFTTrainer:
    """从 sft_config.yaml 解析的字典构建 SFTTrainer。"""
    model_cfg = config.get("model", {})
    lora_cfg  = config.get("lora", {})
    quant_cfg = config.get("quantization", {})
    train_cfg = dict(config.get("training", {}))

    use_4bit = lora_cfg.get("use_4bit", False)
    if use_4bit:
        # 把量化参数以私有键注入，供 load_model_and_tokenizer 读取
        train_cfg["_bnb_4bit_quant_type"]       = quant_cfg.get("bnb_4bit_quant_type", "nf4")
        train_cfg["_bnb_4bit_compute_dtype"]    = quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        train_cfg["_bnb_4bit_use_double_quant"] = quant_cfg.get("bnb_4bit_use_double_quant", False)

    return SFTTrainer(
        model_name_or_path=model_cfg.get("base_model_path", ""),
        output_dir=model_cfg.get("output_dir", "model_output/sft"),
        training_args=train_cfg,
        use_qlora=lora_cfg.get("enabled", True),
        use_4bit=use_4bit,
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        lora_target_modules=lora_cfg.get("target_modules", None),
        cpt_adapter_path=model_cfg.get("cpt_adapter_path", None),
    )
