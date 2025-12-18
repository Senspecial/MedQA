
import os

import torch
from typing import Dict, List, Optional, Union, Any
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset

from src.utils.logger import setup_logger
from ..dataset.medical_dataset import MedicalDataset
from peft import (  # 新增
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


logger = setup_logger(__name__)

class SFTTrainer:
    """基于Hugging Face Trainer的SFT训练器"""
    
    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        training_args: Optional[Dict[str, Any]] = None,
        # ↓↓↓ 新增一些 QLoRA 相关可选参数
        use_qlora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.training_args = training_args or {}

        self.use_qlora = use_qlora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # Qwen2.5 推荐的 LoRA 作用模块
        self.lora_target_modules = lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info(f"Loading model and tokenizer from {self.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 确保分词器有正确的填充token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.use_qlora and torch.cuda.is_available():
            logger.info("Using QLoRA: 4bit quantization + LoRA")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            # 明确指定当前 GPU 作为 device_map，而不是 "auto"
            device_index = torch.cuda.current_device()  # 一般是 0
            device_map = {"": device_index}

            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )
            base_model = prepare_model_for_kbit_training(base_model)
            base_model.config.use_cache = False  # 配合 gradient_checkpointing

            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.lora_target_modules,
            )

            self.model = get_peft_model(base_model, lora_config)
            logger.info("QLoRA model built with LoRA adapters.")
        else:
            # 回退到普通全参数微调（不推荐在大模型上，但逻辑保留）
            logger.info("QLoRA disabled or no CUDA found, using full-precision model.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name_or_path,
        #     trust_remote_code=True,
        #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        #     device_map="auto" if torch.cuda.is_available() else None
        # )
        
        # 为适应医疗问答对话，调整模型配置
        if hasattr(self.model.config, "max_length"):
            self.model.config.max_length = 2048
        
        return self.model, self.tokenizer
    
    def prepare_dataset(self, medical_dataset: MedicalDataset) -> Dataset:
        """准备训练数据集"""
        logger.info("Preparing dataset for SFT")
        return medical_dataset.get_sft_dataset(self.tokenizer)
    
    def create_trainer(self, train_dataset, eval_dataset=None):
        """创建Trainer实例"""
        default_args = {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-5,
            "num_train_epochs": 3,
            "logging_steps": 10,
            "save_steps": 200,
            "eval_strategy": "steps" if eval_dataset else "no",
            "eval_steps": 200 if eval_dataset else None,
            "save_total_limit": 3,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            # QLoRA 一般配合 8bit 优化器 & 梯度检查点
            "optim": "paged_adamw_8bit" if (self.use_qlora and torch.cuda.is_available()) else "adamw_torch",
            "gradient_checkpointing": True if (self.use_qlora and torch.cuda.is_available()) else False,
            "fp16": torch.cuda.is_available(),  # 有显卡就开混合精度
            "report_to": "tensorboard",
            "remove_unused_columns": False,
            "load_best_model_at_end": True if eval_dataset is not None else False,
        }
        
        # 更新默认参数
        for key, value in self.training_args.items():
            default_args[key] = value
        
        training_args = TrainingArguments(**default_args)
        
        def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            """自定义数据整理函数"""
            batch = {}
            for key in features[0].keys():
                if key in ["input_ids", "attention_mask", "labels"]:
                    batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
            return batch
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        return self.trainer
    
    def train(self, medical_dataset: MedicalDataset, eval_split: float = 0.1):
        """执行SFT训练"""
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # 准备数据集
        dataset = medical_dataset.get_sft_dataset(self.tokenizer)
        
        # 划分训练和评估数据集
        if eval_split > 0:
            dataset = dataset.train_test_split(test_size=eval_split)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
        else:
            train_dataset = dataset
            eval_dataset = None
        
        # 创建训练器
        self.create_trainer(train_dataset, eval_dataset)
        
        # 开始训练
        logger.info("Starting SFT training")
        self.trainer.train()
        
        # 保存最终模型
        logger.info(f"Saving final model to {self.output_dir}")
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return self.output_dir






if __name__ == "__main__":
    pass