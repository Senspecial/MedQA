
import os
import json
import random
import yaml
from typing import Dict, List, Optional, Union, Any, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_system_prompt_from_config(config_path: str = "config/system_prompt.yaml") -> str:
    """
    从配置文件加载完整的系统提示
    
    Args:
        config_path: 系统提示配置文件路径（默认: config/system_prompt.yaml）
        
    Returns:
        系统提示字符串
    """
    try:
        # 尝试多个可能的路径
        possible_paths = [
            config_path,  # 直接路径
            os.path.join('/root/autodl-tmp/MedQA', config_path),  # 绝对路径
            os.path.join(os.getcwd(), config_path),  # 当前工作目录
        ]
        
        for full_path in possible_paths:
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config and 'system_prompt' in config:
                        prompt = config['system_prompt']
                        if len(prompt) > 1000:  # 完整版应该>1000字符
                            logger.info(f"成功从 {config_path} 加载完整系统提示")
                            return prompt
    except Exception as e:
        logger.warning(f"无法加载系统提示配置文件: {e}，使用默认简化版本")
    
    logger.warning("未找到完整系统提示配置，使用默认简化版本")
    return get_default_system_prompt()

def get_default_system_prompt() -> str:
    """
    获取默认系统提示（简化版本，作为后备）
    
    Returns:
        默认系统提示字符串
    """
    return """你是一个专业的医疗知识助手，具备全科医学知识，旨在为用户提供准确、安全、实用的医疗健康信息。

请遵循以下原则：
1. 使用不确定性表述（"可能是"、"考虑"、"常见原因包括"）
2. 建议检查项目和就医科室，但不做明确诊断
3. 可以提供药物类别建议（不含具体剂量）
4. 严重症状必须建议就医（胸痛、呼吸困难、意识改变等）
5. 不确定时明确说明，引导专业就医
6. 不编造信息，基于循证医学知识"""

class MedicalDataset:
    """医疗数据集处理类"""
    
    def __init__(
        self,
        data: Union[str, List[Dict]],  # 支持路径或数据列表
        dataset_type: str = "sft",  # "sft" 或 "dpo"
        max_length: int = 512,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化医疗数据集
        
        Args:
            data: 数据路径（str）或数据列表（List[Dict]）
            dataset_type: 数据集类型
            max_length: 最大长度
            system_prompt: 系统提示（如果为None，会尝试从配置文件加载完整版本）
        """
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.raw_data = []
        
        # 系统提示：优先使用传入的，否则从配置文件加载完整版本
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            # 尝试加载完整的系统提示
            self.system_prompt = load_system_prompt_from_config()
            logger.info("已加载完整版系统提示")
        
        # 加载数据
        if isinstance(data, str):
            # 传入的是文件路径
            self.data_path = data
            self._load_data_from_file()
        elif isinstance(data, list):
            # 传入的是数据列表
            self.data_path = None
            self.raw_data = data
            logger.info(f"Loaded {len(self.raw_data)} examples from data list")
        else:
            raise ValueError(f"data must be str (path) or list (data), got {type(data)}")
    
    def _load_data_from_file(self):
        """从文件加载医疗数据集"""
        logger.info(f"Loading medical dataset from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file {self.data_path} not found")
        
        if self.data_path.endswith(".json"):
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.raw_data = json.load(f)
        elif self.data_path.endswith(".jsonl"):
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.raw_data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
        
        logger.info(f"Loaded {len(self.raw_data)} examples")
    
    # 保留向后兼容
    def _load_data(self):
        """加载医疗数据集（向后兼容）"""
        if self.data_path:
            self._load_data_from_file()
    
    def _format_sft_example(self, example: Dict) -> Dict:
        """格式化SFT样本"""
        # 支持多种字段名
        query = example.get("question") or example.get("query") or example.get("instruction") or ""
        response = example.get("answer") or example.get("response") or example.get("output") or ""
        
        if not query or not response:
            logger.warning(f"Empty query or response found in data: {example.keys()}")
            return None
        
        # 构建Qwen2的输入格式
        # 基于Chatml格式
        formatted_text = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        formatted_text += f"<|im_start|>user\n{query}<|im_end|>\n"
        formatted_text += f"<|im_start|>assistant\n{response}<|im_end|>"
        
        return {"text": formatted_text}
    
    def _format_dpo_example(self, example: Dict) -> Dict:
        """格式化DPO样本"""
        query = example.get("question", "")
        chosen = example.get("chosen", "") or example.get("response", "")
        rejected = example.get("rejected", "")
        
        if not query or not chosen or not rejected:
            logger.warning("Missing query, chosen or rejected response in DPO data")
            return None
        
        # 构建查询部分
        query_text = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        query_text += f"<|im_start|>user\n{query}<|im_end|>\n"
        query_text += f"<|im_start|>assistant\n"
        
        # 构建响应部分（只包含response，不包括之前的对话）
        chosen_text = chosen
        rejected_text = rejected
        
        return {
            "query": query_text,
            "chosen": chosen_text,
            "rejected": rejected_text
        }
    
    def get_sft_dataset(self, tokenizer: PreTrainedTokenizer) -> Dataset:
        """获取SFT格式的数据集"""
        if self.dataset_type != "sft":
            logger.warning(f"Dataset type is {self.dataset_type}, but requesting SFT dataset")
        
        # 格式化数据
        formatted_data = []
        for example in self.raw_data:
            formatted_example = self._format_sft_example(example)
            if formatted_example:
                formatted_data.append(formatted_example)
        
        dataset = Dataset.from_list(formatted_data)
        
        # 对数据进行分词处理
        def tokenize_function(examples):
            outputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # 为了训练语言模型，labels与input_ids相同
            outputs["labels"] = outputs["input_ids"].clone()
            
            return outputs
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
    
    def get_dpo_dataset(self, tokenizer: PreTrainedTokenizer, max_length: Optional[int] = None, max_prompt_length: Optional[int] = None, max_response_length: Optional[int] = None,) -> Dataset:
        """获取DPO格式的数据集"""
        if self.dataset_type != "dpo":
            logger.warning(f"Dataset type is {self.dataset_type}, but requesting DPO dataset")
        
        max_length = max_length or self.max_length
        
        # 格式化数据
        formatted_data = []
        for example in self.raw_data:
            formatted_example = self._format_dpo_example(example)
            if formatted_example:
                formatted_data.append(formatted_example)
        
        dataset = Dataset.from_list(formatted_data)
        
        # 对数据进行分词处理
        def tokenize_function(examples):
            # 处理查询部分
            query_tokenized = tokenizer(
                examples["query"],
                truncation=True,
                max_length=max_prompt_length,  
                padding="max_length",
                return_tensors="np"
            )
            
            # 处理选择回复部分
            chosen_tokenized = tokenizer(
                examples["chosen"],
                truncation=True,
                max_length=max_response_length,  
                padding="max_length",
                return_tensors="np"
            )
            
            # 处理拒绝回复部分
            rejected_tokenized = tokenizer(
                examples["rejected"],
                truncation=True,
                max_length=max_response_length,  
                padding="max_length",
                return_tensors="np"
            )
            
            return {
                "query_ids": query_tokenized["input_ids"],
                "query_attention_mask": query_tokenized["attention_mask"],
                "chosen_ids": chosen_tokenized["input_ids"],
                "chosen_attention_mask": chosen_tokenized["attention_mask"],
                "rejected_ids": rejected_tokenized["input_ids"],
                "rejected_attention_mask": rejected_tokenized["attention_mask"],
            }
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["query", "chosen", "rejected"]
        )
        
        return tokenized_dataset

