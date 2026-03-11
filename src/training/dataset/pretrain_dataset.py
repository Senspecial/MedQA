"""
增量预训练数据集 (Continual Pre-Training Dataset)

支持纯文本 CLM (因果语言建模) 目标，即让模型预测下一个 token。
支持多种输入格式: .txt / .json / .jsonl (含逐行 JSON 的 .json 文件)
支持序列拼接 (pack_sequences) 以充分利用每个 batch。
调用方提供独立的 train / valid / test 文件，无需自动划分。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# 原始文本加载
# ---------------------------------------------------------------------------

def _load_texts_from_file(file_path: str, text_field: str = "text") -> List[str]:
    """从单个文件中读取文本列表。

    支持格式:
      - .txt  : 按段落分割（连续空行作为分隔符）
      - .jsonl: 每行一个 JSON 对象
      - .json : 优先按行解析（JSONL 风格）；若失败则整体解析
      - .csv  : 提取 text_field 列
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    suffix = path.suffix.lower()
    texts: List[str] = []

    if suffix == ".txt":
        raw = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        texts.extend(paragraphs)

    elif suffix in (".json", ".jsonl"):
        texts = _load_jsonl_or_json(path, text_field)

    elif suffix == ".csv":
        import csv
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_field in row and row[text_field].strip():
                    texts.append(row[text_field].strip())

    else:
        raise ValueError(
            f"不支持的文件格式: {suffix}，请使用 .txt / .json / .jsonl / .csv"
        )

    logger.info(f"从 {file_path} 加载了 {len(texts)} 条文本")
    return texts


def _load_jsonl_or_json(path: Path, text_field: str) -> List[str]:
    """优先按逐行 JSONL 方式解析；若不是 JSONL 则整体 JSON 解析。"""
    texts: List[str] = []

    with path.open(encoding="utf-8") as f:
        first_line = f.readline().strip()

    # 判断是否为 JSONL：第一行能被解析为 dict/str
    is_jsonl = False
    try:
        first_obj = json.loads(first_line)
        if isinstance(first_obj, (dict, str)):
            is_jsonl = True
    except (json.JSONDecodeError, ValueError):
        pass

    if is_jsonl:
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"第 {lineno} 行 JSON 解析失败，跳过")
                    continue
                if isinstance(obj, str):
                    texts.append(obj)
                elif isinstance(obj, dict):
                    val = obj.get(text_field, "")
                    if val:
                        texts.append(str(val))
    else:
        # 整体解析标准 JSON
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and text_field in item:
                    texts.append(str(item[text_field]))
        elif isinstance(data, dict) and text_field in data:
            texts.append(str(data[text_field]))

    return texts


def load_pretrain_texts(
    file_path: str,
    text_field: str = "text",
    max_samples: Optional[int] = None,
) -> List[str]:
    """从单个文件加载文本列表，可截断最大样本数。"""
    texts = _load_texts_from_file(file_path, text_field)
    if max_samples is not None:
        texts = texts[:max_samples]
    logger.info(f"使用 {len(texts)} 条文本")
    return texts


# ---------------------------------------------------------------------------
# Dataset: 非拼接模式（每条文本独立截断）
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """CLM 预训练数据集（非拼接模式）。

    每条样本独立 tokenize 并截断到 max_length。
    labels 与 input_ids 相同，padding 位置设为 -100。
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict] = []
        self._build(texts)

    def _build(self, texts: List[str]) -> None:
        logger.info(f"Tokenizing {len(texts)} 条文本 (max_length={self.max_length}) ...")
        for text in texts:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_attention_mask=True,
            )
            if len(encoded["input_ids"]) < 4:
                continue
            self.samples.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            })
        logger.info(f"有效样本数: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        input_ids = torch.tensor(s["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(s["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Dataset: 拼接模式（pack_sequences=True，减少 padding 浪费）
# ---------------------------------------------------------------------------

class PackedPretrainDataset(Dataset):
    """CLM 预训练数据集（拼接模式）。

    将所有 token 拼接成一个长序列，然后按 max_length 均匀切块。
    每个块的 labels = input_ids（全部参与 loss，无 padding）。
    此方式更高效，适合大规模预训练。
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
    ):
        self.max_length = max_length
        self.chunks: List[List[int]] = []
        self._build(texts, tokenizer)

    def _build(self, texts: List[str], tokenizer: PreTrainedTokenizer) -> None:
        logger.info(
            f"Tokenizing & packing {len(texts)} 条文本 (max_length={self.max_length}) ..."
        )
        eos_id = tokenizer.eos_token_id
        all_ids: List[int] = []
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            all_ids.extend(ids)
            if eos_id is not None:
                all_ids.append(eos_id)

        for i in range(0, len(all_ids) - self.max_length + 1, self.max_length):
            self.chunks.append(all_ids[i: i + self.max_length])

        logger.info(f"生成 {len(self.chunks)} 个 packed chunk")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_split_dataset(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
    pack_sequences: bool = True,
    max_samples: Optional[int] = None,
    text_field: str = "text",
) -> Dataset:
    """从单个已划分文件构建 Dataset。

    Args:
        file_path: 数据文件路径（train/valid/test 之一）
        tokenizer: 分词器
        max_length: 最大序列长度
        pack_sequences: 是否使用 packed 模式
        max_samples: 最大样本数（None 表示全量）
        text_field: json 中的文本字段名

    Returns:
        Dataset 实例
    """
    texts = load_pretrain_texts(file_path, text_field=text_field, max_samples=max_samples)
    DatasetCls = PackedPretrainDataset if pack_sequences else PretrainDataset
    return DatasetCls(texts, tokenizer, max_length)


# ---------------------------------------------------------------------------
# Data Collator（支持动态 padding，仅 pack=False 时需要）
# ---------------------------------------------------------------------------

class PretrainDataCollator:
    """将变长样本 pad 到 batch 内最大长度。

    仅在 pack_sequences=False 时使用（packed 模式每个样本等长，无需 pad）。
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].size(0) for f in features)

        input_ids_list, attention_mask_list, labels_list = [], [], []
        for f in features:
            seq_len = f["input_ids"].size(0)
            pad_len = max_len - seq_len

            input_ids_list.append(torch.cat([
                f["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
            ]))
            attention_mask_list.append(torch.cat([
                f["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ]))
            labels_list.append(torch.cat([
                f["labels"],
                torch.full((pad_len,), -100, dtype=torch.long),
            ]))

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }
