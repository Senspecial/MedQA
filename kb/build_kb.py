"""
构建本地医学知识库：从 QA 对或百科语料加载文档，生成向量嵌入，构建并保存 FAISS 索引。

支持两种数据格式：
  qa   - QA 对格式，每行 {"instruction": "问题", "input": "", "output": "答案"}
  text - 百科文本格式，每行 {"text": "..."}

用法：
    cd kb/
    python build_kb.py                         # 使用 config.yaml 默认配置
    python build_kb.py --max-docs 50000        # 限制文档数
    python build_kb.py --embed-field answer    # 用答案做嵌入
    python build_kb.py --format text           # 切换为百科文本模式
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import yaml


# ──────────────────────────── 工具函数 ────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def truncate(text: str, max_len: int) -> str:
    """按字符截断，尽量在句子边界处截断。"""
    if len(text) <= max_len:
        return text
    # 在 max_len 附近找最近的句子结束符
    for sep in ["。", "？", "！", "；", "\n"]:
        idx = text.rfind(sep, max_len // 2, max_len)
        if idx != -1:
            return text[: idx + 1]
    return text[:max_len]


# ──────────────────────────── 文档解析 ────────────────────────────────────────

def parse_qa(data: dict, embed_field: str, max_text_length: int,
             min_answer_length: int) -> Optional[dict]:
    """
    解析一条 QA 对，返回知识库文档格式：
    {
        "title":    问题（用于展示）,
        "contents": 问题 + 答案 拼接（用于展示和检索上下文）,
        "embed_text": 实际用于生成嵌入向量的文本
    }

    embed_field 取值：
        "question"         - 只用问题编码，语义最接近用户查询
        "answer"           - 只用答案编码，适合关键词/症状匹配
        "question_answer"  - 问答拼接编码，覆盖广
    """
    question = data.get("instruction", "").strip()
    answer = data.get("output", "").strip()

    if not question or not answer:
        return None
    if len(answer) < min_answer_length:
        return None

    question = truncate(question, max_text_length)
    answer = truncate(answer, max_text_length)

    # 完整 QA 文本（存入 corpus，检索命中后返回给模型）
    full_text = f"问题：{question}\n答案：{answer}"

    if embed_field == "question":
        embed_text = question
    elif embed_field == "answer":
        embed_text = answer
    elif embed_field == "question_answer":
        embed_text = f"{question} {answer}"
    else:
        raise ValueError(f"未知的 embed_field: {embed_field}")

    return {
        "title": question,
        "contents": full_text,
        "embed_text": embed_text,
    }


def parse_text_doc(data: dict, max_text_length: int) -> Optional[dict]:
    """解析百科文本格式，每行 {"text": "..."}"""
    text = data.get("text", "").strip()
    if not text:
        return None

    text = truncate(text, max_text_length)

    # 提取标题：找第一个标点前的内容（不超过 60 字符）
    title = ""
    for sep in ["？", "?", "。", ".", "\n"]:
        idx = text.find(sep)
        if 0 < idx < 60:
            title = text[: idx + 1].strip()
            break
    if not title:
        title = text[:40].strip()

    return {
        "title": title,
        "contents": text,
        "embed_text": text,
    }


# ──────────────────────────── 数据加载 ────────────────────────────────────────

def load_documents(cfg: dict, base_dir: str) -> Tuple[List[dict], List[str]]:
    """
    加载所有数据源，返回 (docs, embed_texts)。
    - docs:        存入 corpus.jsonl 的完整文档列表（含 title, contents）
    - embed_texts: 对应每个 doc 实际用于编码的文本列表
    """
    data_cfg = cfg["data"]
    fmt = data_cfg["format"]
    max_docs = data_cfg.get("max_docs")
    max_text_length = data_cfg.get("max_text_length", 512)

    # QA 格式专属参数
    embed_field = data_cfg.get("embed_field", "question")
    min_answer_length = data_cfg.get("min_answer_length", 10)

    docs: List[dict] = []
    embed_texts: List[str] = []
    total_loaded = 0
    total_skipped = 0

    for rel_path in data_cfg["sources"]:
        filepath = os.path.abspath(os.path.join(base_dir, rel_path))
        if not os.path.exists(filepath):
            print(f"[WARN] 文件不存在，跳过: {filepath}")
            continue

        print(f"[INFO] 加载: {filepath}")
        file_count = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if max_docs is not None and total_loaded >= max_docs:
                    break

                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    total_skipped += 1
                    continue

                if fmt == "qa":
                    doc = parse_qa(data, embed_field, max_text_length, min_answer_length)
                else:
                    doc = parse_text_doc(data, max_text_length)

                if doc is None:
                    total_skipped += 1
                    continue

                embed_texts.append(doc.pop("embed_text"))  # 编码文本单独保存
                docs.append(doc)
                file_count += 1
                total_loaded += 1

        print(f"[INFO]   → 本文件加载 {file_count} 条")

        if max_docs is not None and total_loaded >= max_docs:
            print(f"[INFO] 已达最大文档数 {max_docs}，停止")
            break

    print(f"[INFO] 共加载 {len(docs)} 条文档，跳过 {total_skipped} 条（空/不合格）")
    return docs, embed_texts


# ──────────────────────────── 向量编码 ────────────────────────────────────────

def build_embeddings(embed_texts: List[str], model_name: str,
                     batch_size: int, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"[INFO] 加载嵌入模型: {model_name} (device={device})")
    model = SentenceTransformer(model_name, device=device)

    total = len(embed_texts)
    print(f"[INFO] 开始编码 {total} 条文本（batch_size={batch_size}）...")
    t0 = time.time()

    embeddings = model.encode(
        embed_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 归一化 → 内积 = 余弦相似度
        convert_to_numpy=True,
    )

    elapsed = time.time() - t0
    print(f"[INFO] 编码完成，耗时 {elapsed:.1f}s，维度: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ──────────────────────────── FAISS 索引 ────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray, index_type: str,
                      nlist: int, nprobe: int):
    import faiss

    n, dim = embeddings.shape
    print(f"[INFO] 构建 FAISS 索引（类型={index_type}, n={n}, dim={dim}）")

    if index_type == "flat" or n < 10000:
        index = faiss.IndexFlatIP(dim)
        if n < 10000:
            print(f"[INFO] 文档数 < 10000，自动使用 IndexFlatIP（精确搜索）")
        else:
            print(f"[INFO] 使用 IndexFlatIP（精确搜索）")
    else:
        actual_nlist = min(nlist, n // 10)
        actual_nlist = max(actual_nlist, 1)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, actual_nlist,
                                   faiss.METRIC_INNER_PRODUCT)
        print(f"[INFO] 使用 IndexIVFFlat（nlist={actual_nlist}, nprobe={nprobe}）")
        print(f"[INFO] 训练聚类中心...")
        index.train(embeddings)
        index.nprobe = nprobe

    print(f"[INFO] 添加向量...")
    index.add(embeddings)
    print(f"[INFO] 索引构建完成，共 {index.ntotal} 条向量")
    return index


# ──────────────────────────── 保存 ───────────────────────────────────────────

def save(index, docs: List[dict], cfg: dict, index_dir: str):
    import faiss

    os.makedirs(index_dir, exist_ok=True)
    faiss_path = os.path.join(index_dir, cfg["index"]["faiss_file"])
    corpus_path = os.path.join(index_dir, cfg["index"]["corpus_file"])
    meta_path = os.path.join(index_dir, "meta.json")

    print(f"[INFO] 保存 FAISS 索引 → {faiss_path}")
    faiss.write_index(index, faiss_path)

    print(f"[INFO] 保存语料库 → {corpus_path}")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    meta = {
        "doc_count": len(docs),
        "data_format": cfg["data"]["format"],
        "embed_field": cfg["data"].get("embed_field", "question"),
        "embedding_model": cfg["embedding"]["model_name"],
        "index_type": cfg["faiss"]["index_type"],
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 保存元数据 → {meta_path}")
    print(f"\n{'='*60}")
    print(f"✓ 知识库构建完成！")
    print(f"  文档数:   {len(docs)}")
    print(f"  数据格式: {meta['data_format']}  (embed_field={meta['embed_field']})")
    print(f"  嵌入模型: {meta['embedding_model']}")
    print(f"  索引目录: {os.path.abspath(index_dir)}")
    print(f"{'='*60}")


# ──────────────────────────── 入口 ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="构建本地医学知识库")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-docs", type=int, default=None, help="最大文档数")
    parser.add_argument("--format", choices=["qa", "text"], default=None,
                        help="数据格式（qa / text）")
    parser.add_argument("--embed-field",
                        choices=["question", "answer", "question_answer"],
                        default=None, help="QA 格式嵌入字段")
    parser.add_argument("--model", type=str, default=None, help="嵌入模型")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)

    cfg = load_config(config_path)

    # 命令行参数覆盖配置文件
    if args.max_docs is not None:
        cfg["data"]["max_docs"] = args.max_docs
    if args.format is not None:
        cfg["data"]["format"] = args.format
    if args.embed_field is not None:
        cfg["data"]["embed_field"] = args.embed_field
    if args.model is not None:
        cfg["embedding"]["model_name"] = args.model
    if args.device is not None:
        cfg["embedding"]["device"] = args.device

    index_dir = os.path.join(script_dir, cfg["index"]["dir"])

    print("=" * 60)
    print("医学知识库构建程序")
    print("=" * 60)
    print(f"数据格式:   {cfg['data']['format']}")
    if cfg["data"]["format"] == "qa":
        print(f"嵌入字段:   {cfg['data'].get('embed_field', 'question')}")
    print(f"最大文档数: {cfg['data']['max_docs']}")
    print(f"嵌入模型:   {cfg['embedding']['model_name']}")
    print(f"设备:       {cfg['embedding']['device']}")
    print(f"索引目录:   {index_dir}")
    print("=" * 60)

    docs, embed_texts = load_documents(cfg, script_dir)
    if not docs:
        print("[ERROR] 未加载到任何文档，请检查数据路径配置")
        sys.exit(1)

    embeddings = build_embeddings(
        embed_texts,
        model_name=cfg["embedding"]["model_name"],
        batch_size=cfg["embedding"]["batch_size"],
        device=cfg["embedding"]["device"],
    )

    index = build_faiss_index(
        embeddings,
        index_type=cfg["faiss"]["index_type"],
        nlist=cfg["faiss"]["nlist"],
        nprobe=cfg["faiss"]["nprobe"],
    )

    save(index, docs, cfg, index_dir)


if __name__ == "__main__":
    main()
