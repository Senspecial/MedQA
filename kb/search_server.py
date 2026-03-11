"""
医学知识库 HTTP 搜索服务

提供兼容 WikiSearchTool 的 REST API 接口：
  GET  /health                          - 健康检查
  GET  /search?query=...&top_k=5        - 单条查询
  POST /search  {"queries": [...], "top_k": 5}  - 批量查询

用法：
    cd kb/
    python search_server.py [--config config.yaml] [--port 8000]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

# ─────────────────────────────── 知识库加载 ──────────────────────────────────

class KnowledgeBase:
    """封装 FAISS 索引和语料库，提供向量检索功能。"""

    def __init__(self, index_dir: str, model_name: str, device: str,
                 query_instruction: str, nprobe: int = 32):
        import faiss
        from sentence_transformers import SentenceTransformer

        index_dir = os.path.abspath(index_dir)
        faiss_path = os.path.join(index_dir, "index.bin")
        corpus_path = os.path.join(index_dir, "corpus.jsonl")
        meta_path = os.path.join(index_dir, "meta.json")

        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"FAISS 索引文件不存在: {faiss_path}\n"
                f"请先运行 `python build_kb.py` 构建知识库。"
            )

        print(f"[INFO] 加载 FAISS 索引: {faiss_path}")
        self.index = faiss.read_index(faiss_path)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        print(f"[INFO] 加载语料库: {corpus_path}")
        self.corpus: List[dict] = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.corpus.append(json.loads(line))

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"[INFO] 知识库元数据: {meta}")

        print(f"[INFO] 加载嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.query_instruction = query_instruction

        print(f"[INFO] ✓ 知识库就绪，共 {len(self.corpus)} 条文档")

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """单条查询，返回 top_k 个结果，每个结果包含 score 和 document 字段。"""
        results = self.batch_search([query], top_k=top_k)
        return results[0]

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[dict]]:
        """批量查询。"""
        # BGE 模型查询时需加前缀指令
        prefixed = [self.query_instruction + q for q in queries]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores, ids = self.index.search(embeddings, top_k)

        batch_results = []
        for i in range(len(queries)):
            results = []
            for rank, (score, doc_id) in enumerate(zip(scores[i], ids[i])):
                if doc_id < 0 or doc_id >= len(self.corpus):
                    continue
                doc = self.corpus[doc_id]
                results.append({
                    "score": float(score),
                    "document": {
                        "id": int(doc_id),
                        "title": doc.get("title", ""),
                        "contents": doc.get("contents", ""),
                    },
                })
            batch_results.append(results)
        return batch_results


# ─────────────────────────────── FastAPI 应用 ────────────────────────────────

def create_app(kb: KnowledgeBase, default_top_k: int = 5, max_top_k: int = 20):
    from fastapi import FastAPI, Query, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="医学知识库搜索服务", version="1.0.0")

    class BatchSearchRequest(BaseModel):
        queries: List[str]
        top_k: int = default_top_k

    @app.get("/health")
    def health():
        return {"status": "ok", "doc_count": len(kb.corpus)}

    @app.get("/search")
    def search_get(
        query: str = Query(..., description="搜索查询词"),
        top_k: int = Query(default_top_k, ge=1, le=max_top_k, description="返回结果数"),
    ):
        """
        单条搜索接口（GET）

        返回格式兼容 WikiSearchTool：
        {
            "query_results": [{
                "query": "...",
                "results": [{"score": 0.9, "document": {"title": "...", "contents": "..."}}]
            }]
        }
        """
        try:
            results = kb.search(query, top_k=top_k)
            return JSONResponse({
                "query_results": [{
                    "query": query,
                    "results": results,
                }]
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/search")
    def search_post(req: BatchSearchRequest):
        """
        批量搜索接口（POST）

        请求体：{"queries": ["q1", "q2"], "top_k": 5}
        返回格式：{"query_results": [{"query": "q1", "results": [...]}, ...]}
        """
        try:
            top_k = min(req.top_k, max_top_k)
            batch_results = kb.batch_search(req.queries, top_k=top_k)
            query_results = [
                {"query": q, "results": r}
                for q, r in zip(req.queries, batch_results)
            ]
            return JSONResponse({"query_results": query_results})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats")
    def stats():
        """返回知识库统计信息。"""
        return {
            "doc_count": len(kb.corpus),
            "index_ntotal": kb.index.ntotal,
        }

    return app


# ─────────────────────────────── 入口 ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="医学知识库 HTTP 搜索服务")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--host", type=str, default=None, help="监听地址")
    parser.add_argument("--port", type=int, default=None, help="监听端口")
    parser.add_argument("--device", type=str, default=None, help="嵌入模型设备（cuda/cpu）")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 命令行参数覆盖配置
    host = args.host or cfg["server"]["host"]
    port = args.port or cfg["server"]["port"]
    device = args.device or cfg["embedding"]["device"]

    index_dir = os.path.join(script_dir, cfg["index"]["dir"])

    print("=" * 60)
    print("医学知识库搜索服务启动中")
    print("=" * 60)

    kb = KnowledgeBase(
        index_dir=index_dir,
        model_name=cfg["embedding"]["model_name"],
        device=device,
        query_instruction=cfg["embedding"]["query_instruction"],
        nprobe=cfg["faiss"]["nprobe"],
    )

    app = create_app(
        kb=kb,
        default_top_k=cfg["server"]["default_top_k"],
        max_top_k=cfg["server"]["max_top_k"],
    )

    import uvicorn
    print(f"\n[INFO] 服务已启动：http://{host}:{port}")
    print(f"[INFO] API 文档：http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
