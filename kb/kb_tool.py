"""
医学知识库离线搜索工具（供 Agent-R1 直接调用，无需 HTTP 服务）

注册到 Agent-R1 工具链后，可在 agent_trainer.yaml 中配置：
    tool.tools: ['med_search']
"""

import json
import os
from typing import Dict, List, Any, Optional

import numpy as np


class MedSearchTool:
    """
    医学知识库本地 FAISS 搜索工具，接口兼容 Agent-R1 的 BaseTool。

    可作为独立工具类直接导入，也可通过继承 BaseTool 注册到 Agent-R1。
    """

    name = "search"
    description = (
        "在本地医学知识库中搜索相关医学信息。"
        "知识库包含中文医学百科文档，涵盖疾病、症状、治疗、预防等内容。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索查询词（支持中文）"}
        },
        "required": ["query"],
    }

    def __init__(self, index_dir: Optional[str] = None, model_name: Optional[str] = None,
                 device: str = "cpu", top_k: int = 5, config_path: Optional[str] = None):
        """
        Args:
            index_dir:    FAISS 索引目录路径，默认自动定位 kb/index/
            model_name:   嵌入模型名称，默认从 config.yaml 读取
            device:       推理设备（cuda/cpu）
            top_k:        默认返回结果数
            config_path:  配置文件路径，默认自动定位 kb/config.yaml
        """
        import faiss
        from sentence_transformers import SentenceTransformer

        # 自动定位 kb/ 目录
        kb_dir = os.path.dirname(os.path.abspath(__file__))

        # 读取配置
        if config_path is None:
            config_path = os.path.join(kb_dir, "config.yaml")
        cfg = self._load_config(config_path)

        if index_dir is None:
            index_dir = os.path.join(kb_dir, cfg["index"]["dir"])
        if model_name is None:
            model_name = cfg["embedding"]["model_name"]

        index_dir = os.path.abspath(index_dir)
        faiss_path = os.path.join(index_dir, cfg["index"]["faiss_file"])
        corpus_path = os.path.join(index_dir, cfg["index"]["corpus_file"])

        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"FAISS 索引文件不存在: {faiss_path}\n"
                f"请先在 kb/ 目录下运行 `python build_kb.py` 构建知识库。"
            )

        print(f"[MedSearchTool] 加载 FAISS 索引: {faiss_path}")
        self.index = faiss.read_index(faiss_path)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = cfg["faiss"]["nprobe"]

        print(f"[MedSearchTool] 加载语料库: {corpus_path}")
        self.corpus: List[dict] = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.corpus.append(json.loads(line))

        print(f"[MedSearchTool] 加载嵌入模型: {model_name} (device={device})")
        self.model = SentenceTransformer(model_name, device=device)
        self.query_instruction = cfg["embedding"]["query_instruction"]
        self.default_top_k = top_k

        print(f"[MedSearchTool] ✓ 初始化完成，知识库共 {len(self.corpus)} 条文档")

    @staticmethod
    def _load_config(config_path: str) -> dict:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def execute(self, args: Dict) -> Dict[str, Any]:
        """
        单条搜索（兼容 BaseTool.execute 接口）。

        Returns:
            {"content": json_string, "success": bool}
        """
        try:
            query = args.get("query", "").strip()
            top_k = args.get("top_k", self.default_top_k)
            results = self._search(query, top_k=top_k)
            return {"content": json.dumps({"results": results}, ensure_ascii=False), "success": True}
        except Exception as e:
            return {"content": str(e), "success": False}

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        批量搜索（兼容 BaseTool.batch_execute 接口）。
        """
        try:
            queries = [a.get("query", "").strip() for a in args_list]
            top_k = max(a.get("top_k", self.default_top_k) for a in args_list)
            batch_results = self._batch_search(queries, top_k=top_k)
            return [
                {"content": json.dumps({"results": r}, ensure_ascii=False), "success": True}
                for r in batch_results
            ]
        except Exception as e:
            return [{"content": str(e), "success": False} for _ in args_list]

    def _search(self, query: str, top_k: int) -> List[dict]:
        results = self._batch_search([query], top_k=top_k)
        return results[0]

    def _batch_search(self, queries: List[str], top_k: int) -> List[List[dict]]:
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
            for score, doc_id in zip(scores[i], ids[i]):
                if doc_id < 0 or doc_id >= len(self.corpus):
                    continue
                doc = self.corpus[doc_id]
                results.append({
                    "title": doc.get("title", ""),
                    "content": doc.get("contents", ""),
                })
            batch_results.append(results)
        return batch_results


# ──────────────────────── Agent-R1 BaseTool 适配器 ────────────────────────────

def create_agent_r1_tool(index_dir=None, model_name=None, device="cuda",
                         top_k=5, config_path=None):
    """
    创建兼容 Agent-R1 BaseTool 接口的医学搜索工具实例。

    示例：
        from kb.kb_tool import create_agent_r1_tool
        tool = create_agent_r1_tool(device="cuda")
    """
    try:
        from agent_r1.tool.base import BaseTool

        class MedSearchAgentTool(BaseTool, MedSearchTool):
            name = "search"
            description = MedSearchTool.description
            parameters = MedSearchTool.parameters

            def __init__(self):
                BaseTool.__init__(self)
                MedSearchTool.__init__(
                    self,
                    index_dir=index_dir,
                    model_name=model_name,
                    device=device,
                    top_k=top_k,
                    config_path=config_path,
                )

            def execute(self, args):
                return MedSearchTool.execute(self, args)

            def batch_execute(self, args_list):
                return MedSearchTool.batch_execute(self, args_list)

        return MedSearchAgentTool()

    except ImportError:
        # 未安装 agent_r1 时退回到纯工具类
        return MedSearchTool(
            index_dir=index_dir,
            model_name=model_name,
            device=device,
            top_k=top_k,
            config_path=config_path,
        )


if __name__ == "__main__":
    # 简单测试
    import sys
    tool = MedSearchTool(device="cpu")

    queries = ["糖尿病的症状有哪些", "高血压如何治疗", "感冒发烧怎么办"]
    for q in queries:
        result = tool.execute({"query": q})
        print(f"\n查询: {q}")
        data = json.loads(result["content"])
        for i, r in enumerate(data["results"][:2], 1):
            print(f"  [{i}] {r['title']}")
            print(f"      {r['content'][:100]}...")
