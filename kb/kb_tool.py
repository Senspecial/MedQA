"""
医学知识库离线搜索工具（供 Agent-R1 直接调用，无需 HTTP 服务）

增强特性：
  - BM25 + 向量检索混合排序
  - 相似度分数过滤
  - 结果去重
  - LRU 查询缓存
  - 医学同义词扩展

注册到 Agent-R1 工具链后，可在 agent_trainer.yaml 中配置：
    tool.tools: ['med_search']
"""

import json
import os
from collections import OrderedDict
from typing import Dict, List, Any, Optional

import numpy as np


# ──────────────────────── 医学同义词 / 口语映射 ────────────────────────────────

MED_SYNONYMS = {
    "高血压": ["高血压病", "血压高", "血压升高"],
    "糖尿病": ["血糖高", "血糖升高", "消渴症"],
    "冠心病": ["冠状动脉粥样硬化性心脏病", "冠状动脉硬化"],
    "心梗":   ["心肌梗死", "心肌梗塞", "急性心肌梗死"],
    "脑梗":   ["脑梗死", "脑梗塞", "缺血性脑卒中"],
    "中风":   ["脑卒中", "脑血管意外"],
    "肝炎":   ["病毒性肝炎", "肝脏炎症"],
    "胃炎":   ["胃黏膜炎症", "慢性胃炎", "急性胃炎"],
    "肺炎":   ["肺部感染", "肺部炎症"],
    "感冒":   ["上呼吸道感染", "伤风", "普通感冒"],
    "发烧":   ["发热", "体温升高", "高热"],
    "头疼":   ["头痛", "偏头痛", "头部疼痛"],
    "拉肚子": ["腹泻", "泻肚", "水样便"],
    "便秘":   ["排便困难", "大便干燥"],
    "失眠":   ["睡眠障碍", "入睡困难", "不寐"],
    "抑郁":   ["抑郁症", "忧郁症"],
    "过敏":   ["变态反应", "超敏反应"],
    "哮喘":   ["支气管哮喘"],
    "痛风":   ["高尿酸血症"],
    "贫血":   ["血红蛋白低"],
    "甲亢":   ["甲状腺功能亢进", "甲状腺功能亢进症"],
    "甲减":   ["甲状腺功能减退", "甲状腺功能减退症"],
    "肾结石": ["泌尿系结石", "尿路结石"],
    "胆结石": ["胆囊结石", "胆石症"],
    "痔疮":   ["痔", "内痔", "外痔"],
    "湿疹":   ["湿疹性皮炎"],
    "骨折":   ["骨裂", "断骨"],
    "腰椎间盘突出": ["腰突", "椎间盘突出", "腰椎突出"],
    "颈椎病": ["颈椎退行性变", "颈椎综合征"],
}

_SYNONYM_REVERSE: Dict[str, str] = {}
for _key, _syns in MED_SYNONYMS.items():
    _SYNONYM_REVERSE[_key] = _key
    for _syn in _syns:
        _SYNONYM_REVERSE[_syn] = _key


# ──────────────────────── 轻量 BM25 实现 ───────────────────────────────────────

class BM25Index:
    """基于倒排索引的 BM25 评分器，支持中文 jieba 分词。"""

    def __init__(self, tokenized_corpus: List[List[str]],
                 k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(tokenized_corpus)
        self.avgdl = (
            sum(len(doc) for doc in tokenized_corpus) / self.corpus_size
            if self.corpus_size else 0
        )
        self.doc_lens = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float32)

        self.df: Dict[str, int] = {}
        self.inverted: Dict[str, Dict[int, int]] = {}

        for doc_id, tokens in enumerate(tokenized_corpus):
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for term, freq in tf.items():
                if term not in self.inverted:
                    self.inverted[term] = {}
                    self.df[term] = 0
                self.inverted[term][doc_id] = freq
                self.df[term] += 1

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for token in query_tokens:
            if token not in self.inverted:
                continue
            df = self.df[token]
            idf = float(np.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0))
            for doc_id, tf in self.inverted[token].items():
                dl = self.doc_lens[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * numerator / denominator
        return scores


# ──────────────────────── LRU 缓存 ────────────────────────────────────────────

class LRUCache:
    """线程不安全的简易 LRU 缓存，用于避免同一会话内重复检索。"""

    def __init__(self, capacity: int = 128):
        self._cache: OrderedDict = OrderedDict()
        self._capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ──────────────────────── 搜索工具主类 ─────────────────────────────────────────

class MedSearchTool:
    """
    医学知识库搜索工具，接口兼容 Agent-R1 的 BaseTool。

    增强能力：
      1. 向量 (FAISS) + 关键词 (BM25) 混合检索
      2. 过采样 → 混合评分 → 重排
      3. 分数阈值过滤低相关结果
      4. 标题 Jaccard 去重
      5. 医学同义词/口语自动扩展
      6. LRU 查询缓存
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
        import faiss
        from sentence_transformers import SentenceTransformer

        kb_dir = os.path.dirname(os.path.abspath(__file__))

        if config_path is None:
            config_path = os.path.join(kb_dir, "config.yaml")
        cfg = self._load_config(config_path)

        if index_dir is None:
            index_dir = os.path.join(kb_dir, cfg["index"]["dir"])
        if model_name is None:
            model_name = cfg["embedding"]["model_name"]

        search_cfg = cfg.get("search", {})
        self.score_threshold = search_cfg.get("score_threshold", 0.0)
        self.dedup_threshold = search_cfg.get("dedup_threshold", 0.85)
        self.dense_weight = search_cfg.get("dense_weight", 0.7)
        self.bm25_weight = search_cfg.get("bm25_weight", 0.3)
        self.enable_bm25 = search_cfg.get("enable_bm25", True)
        self.enable_synonym = search_cfg.get("enable_synonym", True)
        self.oversample_factor = search_cfg.get("oversample_factor", 3)
        self.enable_hyde = search_cfg.get("enable_hyde", True)
        self.hyde_weight = search_cfg.get("hyde_weight", 0.4)
        cache_size = search_cfg.get("cache_size", 128)

        # ── FAISS 索引 ──
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

        # ── 语料库 ──
        print(f"[MedSearchTool] 加载语料库: {corpus_path}")
        self.corpus: List[dict] = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.corpus.append(json.loads(line))

        # ── 嵌入模型 ──
        print(f"[MedSearchTool] 加载嵌入模型: {model_name} (device={device})")
        self.model = SentenceTransformer(model_name, device=device)
        self.query_instruction = cfg["embedding"]["query_instruction"]
        self.default_top_k = top_k

        # ── BM25 索引 ──
        self.bm25: Optional[BM25Index] = None
        if self.enable_bm25:
            self._build_bm25()

        # ── 缓存 ──
        self.cache = LRUCache(capacity=cache_size)

        print(f"[MedSearchTool] ✓ 初始化完成")
        print(f"  知识库: {len(self.corpus)} 条文档")
        print(f"  BM25 混合检索: {'✓' if self.bm25 else '✗'}")
        print(f"  混合权重: dense={self.dense_weight}, bm25={self.bm25_weight}")
        print(f"  分数阈值: {self.score_threshold}")
        print(f"  去重阈值: {self.dedup_threshold}")
        print(f"  同义词扩展: {'✓' if self.enable_synonym else '✗'}")
        print(f"  HyDE: {'✓' if self.enable_hyde else '✗'}"
              f"{f' (weight={self.hyde_weight})' if self.enable_hyde else ''}")
        print(f"  缓存容量: {cache_size}")

    # ─────────── 初始化辅助 ───────────

    def _build_bm25(self):
        try:
            import jieba
            print(f"[MedSearchTool] 构建 BM25 索引（jieba 分词中）...")
            tokenized = []
            for doc in self.corpus:
                text = doc.get("title", "") + " " + doc.get("contents", "")
                tokenized.append(list(jieba.cut_for_search(text)))
            self.bm25 = BM25Index(tokenized)
            print(f"[MedSearchTool] BM25 索引构建完成")
        except ImportError:
            print(f"[MedSearchTool] jieba 未安装，BM25 已禁用（pip install jieba）")
            self.bm25 = None

    @staticmethod
    def _load_config(config_path: str) -> dict:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ─────────── 查询增强 ───────────

    def _expand_query(self, query: str) -> str:
        if not self.enable_synonym:
            return query

        added = []
        for term, canonical in _SYNONYM_REVERSE.items():
            if term in query:
                for syn in MED_SYNONYMS.get(canonical, [])[:2]:
                    if syn != term and syn not in query:
                        added.append(syn)
                break

        return (query + " " + " ".join(added)) if added else query

    # ─────────── 结果后处理 ───────────

    def _dedup_results(self, results: List[dict]) -> List[dict]:
        if self.dedup_threshold >= 1.0:
            return results

        deduped: List[dict] = []
        seen_titles: List[str] = []
        for r in results:
            title = r.get("title", "")
            if any(self._jaccard(title, s) >= self.dedup_threshold for s in seen_titles):
                continue
            deduped.append(r)
            seen_titles.append(title)
        return deduped

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    # ─────────── 公开接口 ───────────

    def execute(self, args: Dict) -> Dict[str, Any]:
        """单条搜索（兼容 BaseTool.execute 接口）。支持 HyDE 补充向量。"""
        try:
            query = args.get("query", "").strip()
            top_k = args.get("top_k", self.default_top_k)
            hyde_text = args.get("hyde_text", "").strip() if self.enable_hyde else ""

            cache_key = f"{query}||{hyde_text[:60]}||{top_k}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return {"content": json.dumps({"results": cached}, ensure_ascii=False),
                        "success": True}

            if hyde_text:
                results = self._search_with_hyde(query, hyde_text, top_k=top_k)
            else:
                results = self._search(query, top_k=top_k)
            self.cache.put(cache_key, results)

            return {"content": json.dumps({"results": results}, ensure_ascii=False),
                    "success": True}
        except Exception as e:
            return {"content": str(e), "success": False}

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        """批量搜索（兼容 BaseTool.batch_execute 接口）。"""
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

    # ─────────── 核心检索 ───────────

    def _search(self, query: str, top_k: int) -> List[dict]:
        return self._batch_search([query], top_k=top_k)[0]

    def _search_with_hyde(self, query: str, hyde_text: str,
                          top_k: int) -> List[dict]:
        """
        HyDE 增强搜索：用原始 query 和 think 文本两个向量分别检索，
        对同一文档取加权最大分数后合并排序。
        """
        expanded = self._expand_query(query)
        fetch_k = min(top_k * self.oversample_factor, self.index.ntotal)

        hyde_text = hyde_text[:300]

        texts_to_encode = [
            self.query_instruction + expanded,
            self.query_instruction + hyde_text,
        ]
        all_emb = self.model.encode(
            texts_to_encode,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        all_scores, all_ids = self.index.search(all_emb, fetch_k)

        # Merge candidates from both query vectors
        raw_candidates: Dict[int, float] = {}
        q_scores, q_ids = all_scores[0], all_ids[0]
        h_scores, h_ids = all_scores[1], all_ids[1]

        for score, doc_id in zip(q_scores, q_ids):
            doc_id = int(doc_id)
            if 0 <= doc_id < len(self.corpus):
                raw_candidates[doc_id] = max(
                    raw_candidates.get(doc_id, -999.0),
                    float(score),
                )

        for score, doc_id in zip(h_scores, h_ids):
            doc_id = int(doc_id)
            if 0 <= doc_id < len(self.corpus):
                hyde_score = float(score) * self.hyde_weight
                orig = raw_candidates.get(doc_id, -999.0)
                raw_candidates[doc_id] = max(orig, hyde_score)

        # Normalize dense scores
        if not raw_candidates:
            return []
        all_raw = list(raw_candidates.values())
        d_min, d_max = min(all_raw), max(all_raw)
        d_range = d_max - d_min if d_max > d_min else 1.0

        # BM25 scoring
        bm25_scores = None
        if self.bm25 is not None:
            try:
                import jieba
                tokens = list(jieba.cut_for_search(expanded))
                bm25_scores = self.bm25.get_scores(tokens)
            except ImportError:
                pass

        b_max = 1.0
        if bm25_scores is not None and bm25_scores.max() > 0:
            b_max = float(bm25_scores.max())

        scored: List[dict] = []
        for doc_id, raw_dense in raw_candidates.items():
            dense_norm = (raw_dense - d_min) / d_range
            bm25_norm = 0.0
            if bm25_scores is not None and doc_id < len(bm25_scores):
                bm25_norm = float(bm25_scores[doc_id]) / b_max

            fused = self.dense_weight * dense_norm + self.bm25_weight * bm25_norm

            if raw_dense < self.score_threshold and bm25_norm < 0.1:
                continue

            doc = self.corpus[doc_id]
            scored.append({
                "title": doc.get("title", ""),
                "content": doc.get("contents", ""),
                "score": round(fused, 4),
                "dense_score": round(raw_dense, 4),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        scored = self._dedup_results(scored)
        return scored[:top_k]

    def _batch_search(self, queries: List[str], top_k: int) -> List[List[dict]]:
        expanded = [self._expand_query(q) for q in queries]

        # Dense retrieval: oversample then rerank
        fetch_k = min(top_k * self.oversample_factor, self.index.ntotal)
        prefixed = [self.query_instruction + q for q in expanded]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        dense_scores_all, dense_ids_all = self.index.search(embeddings, fetch_k)

        # BM25 scoring (optional)
        bm25_scores_all: List[Optional[np.ndarray]] = [None] * len(queries)
        if self.bm25 is not None:
            try:
                import jieba
                for i, q in enumerate(expanded):
                    tokens = list(jieba.cut_for_search(q))
                    bm25_scores_all[i] = self.bm25.get_scores(tokens)
            except ImportError:
                pass

        batch_results: List[List[dict]] = []
        for i in range(len(queries)):
            d_scores = dense_scores_all[i]
            d_ids = dense_ids_all[i]
            bm25_scores = bm25_scores_all[i]

            # Collect all candidate doc_ids
            candidates: Dict[int, Dict[str, float]] = {}

            # --- Dense candidates ---
            d_min = float(d_scores.min())
            d_max = float(d_scores.max())
            d_range = d_max - d_min if d_max > d_min else 1.0

            for score, doc_id in zip(d_scores, d_ids):
                doc_id = int(doc_id)
                if doc_id < 0 or doc_id >= len(self.corpus):
                    continue
                norm = (float(score) - d_min) / d_range
                candidates[doc_id] = {
                    "dense_norm": norm,
                    "bm25_norm": 0.0,
                    "raw_dense": float(score),
                }

            # --- BM25 candidates (also inject top-k BM25 not in dense set) ---
            if bm25_scores is not None:
                b_max = float(bm25_scores.max()) if bm25_scores.max() > 0 else 1.0
                bm25_top_ids = np.argsort(bm25_scores)[-fetch_k:][::-1]
                for doc_id in bm25_top_ids:
                    doc_id = int(doc_id)
                    if doc_id not in candidates and doc_id < len(self.corpus):
                        candidates[doc_id] = {
                            "dense_norm": 0.0,
                            "bm25_norm": 0.0,
                            "raw_dense": 0.0,
                        }
                for doc_id in candidates:
                    if doc_id < len(bm25_scores):
                        candidates[doc_id]["bm25_norm"] = float(bm25_scores[doc_id]) / b_max

            # --- Fused scoring ---
            scored: List[dict] = []
            for doc_id, s in candidates.items():
                fused = self.dense_weight * s["dense_norm"] + self.bm25_weight * s["bm25_norm"]

                if s["raw_dense"] < self.score_threshold and s["bm25_norm"] < 0.1:
                    continue

                doc = self.corpus[doc_id]
                scored.append({
                    "title": doc.get("title", ""),
                    "content": doc.get("contents", ""),
                    "score": round(fused, 4),
                    "dense_score": round(s["raw_dense"], 4),
                })

            scored.sort(key=lambda x: x["score"], reverse=True)
            scored = self._dedup_results(scored)
            batch_results.append(scored[:top_k])

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
        return MedSearchTool(
            index_dir=index_dir,
            model_name=model_name,
            device=device,
            top_k=top_k,
            config_path=config_path,
        )


if __name__ == "__main__":
    tool = MedSearchTool(device="cpu")

    queries = ["糖尿病的症状有哪些", "高血压如何治疗", "感冒发烧怎么办"]
    for q in queries:
        result = tool.execute({"query": q})
        print(f"\n查询: {q}")
        data = json.loads(result["content"])
        for j, r in enumerate(data["results"][:3], 1):
            print(f"  [{j}] (score={r.get('score', '?')}) {r['title']}")
            print(f"      {r['content'][:100]}...")
    print(f"\n缓存命中率: {tool.cache.hit_rate:.1%}")
