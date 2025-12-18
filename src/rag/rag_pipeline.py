# src/rag/rag_pipeline.py

import os
from typing import Dict, List, Optional, Union, Any, Tuple
from src.knowledge_base.retrieval.knn_retriever import KNNRetriever
from src.knowledge_base.retrieval.similarity_retriever import SimilarityRetriever
from src.knowledge_base.retrieval.bm25_retriever import BM25Retriever
from src.knowledge_base.retrieval.l2_retriever import L2Retriever
from src.knowledge_base.embedding_manager import EmbeddingManager
from .query_processor import QueryProcessor
from .context_builder import ContextBuilder
from .response_generator import ResponseGenerator
from src.models.base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class RAGPipeline:
    """
    RAG流水线，整合查询处理、文档检索、上下文构建和响应生成
    """
    
    def __init__(
        self,
        retriever_type: str = "knn",  # "knn", "similarity", "bm25", "l2", "hybrid"
        embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embedding_dimension: int = 384,
        index_path: Optional[str] = None,
        top_k: int = 5,
        model: Optional[BaseModel] = None,
        cache_dir: Optional[str] = None,
        reranker_model_name: Optional[str] = None,
        use_reranker: bool = False,
        hybrid_weight: float = 0.5,  # 混合检索时的权重(dense:sparse)
        chunk_size: int = 500,  # 上下文构建时的块大小
        chunk_overlap: int = 100,  # 上下文块重叠大小
        response_template: Optional[str] = None,  # 响应模板
        max_new_tokens: int = 1024,  # 生成的最大token数量
        temperature: float = 0.7,  # 生成温度
        top_p: float = 0.9,  # 生成top_p值
    ):
        """
        初始化RAG流水线
        
        Args:
            retriever_type: 检索器类型
            embedding_model_name: 嵌入模型名称
            embedding_dimension: 嵌入维度
            index_path: 索引路径
            top_k: 检索的文档数量
            model: 用于生成响应的模型
            cache_dir: 缓存目录
            reranker_model_name: 重排序模型名称
            use_reranker: 是否使用重排序
            hybrid_weight: 混合检索时的权重
            chunk_size: 上下文构建时的块大小
            chunk_overlap: 上下文块重叠大小
            response_template: 响应模板
            max_new_tokens: 生成的最大token数量
            temperature: 生成温度
            top_p: 生成top_p值
        """
        self.retriever_type = retriever_type
        self.top_k = top_k
        self.model = model
        self.use_reranker = use_reranker
        self.hybrid_weight = hybrid_weight
        
        # 初始化查询处理器
        self.query_processor = QueryProcessor()
        
        # 初始化嵌入管理器 (对于某些检索器需要)
        if retriever_type in ["knn", "similarity", "l2", "hybrid"]:
            self.embedding_manager = EmbeddingManager(
                embedding_model_name=embedding_model_name,  # ✅ 修复参数名
                cache_dir=cache_dir
            )
        else:
            self.embedding_manager = None
        
        # 根据类型初始化检索器
        if retriever_type == "knn":
            self.retriever = KNNRetriever(
                embedding_manager=self.embedding_manager
                # ✅ 移除 index_path 参数
            )
            # 如果索引路径存在，尝试加载
            if index_path and os.path.exists(index_path):
                try:
                    self.retriever.load(index_path)
                except Exception as e:
                    logger.warning(f"加载索引失败，将创建新索引: {e}")
                    
        elif retriever_type == "similarity":
            self.retriever = SimilarityRetriever(
                embedding_manager=self.embedding_manager
                # ✅ 移除 index_path 参数
            )
            if index_path and os.path.exists(index_path):
                try:
                    self.retriever.load(index_path)
                except Exception as e:
                    logger.warning(f"加载索引失败，将创建新索引: {e}")
                    
        elif retriever_type == "bm25":
            self.retriever = BM25Retriever(
                # ✅ 移除 index_path 参数
            )
            if index_path and os.path.exists(index_path):
                try:
                    self.retriever.load(index_path)
                except Exception as e:
                    logger.warning(f"加载索引失败，将创建新索引: {e}")
                    
        elif retriever_type == "l2":
            self.retriever = L2Retriever(
                embedding_manager=self.embedding_manager
                # ✅ 移除 index_path 参数
            )
            if index_path and os.path.exists(index_path):
                try:
                    self.retriever.load(index_path)
                except Exception as e:
                    logger.warning(f"加载索引失败，将创建新索引: {e}")
                    
        elif retriever_type == "hybrid":
            # 混合检索需要同时初始化向量检索和BM25
            dense_index_path = index_path.replace(".bin", "_dense") if index_path else None
            sparse_index_path = index_path.replace(".bin", "_sparse") if index_path else None
            
            self.dense_retriever = KNNRetriever(
                embedding_manager=self.embedding_manager
                # ✅ 移除 index_path 参数
            )
            # 如果密集索引存在，尝试加载
            if dense_index_path and os.path.exists(dense_index_path):
                try:
                    self.dense_retriever.load(dense_index_path)
                except Exception as e:
                    logger.warning(f"加载密集索引失败: {e}")
            
            self.sparse_retriever = BM25Retriever(
                # ✅ 移除 index_path 参数
            )
            # 如果稀疏索引存在，尝试加载
            if sparse_index_path and os.path.exists(sparse_index_path):
                try:
                    self.sparse_retriever.load(sparse_index_path)
                except Exception as e:
                    logger.warning(f"加载稀疏索引失败: {e}")
                    
            self.retriever = None  # 混合模式下不使用单一检索器
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")
        
        # 初始化上下文构建器
        self.context_builder = ContextBuilder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            reranker_model_name=reranker_model_name if use_reranker else None
        )
        
        # 如果提供了模型，初始化响应生成器
        if model:
            self.response_generator = ResponseGenerator(
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                template=response_template
            )
        else:
            self.response_generator = None
    
    def set_model(self, model: BaseModel) -> None:
        """
        设置用于生成响应的模型
        
        Args:
            model: 模型实例
        """
        self.model = model
        self.response_generator = ResponseGenerator(
            model=model,
            max_new_tokens=self.response_generator.max_new_tokens if self.response_generator else 1024,
            temperature=self.response_generator.temperature if self.response_generator else 0.7,
            top_p=self.response_generator.top_p if self.response_generator else 0.9,
            template=self.response_generator.template if self.response_generator else None
        )
    
    def _query_hybrid(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        使用混合检索方式进行查询
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            
        Returns:
            检索到的文档列表
        """
        k = top_k or self.top_k
        
        # 获取密集(向量)检索结果
        dense_docs = self.dense_retriever.retrieve(query_text, k=k*2)
        
        # 获取稀疏(BM25)检索结果
        sparse_docs = self.sparse_retriever.retrieve(query_text, k=k*2)
        
        # 合并结果
        combined_docs = {}
        
        # 处理密集检索结果
        for i, doc in enumerate(dense_docs):
            # 使用文档ID或文本哈希作为唯一标识
            doc_id = doc.get("id") or doc.get("metadata", {}).get("doc_id") or str(hash(doc.get("text", "")))
            
            if doc_id not in combined_docs:
                combined_docs[doc_id] = doc.copy()
                combined_docs[doc_id]["combined_score"] = 0
            
            # 添加密集检索得分
            dense_score = doc.get("score", 0.0)  # 使用实际得分
            combined_docs[doc_id]["dense_score"] = dense_score
            combined_docs[doc_id]["combined_score"] += self.hybrid_weight * dense_score
        
        # 处理稀疏检索结果
        for i, doc in enumerate(sparse_docs):
            doc_id = doc.get("id") or doc.get("metadata", {}).get("doc_id") or str(hash(doc.get("text", "")))
            
            if doc_id not in combined_docs:
                combined_docs[doc_id] = doc.copy()
                combined_docs[doc_id]["combined_score"] = 0
                combined_docs[doc_id]["dense_score"] = 0
            
            # 添加稀疏检索得分
            sparse_score = doc.get("score", 0.0)
            combined_docs[doc_id]["sparse_score"] = sparse_score
            combined_docs[doc_id]["combined_score"] += (1 - self.hybrid_weight) * sparse_score
        
        # 根据组合得分排序
        sorted_docs = sorted(
            combined_docs.values(), 
            key=lambda x: x.get("combined_score", 0), 
            reverse=True
        )
        
        # 返回top-k结果
        return sorted_docs[:k]
    
    def query(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        根据查询文本检索相关文档
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            
        Returns:
            检索到的文档列表
        """
        # 处理查询文本
        processed_query = self.query_processor.process_query(query_text)
        
        # 确定检索数量
        k = top_k or self.top_k
        
        # 如果是混合检索
        if self.retriever_type == "hybrid":
            return self._query_hybrid(processed_query, k)
        
        # 使用单一检索器
        return self.retriever.retrieve(processed_query, k)
    
    def build_context(self, query_text: str, top_k: Optional[int] = None, 
                     use_reranker: Optional[bool] = None) -> str:
        """
        构建增强上下文
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            use_reranker: 是否使用重排序
            
        Returns:
            构建的上下文字符串
        """
        # 检索相关文档
        retrieved_docs = self.query(query_text, top_k)
        
        if not retrieved_docs:
            logger.warning("没有检索到相关文档")
            return ""
        
        # 确定是否使用重排序
        _use_reranker = use_reranker if use_reranker is not None else self.use_reranker
        
        # 构建上下文
        context = self.context_builder.build_context(
            query=query_text,
            documents=retrieved_docs,
            use_reranker=_use_reranker
        )
        
        return context
    
    def build_rag_prompt(self, query_text: str, top_k: Optional[int] = None,
                       use_reranker: Optional[bool] = None) -> Dict[str, Any]:
        """
        构建RAG提示
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            use_reranker: 是否使用重排序
            
        Returns:
            包含查询、上下文和格式化提示的字典
        """
        # 获取上下文
        context = self.build_context(query_text, top_k, use_reranker)
        
        # 检索相关文档
        documents = self.query(query_text, top_k)
        
        # 构建提示
        prompt = {
            "query": query_text,
            "context": context,
            "documents": documents,
            "formatted_prompt": self.context_builder.format_prompt(query_text, context)
        }
        
        return prompt
    
    def generate_response(self, query_text: str, top_k: Optional[int] = None,
                        use_reranker: Optional[bool] = None) -> Dict[str, Any]:
        """
        生成RAG增强的响应
        
        Args:
            query_text: 查询文本
            top_k: 检索文档数量
            use_reranker: 是否使用重排序
            
        Returns:
            包含响应和元数据的字典
        """
        if not self.model or not self.response_generator:
            raise ValueError("未设置模型，无法生成响应")
        
        # 构建RAG提示
        rag_prompt = self.build_rag_prompt(query_text, top_k, use_reranker)
        
        # 生成响应
        response = self.response_generator.generate(rag_prompt)
        
        return response
    
    def update_retriever_index(self, documents: List[Dict[str, Any]], save_path: Optional[str] = None) -> None:
        """
        更新检索器索引
        
        Args:
            documents: 文档列表（字典格式）
            save_path: 保存路径
        """
        from langchain_core.documents import Document
        
        # 将字典格式转换为 LangChain Document 格式
        langchain_docs = []
        for doc_dict in documents:
            doc = Document(
                page_content=doc_dict.get("content", ""),
                metadata={
                    "doc_id": doc_dict.get("id", ""),
                    "title": doc_dict.get("title", ""),
                    "source": doc_dict.get("source", ""),
                    **(doc_dict.get("metadata", {}) or {})
                }
            )
            langchain_docs.append(doc)
        
        if self.retriever_type == "hybrid":
            # 更新密集检索器
            self.dense_retriever.add_documents(langchain_docs)
            
            # 更新稀疏检索器
            self.sparse_retriever.add_documents(langchain_docs)
            
            # 保存索引
            if save_path:
                dense_save_path = save_path.replace(".bin", "_dense") if save_path else None
                sparse_save_path = save_path.replace(".bin", "_sparse") if save_path else None
                if dense_save_path:
                    self.dense_retriever.save(dense_save_path)
                if sparse_save_path:
                    self.sparse_retriever.save(sparse_save_path)
        else:
            # 更新单一检索器
            self.retriever.add_documents(langchain_docs)
            
            # 保存索引
            if save_path:
                self.retriever.save(save_path)
        
        logger.info(f"检索器索引已更新，共 {len(langchain_docs)} 个文档")