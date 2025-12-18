"""
向量数据库构建详细演示
展示完整的文档加载、向量化、索引构建过程
"""
import os
import sys

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.knowledge_base.document_loader import DocumentLoader
from src.knowledge_base.embedding_manager import EmbeddingManager
from src.knowledge_base.retrieval.knn_retriever import KNNRetriever
from langchain_core.documents import Document


def demo_step_by_step():
    """逐步演示向量数据库构建"""
    
    print("="*70)
    print("向量数据库构建详细演示")
    print("="*70)
    
    # ============== 阶段 1: 准备文档 ==============
    print("\n【阶段 1】准备文档数据")
    print("-"*70)
    
    # 方式1: 直接创建Document对象
    documents = [
        Document(
            page_content="高血压是一种常见的慢性疾病，需要长期管理。主要治疗方法包括药物治疗和生活方式改变。",
            metadata={"doc_id": "doc_001", "title": "高血压管理", "category": "慢性病"}
        ),
        Document(
            page_content="糖尿病患者应注意饮食控制，减少糖分摄入，增加膳食纤维。定期监测血糖水平很重要。",
            metadata={"doc_id": "doc_002", "title": "糖尿病护理", "category": "慢性病"}
        ),
        Document(
            page_content="感冒通常由病毒引起，症状包括发热、咳嗽、流涕。多休息多喝水有助于恢复。",
            metadata={"doc_id": "doc_003", "title": "感冒治疗", "category": "常见病"}
        )
    ]
    
    print(f"✓ 创建了 {len(documents)} 个文档对象")
    for i, doc in enumerate(documents):
        print(f"  文档 {i+1}: {doc.metadata['title']} | 内容长度: {len(doc.page_content)} 字符")
    
    # 方式2: 从文件加载 (演示)
    print("\n提示: 也可以使用 DocumentLoader 从文件加载:")
    print("  loader = DocumentLoader()")
    print("  docs = loader.load_document('path/to/file.pdf')")
    print("  支持: PDF, TXT, DOCX, MD, CSV, XLSX, HTML, JSON")
    
    # ============== 阶段 2: 初始化嵌入模型 ==============
    print("\n【阶段 2】初始化嵌入模型")
    print("-"*70)
    
    embedding_model = "moka-ai/m3e-base"  # 中文嵌入模型
    print(f"正在加载嵌入模型: {embedding_model}")
    print("提示: 首次运行会自动下载模型，后续会使用缓存")
    
    embedding_manager = EmbeddingManager(
        embedding_model_name=embedding_model,
        cache_dir="embedding_cache",  # 缓存目录
        use_cache=True  # 启用缓存
    )
    
    print(f"✓ 嵌入模型加载完成")
    print(f"  模型名称: {embedding_model}")
    print(f"  向量维度: {embedding_manager.get_embedding_dimension()}")
    print(f"  缓存目录: embedding_cache/")
    
    # ============== 阶段 3: 生成嵌入向量 ==============
    print("\n【阶段 3】生成文档嵌入向量")
    print("-"*70)
    
    print("正在将文档转换为向量...")
    embeddings_dict = embedding_manager.embed_documents(documents)
    
    print(f"✓ 生成了 {len(embeddings_dict)} 个嵌入向量")
    for doc_id, embedding in embeddings_dict.items():
        print(f"  {doc_id}: 向量维度 {len(embedding)}, 前5维: {embedding[:5]}")
    
    # ============== 阶段 4: 构建FAISS索引 ==============
    print("\n【阶段 4】构建 FAISS 向量索引")
    print("-"*70)
    
    print("正在初始化 KNN 检索器...")
    retriever = KNNRetriever(
        embedding_manager=embedding_manager,
        index_type="Flat"  # 使用精确搜索
    )
    
    print("正在添加文档到索引...")
    retriever.add_documents(documents)
    
    print(f"✓ 向量索引构建完成")
    print(f"  索引类型: FAISS Flat (精确搜索)")
    print(f"  文档数量: {len(retriever.documents)}")
    print(f"  向量维度: {retriever.dimension}")
    
    # ============== 阶段 5: 测试检索 ==============
    print("\n【阶段 5】测试向量检索")
    print("-"*70)
    
    test_queries = [
        "如何控制高血压？",
        "糖尿病饮食要注意什么？",
        "感冒了怎么办？"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        
        # 方法1: 使用search (返回Document和score)
        results = retriever.search(query, top_k=2)
        
        for i, (doc, score) in enumerate(results):
            title = doc.metadata.get('title', '未知')
            content = doc.page_content[:40]
            print(f"  [{i+1}] 相似度: {score:.4f} | {title}")
            print(f"      {content}...")
    
    # ============== 阶段 6: 保存索引 ==============
    print("\n【阶段 6】保存索引到磁盘")
    print("-"*70)
    
    save_dir = "data/indexes/demo_kb"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"正在保存索引到: {save_dir}")
    retriever.save(save_dir)
    
    print("✓ 索引保存完成")
    print(f"  索引文件: {save_dir}/faiss_index.bin")
    print(f"  文档数据: {save_dir}/documents.pkl")
    print(f"  配置文件: {save_dir}/config.json")
    print(f"  文档ID: {save_dir}/document_ids.json")
    
    # 获取文件大小
    if os.path.exists(f"{save_dir}/faiss_index.bin"):
        index_size = os.path.getsize(f"{save_dir}/faiss_index.bin") / 1024
        docs_size = os.path.getsize(f"{save_dir}/documents.pkl") / 1024
        print(f"  文件大小: faiss_index.bin={index_size:.1f}KB, documents.pkl={docs_size:.1f}KB")
    else:
        print("  注意: 索引文件未找到，可能保存失败")
    
    # ============== 阶段 7: 加载索引 ==============
    print("\n【阶段 7】从磁盘加载索引")
    print("-"*70)
    
    print("创建新的检索器并加载已保存的索引...")
    new_retriever = KNNRetriever(
        embedding_manager=embedding_manager,
        index_type="Flat"
    )
    new_retriever.load(save_dir)
    
    print(f"✓ 索引加载完成")
    print(f"  文档数量: {len(new_retriever.documents)}")
    print(f"  向量维度: {new_retriever.dimension}")
    
    # 测试加载的索引
    print("\n测试加载的索引:")
    query = "高血压治疗"
    results = new_retriever.search(query, top_k=1)
    doc, score = results[0]
    print(f"  查询: {query}")
    print(f"  结果: {doc.metadata['title']} (相似度: {score:.4f})")
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
    
    return retriever


def demo_from_file():
    """从文件构建向量数据库的演示"""
    
    print("\n\n")
    print("="*70)
    print("【补充】从文件构建向量数据库")
    print("="*70)
    
    # 示例：如果有JSON文件
    json_file = "dpo.json"  # 您的DPO数据集
    
    if os.path.exists(json_file):
        print(f"\n正在从文件加载: {json_file}")
        
        try:
            loader = DocumentLoader()
            documents = loader.load_document(json_file)
            
            print(f"✓ 加载了 {len(documents)} 个文档")
            
            # 显示前几个文档
            print("\n前3个文档示例:")
            for i, doc in enumerate(documents[:3]):
                content = doc.page_content[:60].replace('\n', ' ')
                print(f"  [{i+1}] 长度: {len(doc.page_content)} | {content}...")
            
            # 可以继续使用上面的流程构建索引
            print("\n提示: 使用上面的流程可以为这些文档构建索引")
        except Exception as e:
            print(f"✗ 加载失败: {e}")
    else:
        print(f"\n文件不存在: {json_file}")
        print("演示跳过")


if __name__ == "__main__":
    # 运行详细演示
    retriever = demo_step_by_step()
    
    # 从文件加载的演示
    demo_from_file()
    
    print("\n\n" + "="*70)
    print("💡 重要提示")
    print("="*70)
    print("\n索引文件说明:")
    print("  📁 data/indexes/demo_kb/")
    print("     ├── faiss_index.bin      # FAISS向量索引")
    print("     ├── documents.pkl        # 文档内容和元数据")
    print("     ├── document_ids.json    # 文档ID映射")
    print("     └── config.json          # 配置信息")
    print("\n  📁 embedding_cache/        # 嵌入向量缓存")
    print("     └── moka-ai_m3e-base_embedding_cache.pkl")
    print("\n下次运行:")
    print("  - ✅ 直接使用缓存，速度更快")
    print("  - ✅ 可以加载已保存的索引")
    print("  - ✅ 避免重复计算嵌入向量")
    print("\n运行其他演示:")
    print("  python -m src.rag.rag_demo              # RAG完整流程演示")
    print("="*70)

