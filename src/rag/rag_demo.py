"""
完整的交互式 RAG 系统
支持文档加载、向量索引构建、交互式问答
"""
import os
import sys
import time
from typing import List, Dict, Any

# 确保项目根目录在 python path 中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.rag.rag_pipeline import RAGPipeline
from src.models.qwen_model import Qwen2Model
from src.knowledge_base.document_loader import DocumentLoader
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger("rag_demo")

def load_documents_from_source(source_type: str = "builtin") -> List[Dict[str, Any]]:
    """
    从不同来源加载文档
    
    Args:
        source_type: 文档来源类型 ("builtin", "file", "directory")
    
    Returns:
        文档字典列表
    """
    if source_type == "builtin":
        # 内置测试文档
        return [
        {
            "id": "doc_001",
                "content": "感冒通常由病毒引起，症状包括鼻塞、流涕、咳嗽、喉咙痛等。休息和多喝水是主要的治疗建议。轻度感冒通常7-10天可自愈。",
                "metadata": {"source": "常见病指南", "title": "感冒概述", "category": "常见病"}
        },
        {
            "id": "doc_002",
                "content": "高血压患者应控制盐分摄入，定期监测血压。常用药物包括利尿剂、钙通道阻滞剂、ACEI类药物等。生活方式干预包括戒烟限酒、适度运动、减轻体重。",
                "metadata": {"source": "慢性病管理", "title": "高血压护理", "category": "慢性病"}
        },
        {
            "id": "doc_003",
                "content": "糖尿病饮食控制非常重要，应减少糖分和精制碳水化合物的摄入，增加膳食纤维。推荐低GI食物，如全谷物、豆类、蔬菜等。",
                "metadata": {"source": "饮食健康", "title": "糖尿病饮食", "category": "慢性病"}
            },
            {
                "id": "doc_004",
                "content": "心脏病的预防需要控制危险因素，包括高血压、高血脂、糖尿病、吸烟等。定期体检和心电图检查很重要。",
                "metadata": {"source": "心血管疾病", "title": "心脏病预防", "category": "慢性病"}
            },
            {
                "id": "doc_005",
                "content": "失眠的治疗可以从改善睡眠习惯开始，如固定作息时间、避免睡前使用电子设备、保持卧室安静舒适。必要时可考虑认知行为疗法或药物治疗。",
                "metadata": {"source": "睡眠健康", "title": "失眠治疗", "category": "常见病"}
            },
            {
                "id": "doc_006",
                "content": "新冠疫苗接种可以有效预防重症和死亡。常见副作用包括注射部位疼痛、发热、疲劳等，通常在1-2天内缓解。",
                "metadata": {"source": "疫苗指南", "title": "新冠疫苗", "category": "预防"}
            },
            {
                "id": "doc_007",
                "content": "骨质疏松症患者应增加钙和维生素D的摄入，进行适度的负重运动。高危人群应定期进行骨密度检测。",
                "metadata": {"source": "骨骼健康", "title": "骨质疏松", "category": "慢性病"}
        }
    ]
    
    elif source_type == "file":
        print("\n请输入文档文件路径 (支持: PDF, TXT, DOCX, MD, JSON 等):")
        file_path = input("文件路径: ").strip()
        
        if not os.path.exists(file_path):
            print(f"✗ 文件不存在: {file_path}")
            return []
        
        loader = DocumentLoader()
        try:
            langchain_docs = loader.load_document(file_path)
            documents = []
            for i, doc in enumerate(langchain_docs):
                documents.append({
                    "id": f"doc_{i:03d}",
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            return documents
        except Exception as e:
            print(f"✗ 加载文件失败: {e}")
            return []
    
    return []


def interactive_query_loop(rag: RAGPipeline, use_llm: bool = False):
    """
    交互式查询循环
    
    Args:
        rag: RAG流水线实例
        use_llm: 是否使用LLM生成回答
    """
    print("\n" + "="*70)
    print("🤖 交互式 RAG 问答系统")
    print("="*70)
    print("\n使用说明:")
    print("  - 输入问题进行查询")
    print("  - 输入 'exit' 或 'quit' 退出")
    print("  - 输入 'stats' 查看系统统计")
    print("  - 输入 'help' 查看帮助")
    print("="*70)
    
    query_count = 0
    
    while True:
        print("\n" + "-"*70)
        user_input = input("\n💬 您的问题: ").strip()
        
        if not user_input:
            continue
        
        # 处理特殊命令
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\n👋 感谢使用，再见！")
            break
        
        elif user_input.lower() == 'help':
            print("\n📖 帮助信息:")
            print("  exit/quit - 退出系统")
            print("  stats     - 显示统计信息")
            print("  help      - 显示此帮助")
            continue
        
        elif user_input.lower() == 'stats':
            print("\n📊 系统统计:")
            print(f"  查询次数: {query_count}")
            print(f"  文档数量: {len(rag.retriever.documents) if hasattr(rag.retriever, 'documents') else '未知'}")
            print(f"  索引类型: {rag.retriever_type}")
            print(f"  使用LLM: {'是' if use_llm else '否'}")
            continue
        
        # 执行查询
        query_count += 1
        print(f"\n🔍 正在检索相关文档...")
        
        try:
            start_time = time.time()
            
            # 检索文档
            retrieved_docs = rag.query(user_input, top_k=3)
            retrieval_time = time.time() - start_time
            
            if not retrieved_docs:
                print("⚠️  未找到相关文档")
                continue
            
            print(f"✓ 检索完成 ({retrieval_time:.3f}秒)")
            print(f"\n📄 找到 {len(retrieved_docs)} 个相关文档:\n")
            
            # 显示检索结果
            for i, doc in enumerate(retrieved_docs):
                score = doc.get('score', 0)
                content = doc.get('content', doc.get('text', ''))
                metadata = doc.get('metadata', {})
                title = metadata.get('title', '未命名文档')
                
                print(f"  [{i+1}] 📌 {title} (相关度: {score:.4f})")
                print(f"      {content[:100]}...")
                print()
            
            # 如果启用LLM，生成回答
            if use_llm:
                print("🤖 正在生成回答...")
                try:
                    gen_start = time.time()
                    response = rag.generate_response(user_input, top_k=3)
                    gen_time = time.time() - gen_start
                    
                    print("\n" + "="*70)
                    print("💡 AI 回答:")
                    print("-"*70)
                    
                    # 提取回答内容
                    if isinstance(response, dict):
                        answer = response.get('answer', response.get('response', str(response)))
                    else:
                        answer = str(response)
                    
                    print(answer)
                    print("-"*70)
                    print(f"⏱️  生成耗时: {gen_time:.2f}秒")
                    print("="*70)
                    
                except Exception as e:
                    print(f"✗ 生成回答失败: {e}")
            else:
                print("💡 提示: 启用LLM可以获得更详细的回答")
        
        except Exception as e:
            print(f"✗ 查询失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数：完整的RAG系统初始化和交互流程"""
    
    print("="*70)
    print("🚀 完整 RAG 系统 - 交互式问答")
    print("="*70)

    # ========== 配置参数 ==========
    EMBEDDING_MODEL = "moka-ai/m3e-base"
    LLM_MODEL_PATH = "save/Qwen2_5-1_5B-medqa-merged"
    INDEX_PATH = "data/indexes/rag_demo"
    
    # ========== 阶段 1: 文档准备 ==========
    print("\n【阶段 1/4】文档准备")
    print("-"*70)
    
    print("\n选择文档来源:")
    print("  1. 使用内置测试文档 (7个医疗知识文档)")
    print("  2. 从文件加载 (PDF, TXT, DOCX, JSON等)")
    
    choice = input("\n请选择 (1/2, 默认1): ").strip() or "1"
    
    if choice == "2":
        documents = load_documents_from_source("file")
        if not documents:
            print("使用内置文档作为备选")
            documents = load_documents_from_source("builtin")
    else:
        documents = load_documents_from_source("builtin")
    
    print(f"✓ 已准备 {len(documents)} 个文档")
    
    # ========== 阶段 2: 初始化RAG流水线 ==========
    print("\n【阶段 2/4】初始化 RAG 流水线")
    print("-"*70)
    print(f"📦 嵌入模型: {EMBEDDING_MODEL}")
    print(f"🔍 检索器类型: KNN (FAISS)")
    print(f"💾 索引路径: {INDEX_PATH}")
    
    try:
    rag = RAGPipeline(
        retriever_type="knn",
        embedding_model_name=EMBEDDING_MODEL,
        index_path=INDEX_PATH,
        top_k=3
    )
        print("✓ RAG 流水线初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return
    
    # ========== 阶段 3: 构建向量索引 ==========
    print("\n【阶段 3/4】构建向量索引")
    print("-"*70)
    
    os.makedirs(os.path.dirname(INDEX_PATH) if os.path.dirname(INDEX_PATH) else "data/indexes", exist_ok=True)
    
    # 检查是否已有索引
    if os.path.exists(INDEX_PATH) and os.path.exists(f"{INDEX_PATH}/faiss_index.bin"):
        use_cache = input("检测到已有索引，是否使用？(Y/n): ").strip().lower()
        if use_cache != 'n':
            try:
                print("正在加载已有索引...")
                # 注意：需要先添加文档才能正确加载
                rag.update_retriever_index(documents, save_path=INDEX_PATH)
                print("✓ 索引加载成功")
            except:
                print("加载失败，重新构建索引...")
                rag.update_retriever_index(documents, save_path=INDEX_PATH)
                print("✓ 索引构建完成")
        else:
            rag.update_retriever_index(documents, save_path=INDEX_PATH)
            print("✓ 索引构建完成")
    else:
        print("正在构建向量索引...")
        rag.update_retriever_index(documents, save_path=INDEX_PATH)
        print("✓ 索引构建完成")
    
    # ========== 阶段 4: LLM加载 (可选) ==========
    print("\n【阶段 4/4】LLM 加载 (可选)")
    print("-"*70)
    print("是否加载 LLM 进行智能回答生成？")
    print("  - 选择 'y': 加载医疗模型，生成详细回答 (需要GPU)")
    print("  - 选择 'n': 仅进行文档检索 (更快，无需GPU)")
    
    use_llm_choice = input("\n是否加载 LLM？(y/N): ").strip().lower()
    use_llm = False
    
    if use_llm_choice == 'y':
        print(f"\n正在加载模型: {LLM_MODEL_PATH}")
        print("⏳ 加载中，请稍候...")
        
        try:
        model = Qwen2Model(
            model_path=LLM_MODEL_PATH,
            device="cuda",
            load_in_4bit=True, 
            trust_remote_code=True
        )
        rag.set_model(model)
            print("✓ 模型加载成功")
            use_llm = True
    except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            print("将以仅检索模式运行")
            use_llm = False
    else:
        print("✓ 将以仅检索模式运行")
    
    # ========== 进入交互式查询循环 ==========
    interactive_query_loop(rag, use_llm=use_llm)

if __name__ == "__main__":
    main()
#python -m src.rag.rag_demo