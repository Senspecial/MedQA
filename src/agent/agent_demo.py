#!/usr/bin/env python3
"""
Agent 交互式演示脚本

这个脚本展示了如何使用医疗 Agent 系统，包括：
1. 初始化模型和 RAG 流水线
2. 创建医疗 Agent
3. 注册自定义工具
4. 与 Agent 进行交互式对话

使用方法：
    python src/agent/agent_demo.py
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from typing import Dict, Any, Optional
import json

from src.models.qwen_model import Qwen2Model
from src.rag.rag_pipeline import RAGPipeline
from src.agent.medical_agent import MedicalAgent
from src.agent.tools.tool_base import ToolBase
from src.knowledge_base.embedding_manager import EmbeddingManager


# ==================== 自定义工具示例 ====================

class DrugQueryTool(ToolBase):
    """药物查询工具示例"""
    
    def __init__(self):
        super().__init__(
            name="药物查询",
            description="查询药物的基本信息、用法用量和注意事项",
            parameters={
                "drug_name": {
                    "type": "string",
                    "description": "药物名称",
                    "required": True
                }
            }
        )
        
        # 简单的药物知识库（实际应该连接真实数据库）
        self.drug_database = {
            "阿司匹林": {
                "通用名": "乙酰水杨酸",
                "类别": "解热镇痛抗炎药",
                "用途": "用于解热、镇痛，也用于预防血栓",
                "用法用量": "成人常用量为每次0.3-0.6g，每日3-4次",
                "注意事项": "胃溃疡患者慎用，孕妇及哺乳期妇女慎用"
            },
            "青霉素": {
                "通用名": "青霉素G",
                "类别": "β-内酰胺类抗生素",
                "用途": "用于敏感菌引起的感染",
                "用法用量": "成人常用量为每次80万-200万单位，肌注或静滴",
                "注意事项": "使用前必须做皮试，过敏者禁用"
            },
            "布洛芬": {
                "通用名": "异丁苯丙酸",
                "类别": "非甾体抗炎药",
                "用途": "用于解热、镇痛和抗炎",
                "用法用量": "成人常用量为每次0.2-0.4g，每日2-3次",
                "注意事项": "胃肠道反应较轻，但仍需饭后服用"
            }
        }
    
    def _run(self, drug_name: str) -> str:
        """执行药物查询"""
        if drug_name in self.drug_database:
            drug_info = self.drug_database[drug_name]
            result = f"药物名称：{drug_name}\n"
            for key, value in drug_info.items():
                result += f"{key}：{value}\n"
            return result
        
        return f"抱歉，数据库中暂未收录{drug_name}的信息。建议咨询专业药师或查阅药品说明书。"


class SymptomCheckerTool(ToolBase):
    """症状检查工具示例"""
    
    def __init__(self):
        super().__init__(
            name="症状检查",
            description="根据症状提供可能的疾病方向（仅供参考，不能替代医生诊断）",
            parameters={
                "symptoms": {
                    "type": "string",
                    "description": "症状描述，多个症状用逗号分隔",
                    "required": True
                }
            }
        )
        
        # 简单的症状-疾病映射（实际应该使用专业医学知识图谱）
        self.symptom_disease_map = {
            "发热": ["感冒", "流感", "肺炎", "COVID-19"],
            "咳嗽": ["感冒", "支气管炎", "肺炎", "哮喘"],
            "头痛": ["感冒", "偏头痛", "紧张性头痛", "高血压"],
            "胸痛": ["心绞痛", "肋间神经痛", "胸膜炎"],
            "腹痛": ["胃炎", "阑尾炎", "肠炎", "消化不良"],
            "乏力": ["贫血", "甲减", "慢性疲劳综合征"]
        }
    
    def _run(self, symptoms: str) -> str:
        """执行症状检查"""
        symptom_list = [s.strip() for s in symptoms.split("，") if s.strip()]
        
        if not symptom_list:
            symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]
        
        if not symptom_list:
            return "请提供具体的症状描述。"
        
        possible_diseases = set()
        matched_symptoms = []
        
        for symptom in symptom_list:
            if symptom in self.symptom_disease_map:
                matched_symptoms.append(symptom)
                possible_diseases.update(self.symptom_disease_map[symptom])
        
        if not matched_symptoms:
            return f"未能识别您提供的症状：{symptoms}。建议详细描述症状或直接就医。"
        
        result = f"根据您提供的症状：{', '.join(matched_symptoms)}\n\n"
        result += "可能相关的疾病包括（仅供参考）：\n"
        for i, disease in enumerate(sorted(possible_diseases), 1):
            result += f"{i}. {disease}\n"
        
        result += "\n⚠️ 重要提示：\n"
        result += "- 这仅是基于症状的初步分析，不能替代专业医生的诊断\n"
        result += "- 如果症状持续或加重，请及时就医\n"
        result += "- 对于急性严重症状（如剧烈胸痛、呼吸困难等），请立即就医或拨打急救电话\n"
        
        return result


# ==================== Agent 演示类 ====================

class AgentDemo:
    """Agent 交互式演示"""
    
    def __init__(self):
        self.model = None
        self.rag_pipeline = None
        self.agent = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 配置
        self.config = {
            "model_path": "save/Qwen2_5-1_5B-medqa-merged",
            "embedding_model": "moka-ai/m3e-base",
            "index_dir": "data/indexes/agent_demo",
            "max_iterations": 5,
            "temperature": 0.7,
            "verbose": True
        }
    
    def initialize_model(self):
        """初始化语言模型"""
        print("=" * 60)
        print("正在初始化语言模型...")
        print(f"模型路径: {self.config['model_path']}")
        print(f"设备: {self.device}")
        
        self.model = Qwen2Model(
            model_path=self.config['model_path'],
            device=self.device
        )
        
        print("✓ 语言模型初始化完成")
    
    def initialize_rag(self, use_rag: bool = True):
        """初始化 RAG 流水线"""
        if not use_rag:
            print("跳过 RAG 初始化")
            return
        
        print("\n正在初始化 RAG 流水线...")
        print(f"嵌入模型: {self.config['embedding_model']}")
        
        try:
            # 初始化嵌入管理器
            embedding_manager = EmbeddingManager(
                embedding_model_name=self.config['embedding_model']
            )
            
            # 初始化 RAG 流水线
            self.rag_pipeline = RAGPipeline(
                retriever_config={
                    "name": "knn_retriever",
                    "score_threshold": 0.3,
                    "embedding_model_name": self.config['embedding_model'],
                    "index_type": "Flat"
                },
                embedding_manager=embedding_manager,
                use_rerank=False,
                verbose=self.config['verbose']
            )
            
            # 尝试加载已有索引
            index_path = Path(self.config['index_dir'])
            if index_path.exists():
                print(f"加载索引: {index_path}")
                self.rag_pipeline.retriever.load(str(index_path))
                print("✓ RAG 流水线初始化完成（已加载索引）")
            else:
                print("✓ RAG 流水线初始化完成（未找到索引，可以手动构建）")
        
        except Exception as e:
            print(f"⚠️ RAG 初始化失败: {e}")
            print("继续使用无 RAG 模式")
            self.rag_pipeline = None
    
    def initialize_agent(self):
        """初始化 Agent"""
        print("\n正在初始化医疗 Agent...")
        
        self.agent = MedicalAgent(
            model=self.model,
            rag_pipeline=self.rag_pipeline,
            name="智能医疗助手",
            max_iterations=self.config['max_iterations'],
            temperature=self.config['temperature'],
            verbose=self.config['verbose']
        )
        
        # 注册自定义工具
        print("注册自定义工具...")
        self.agent.add_tool(DrugQueryTool())
        self.agent.add_tool(SymptomCheckerTool())
        
        print("✓ Agent 初始化完成")
        print("\n可用工具：")
        print("  1. 药物查询 - 查询药物信息")
        print("  2. 症状检查 - 根据症状分析可能的疾病")
        if self.rag_pipeline:
            print("  3. 医疗知识检索 - 从知识库检索相关信息（如果有索引）")
    
    def run_query(self, query: str) -> Dict[str, Any]:
        """运行单次查询"""
        if not self.agent:
            return {"error": "Agent 未初始化"}
        
        print("\n" + "=" * 60)
        print(f"用户查询: {query}")
        print("-" * 60)
        
        result = self.agent.run(query)
        
        print("\nAgent 响应:")
        print(result.get("response", "无响应"))
        
        # 显示元数据
        metadata = result.get("metadata", {})
        print("\n" + "-" * 60)
        print(f"迭代次数: {metadata.get('iterations', 0)}")
        print(f"使用 RAG: {metadata.get('rag_used', False)}")
        print(f"工具调用次数: {len(metadata.get('tool_calls', []))}")
        
        if metadata.get('tool_calls'):
            print("\n工具调用详情:")
            for i, tool_call in enumerate(metadata['tool_calls'], 1):
                print(f"  {i}. {tool_call.get('name', '未知')}")
        
        if metadata.get('timing'):
            timing = metadata['timing']
            print(f"\n总耗时: {timing.get('total', 0):.2f}秒")
        
        return result
    
    def interactive_mode(self):
        """交互式对话模式"""
        print("\n" + "=" * 60)
        print("进入交互式对话模式")
        print("=" * 60)
        print("\n使用提示:")
        print("  - 直接输入问题进行提问")
        print("  - 输入 'exit' 或 'quit' 退出")
        print("  - 输入 'reset' 重置对话历史")
        print("  - 输入 'help' 查看帮助")
        print("\n示例问题:")
        print("  - 高血压患者应该注意什么？")
        print("  - 查询阿司匹林的信息")
        print("  - 我有发热和咳嗽的症状")
        print()
        
        while True:
            try:
                user_input = input("\n您: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() in ['exit', 'quit', '退出']:
                    print("\n感谢使用！再见！")
                    break
                
                if user_input.lower() in ['reset', '重置']:
                    self.agent.reset()
                    print("\n对话历史已重置")
                    continue
                
                if user_input.lower() in ['help', '帮助']:
                    self.show_help()
                    continue
                
                # 运行查询
                self.run_query(user_input)
            
            except KeyboardInterrupt:
                print("\n\n检测到中断信号，退出...")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                if self.config['verbose']:
                    import traceback
                    traceback.print_exc()
    
    def show_help(self):
        """显示帮助信息"""
        print("\n" + "=" * 60)
        print("帮助信息")
        print("=" * 60)
        print("\n命令:")
        print("  exit/quit  - 退出程序")
        print("  reset      - 重置对话历史")
        print("  help       - 显示此帮助信息")
        print("\n可用工具:")
        print("  1. 药物查询")
        print("     示例: 查询阿司匹林的信息")
        print("     示例: 布洛芬怎么用")
        print("\n  2. 症状检查")
        print("     示例: 我有发热和咳嗽的症状")
        print("     示例: 头痛和乏力是什么原因")
        print("\n  3. 一般医疗咨询")
        print("     示例: 高血压患者应该注意什么？")
        print("     示例: 如何预防糖尿病？")
        
        if self.rag_pipeline:
            print("\n  4. 知识库检索（如果已构建索引）")
            print("     Agent 会自动使用知识库中的信息回答问题")
    
    def run_demo(self, use_rag: bool = True):
        """运行完整演示"""
        try:
            # 初始化
            self.initialize_model()
            self.initialize_rag(use_rag)
            self.initialize_agent()
            
            # 运行几个示例查询
            print("\n" + "=" * 60)
            print("运行示例查询")
            print("=" * 60)
            
            example_queries = [
                "查询阿司匹林的信息",
                "我有发热和咳嗽的症状，可能是什么问题？",
                "高血压患者日常应该注意什么？"
            ]
            
            for query in example_queries:
                input("\n按 Enter 键继续下一个示例...")
                self.run_query(query)
            
            # 进入交互模式
            print("\n" + "=" * 60)
            user_choice = input("\n是否进入交互式对话模式？(y/n): ").strip().lower()
            if user_choice in ['y', 'yes', '是']:
                self.interactive_mode()
        
        except Exception as e:
            print(f"\n❌ 演示过程中发生错误: {e}")
            if self.config['verbose']:
                import traceback
                traceback.print_exc()


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("医疗 Agent 交互式演示")
    print("=" * 60)
    
    # 检查模型是否存在
    model_path = Path("save/Qwen2_5-1_5B-medqa-merged")
    if not model_path.exists():
        print(f"\n❌ 错误: 未找到模型文件")
        print(f"请确保模型已保存在: {model_path}")
        print("\n可以通过以下方式获取模型:")
        print("  1. 训练一个新模型")
        print("  2. 从已有位置复制模型")
        return
    
    # 选择模式
    print("\n请选择运行模式:")
    print("  1. 完整演示（包括 RAG）")
    print("  2. 简单演示（不使用 RAG）")
    print("  3. 直接进入交互模式（包括 RAG）")
    print("  4. 直接进入交互模式（不使用 RAG）")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    demo = AgentDemo()
    
    if choice == "1":
        demo.run_demo(use_rag=True)
    elif choice == "2":
        demo.run_demo(use_rag=False)
    elif choice == "3":
        demo.initialize_model()
        demo.initialize_rag(use_rag=True)
        demo.initialize_agent()
        demo.interactive_mode()
    elif choice == "4":
        demo.initialize_model()
        demo.initialize_rag(use_rag=False)
        demo.initialize_agent()
        demo.interactive_mode()
    else:
        print("无效选择，使用默认模式（完整演示）")
        demo.run_demo(use_rag=True)


if __name__ == "__main__":
    main()

