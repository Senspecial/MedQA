"""
kb - 本地医学知识库模块

主要组件：
- build_kb.py      构建 FAISS 向量索引
- search_server.py HTTP 搜索服务（兼容 WikiSearchTool）
- kb_tool.py       离线搜索工具类（供 Agent-R1 直接调用）
- config.yaml      知识库配置
"""
