"""  
RAG服务路由  
提供知识检索和增强生成接口  
"""  

from typing import Dict, Any, List, Optional  
from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks  
from fastapi.responses import JSONResponse, StreamingResponse  
import time  
import json  
import asyncio  
from pydantic import BaseModel, Field
from src.knowledge_base.document_loader import DocumentLoader
import os
from pathlib import Path as PathLib  # ✅ 重命名避免与 FastAPI 的 Path 冲突
import logging

logger = logging.getLogger(__name__)

from src.api.schemas.rag import (
    RetrieveRequest, RetrieveResponse,   
    RagQuestionRequest, RagQuestionResponse,  
    KnowledgeBaseRequest, KnowledgeBaseResponse,  
    DocumentUploadRequest  
)  
from src.api.services.rag_service import get_rag_service, RAGService

router = APIRouter()  

@router.post("/retrieve", response_model=RetrieveResponse)  
async def retrieve_documents(  
    request: RetrieveRequest,  
    rag_service: RAGService = Depends(get_rag_service)  
):  
    """  
    检索相关文档  
    
    Args:  
        request: 检索请求  
        rag_service: RAG服务  
    
    Returns:  
        检索结果  
    """  
    start_time = time.time()  
    
    try:  
        # 执行检索  
        documents = rag_service.retrieve(  
            kb_name=request.kb_name,  
            query=request.query,  
            top_k=request.top_k  
        )  
        
        # 处理结果
        
        process_time = time.time() - start_time
        
        return RetrieveResponse(
            query=request.query,
            documents=documents,
            kb_name=request.kb_name,
            process_time=process_time,
            total_results=len(documents)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    


@router.post("/ask", response_model=RagQuestionResponse)
async def ask_rag_question(
    request: RagQuestionRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    基于知识库回答问题
    
    Args:
        request: RAG问题请求
        rag_service: RAG服务
    
    Returns:
        回答结果
    """
    start_time = time.time()
    
    try:
        # 生成回答
        response = rag_service.generate_response(
            kb_name=request.kb_name,
            query=request.question,
            model_name=request.model_name,
            top_k=request.top_k
        )
        
        # 处理结果
        process_time = time.time() - start_time
        
        return RagQuestionResponse(
            question=request.question,
            answer=response.get("answer", ""),
            contexts=response.get("contexts", []),
            sources=response.get("sources", []),
            kb_name=request.kb_name,
            model=request.model_name or "default",
            process_time=process_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/kb", response_model=List[KnowledgeBaseResponse])
async def list_knowledge_bases(
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    获取可用知识库列表
    
    Args:
        rag_service: RAG服务
    
    Returns:
        知识库列表
    """
    try:
        kbs = rag_service.get_available_knowledge_bases()
        return [KnowledgeBaseResponse(**kb) for kb in kbs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/kb", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(
    request: KnowledgeBaseRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    创建新知识库
    
    Args:
        request: 知识库创建请求
        rag_service: RAG服务
    
    Returns:
        创建的知识库信息
    """
    try:
        success = rag_service.create_knowledge_base(
            kb_name=request.name,
            description=request.description
        )
        
        if not success:
            raise HTTPException(status_code=400, detail=f"创建知识库失败: {request.name}")
        
        # 获取创建的知识库信息
        kbs = rag_service.get_available_knowledge_bases()
        for kb in kbs:
            if kb.get("name") == request.name:
                return KnowledgeBaseResponse(**kb)
        
        # 如果没找到，返回基本信息
        return KnowledgeBaseResponse(
            name=request.name,
            description=request.description,
            document_count=0,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/kb/{kb_name}", response_model=Dict[str, Any])
async def delete_knowledge_base(
    kb_name: str = Path(..., description="知识库名称"),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    删除知识库
    
    Args:
        kb_name: 知识库名称
        rag_service: RAG服务
    
    Returns:
        操作结果
    """
    try:
        success = rag_service.delete_knowledge_base(kb_name)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"删除知识库失败: {kb_name}")
        
        return {"success": True, "message": f"知识库已删除: {kb_name}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
    

@router.post("/kb/{kb_name}/documents", response_model=Dict[str, Any])
async def add_documents(
    request: DocumentUploadRequest,
    kb_name: str = Path(..., description="知识库名称"),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    向知识库添加文档
    
    Args:
        request: 文档上传请求
        kb_name: 知识库名称
        rag_service: RAG服务
    
    Returns:
        操作结果
    """
    try:
        success = rag_service.add_documents(kb_name, request.documents)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"添加文档失败: {kb_name}")
        
        return {
            "success": True, 
            "message": f"已添加 {len(request.documents)} 个文档到知识库: {kb_name}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DocumentPathRequest(BaseModel):
    """文档路径上传请求"""
    file_paths: List[str] = Field(..., description="文档文件路径列表（支持单个文件或目录）")
    recursive: bool = Field(False, description="如果是目录，是否递归加载子目录")

@router.post("/kb/{kb_name}/documents/from-path", response_model=Dict[str, Any])
async def add_documents_from_path(
    request: DocumentPathRequest,
    kb_name: str = Path(..., description="知识库名称"),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    通过文件路径向知识库添加文档
    
    支持的文件格式：PDF, TXT, DOCX, MD, CSV, XLSX, HTML, JSON
    支持单个文件或目录路径
    
    Args:
        request: 文档路径请求
        kb_name: 知识库名称
        rag_service: RAG服务
    
    Returns:
        操作结果
    """
    try:
        loader = DocumentLoader()
        all_documents = []
        
        # 处理每个路径
        for path_str in request.file_paths:
            path = PathLib(path_str)
            
            if not path.exists():
                raise HTTPException(
                    status_code=404, 
                    detail=f"路径不存在: {path_str}"
                )
            
            try:
                if path.is_file():
                    # 单个文件
                    langchain_docs = loader.load_document(str(path))
                    for doc in langchain_docs:
                        doc_dict = {
                            "id": f"{path.name}_{hash(doc.page_content)}",
                            "title": path.name,
                            "content": doc.page_content,
                            "metadata": doc.metadata or {},
                            "source": str(path)
                        }
                        all_documents.append(doc_dict)
                        
                elif path.is_dir():
                    # 目录
                    langchain_docs = loader.load_from_directory(
                        str(path), 
                        recursive=request.recursive
                    )
                    for doc in langchain_docs:
                        source = doc.metadata.get("source", "unknown")
                        doc_dict = {
                            "id": f"{PathLib(source).name}_{hash(doc.page_content)}",
                            "title": PathLib(source).name,
                            "content": doc.page_content,
                            "metadata": doc.metadata or {},
                            "source": source
                        }
                        all_documents.append(doc_dict)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"无效的路径类型: {path_str}"
                    )
                    
            except Exception as e:
                logger.error(f"加载文档 {path_str} 失败: {e}")
                continue
        
        if not all_documents:
            raise HTTPException(
                status_code=400, 
                detail="没有成功加载任何文档"
            )
        
        # 添加到知识库
        success = rag_service.add_documents(kb_name, all_documents)
        
        if not success:
            raise HTTPException(
                status_code=400, 
                detail=f"添加文档失败: {kb_name}"
            )
        
        return {
            "success": True,
            "message": f"已成功从 {len(request.file_paths)} 个路径加载 {len(all_documents)} 个文档片段到知识库: {kb_name}",
            "paths_processed": len(request.file_paths),
            "documents_added": len(all_documents)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"从路径加载文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载文档失败: {str(e)}")