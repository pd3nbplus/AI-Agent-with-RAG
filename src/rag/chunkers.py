# src/rag/chunkers.py
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from typing import List, Dict, Any
from langchain_core.documents import Document
from abc import ABC, abstractmethod
import uuid
import logging

logger = logging.getLogger(__name__)

class BaseChunker(ABC):
    """分块器基类"""
    def split_documents(self, docs: List[Document]) -> List[Document]:
        raise NotImplementedError

    @abstractmethod
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        输入：原始 Document 列表
        输出：分块后的 Document 列表 (每个 Document 的 metadata 可能包含增强信息)
        """
        pass

class RecursiveChunker(BaseChunker):
    """递归字符分块策略 (当前默认)"""
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str]):
        logger.info(f"✂️ 初始化递归分块器：Size={chunk_size}, Overlap={chunk_overlap}")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)

class FixedChunker(BaseChunker):
    """固定长度分块策略 (简单粗暴)"""
    def __init__(self, chunk_size: int, chunk_overlap: int):
        logger.info(f"✂️ 初始化固定分块器：Size={chunk_size}, Overlap={chunk_overlap}")
        self.splitter = CharacterTextSplitter(
            separator="\n", # 简单按行切
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)

# 可以在这里继续添加 SentenceChunker, SemanticChunker 等

class ParentChildChunker(BaseChunker):
    """
    父子分块策略
    生成两组数据：
    1. Child Chunks: 小尺寸，用于向量化检索
    2. Parent Chunks: 大尺寸 (或原文)，用于提供给 LLM
    3. 返回所有小块 (Child)，但在 metadata 中注入 parent_id 和 parent_text
    """
    def __init__(self, parent_size: int = 500, child_size: int = 50, overlap: int = 50,child_overlap: int = 10, separators: List[str] = None):
        if separators is None:
            separators = ["\n\n", "\n", "。", "！", "？", " ", ""]
            
        logger.info(f"✂️ 初始化父子分块器：Parent={parent_size}, Child={child_size}, Overlap={overlap}")
        
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=separators
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=child_overlap
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        兼容标准接口：输入 Document 列表，返回 Document 列表 (子块)
        """
        all_child_docs = []
        
        for doc in docs:
            # 1. 切分父块
            parent_docs = self.parent_splitter.split_documents([doc])
            
            for p_doc in parent_docs:
                # 为每个父块生成唯一 ID
                parent_id = str(uuid.uuid4())
                parent_text = p_doc.page_content
                
                # 2. 将父块切分为子块
                child_docs = self.child_splitter.split_documents([p_doc])
                
                for c_doc in child_docs:
                    # 3. 注入父子关系元数据
                    # 复制原有 metadata，避免修改原始对象
                    new_metadata = {
                        **c_doc.metadata,
                        "parent_id": parent_id,
                        "parent_text": parent_text, # 关键：存入大块文本
                        "is_child": True,
                        "chunk_type": "child"
                    }
                    
                    # 创建新的 Document 对象
                    child_doc = Document(
                        page_content=c_doc.page_content, # 小子块内容 (用于向量化)
                        metadata=new_metadata
                    )
                    all_child_docs.append(child_doc)
        
        logger.info(f"✅ 父子分块完成：输入 {len(docs)} 文档 -> 生成 {len(all_child_docs)} 个子块 (关联 {len(parent_docs)} 个父块)")
        return all_child_docs