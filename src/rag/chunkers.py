# src/rag/chunkers.py
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from abc import ABC, abstractmethod
import uuid
import logging
import jieba

logger = logging.getLogger(__name__)

# ==========================================
# 1. 基础接口定义 (Abstract Base Classes)
# ==========================================

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

class BaseChildSplitter(ABC):
    """
    子块分块器基类 (专门用于 Parent-Child 模式)
    负责将单个父块文本切分为多个子块 Document
    """
    @abstractmethod
    def split_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Args:
            text: 父块的文本内容
            metadata: 父块的元数据 (将传递给子块)
        Returns:
            子块 Document 列表
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
    父子分块策略 (Parent-Child Indexing)
    
    逻辑：
    1. 将原文切分为大块 (Parent)。
    2. 利用注入的 child_splitter 将每个大块切分为小块 (Child)。
    3. 返回所有小块 (Child)，但在 metadata 中注入 parent_id 和 parent_text。
    
    依赖注入：
    - child_splitter: 由外部工厂传入的具体 BaseChildSplitter 实例
    """
    
    def __init__(
        self, 
        parent_size: int = 1000, 
        parent_overlap: int = 100,
        child_splitter: Optional[BaseChildSplitter] = None,
        parent_separators: List[str] = None
    ):
            
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        
        # 初始化父分块器
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            length_function=len,
            separators=parent_separators
        )
        
        # 依赖注入：子分块器 (必须由外部传入，此处不依赖 settings)
        if child_splitter is None:
            # 如果外部没传，默认给一个递归子分块器 (兜底策略)
            logger.warning("⚠️ ParentChildChunker 未传入 child_splitter，使用默认 RecursiveChildSplitter")
            self.child_splitter = RecursiveChildSplitter(
                chunk_size=200, 
                chunk_overlap=0, 
                separators=["\n\n", "\n", "。", "！", "？", " ", ""]
            )
        else:
            self.child_splitter = child_splitter
            
        logger.info(f"✂️ 初始化父子分块器：ParentSize={parent_size}, ChildStrategy={self.child_splitter.__class__.__name__}")

    def split_documents(self, docs: List[Document]) -> List[Document]:
        all_child_docs = []
        total_parents = 0
        
        for doc in docs:
            # 1. 切分父块
            parent_docs = self.parent_splitter.split_documents([doc])
            total_parents += len(parent_docs)
            
            for p_doc in parent_docs:
                parent_id = str(uuid.uuid4())
                parent_text = p_doc.page_content
                
                # 2. 使用注入的子分块器切分父块
                # 传入父块的元数据，子分块器会保留并增强它
                child_docs = self.child_splitter.split_text(parent_text, p_doc.metadata)
                
                for c_doc in child_docs:
                    # 3. 注入父子关系元数据
                    new_metadata = {
                        **c_doc.metadata,
                        "parent_id": parent_id,
                        "parent_text": parent_text, # 关键：存入大块文本供检索后返回
                        "is_child": True,
                        "chunk_type": f"child_{self.child_splitter.__class__.__name__}"
                    }
                    
                    child_doc = Document(
                        page_content=c_doc.page_content, # 小子块内容 (用于向量化)
                        metadata=new_metadata
                    )
                    all_child_docs.append(child_doc)
        
        logger.info(f"✅ 父子分块完成：输入 {len(docs)} 文档 -> 生成 {len(all_child_docs)} 个子块 (关联 {total_parents} 个父块)")
        return all_child_docs

# ==========================================
# 2. 子块分块器实现 (Concrete Child Splitters)
# ==========================================

class RecursiveChildSplitter(BaseChildSplitter):
    """
    基于字符递归切分的子块策略
    适用于：通用文档，追求均匀切分
    """
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str]):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )
        logger.debug(f"   [ChildSplitter] 初始化：Recursive (Size={chunk_size}, Overlap={chunk_overlap})")

    def split_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        temp_doc = Document(page_content=text, metadata=metadata)
        return self.splitter.split_documents([temp_doc])

class SentenceChildSplitter(BaseChildSplitter):
    """
    基于 Jieba 的句子切分策略 (Sentence-as-Child)
    
    优势：
    1. 利用 jieba 强大的中文分句能力，准确识别 。！？等标点。
    2. 自动处理中英文混排、数字、缩写等复杂情况。
    3. 比正则更稳健，不易产生空切片或错误截断。
    """
    def __init__(self, min_sentence_len: int = 10):
        self.min_sentence_len = min_sentence_len
        # 预加载 jieba 字典 (可选，加速首次运行)
        # jieba.initialize() 
        logger.debug(f"   [ChildSplitter] 初始化：Jieba Sentence (MinLen={min_sentence_len})")

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        使用 jieba 进行分句
        原理：jieba.lcut 会把标点符号也作为单独的词切分出来，我们据此重组句子。
        """
        # 使用 jieba 精确模式切分
        words = jieba.lcut(text)
        
        sentences = []
        current_sent = ""
        
        # 定义句末标点集合 (中文 + 英文)
        end_punctuations = {'。', '！', '？', '!', '?', '…', '…', '.'}
        # 换行符也视为句子结束
        # newlines = {'\n', '\r'}
        
        for word in words:
            current_sent += word
            
            # 判断是否结束一个句子
            if word.strip() in end_punctuations:
                s = current_sent.strip()
                if len(s) >= self.min_sentence_len:
                    sentences.append(s)
                current_sent = ""
        
        # 处理最后一段没有标点的情况
        if current_sent.strip():
            s = current_sent.strip()
            if len(s) >= self.min_sentence_len:
                sentences.append(s)
                
        return sentences

    def split_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        sentences = self._split_into_sentences(text)
        docs = []
        
        if not sentences:
            # 如果分句失败（极少见），返回整个文本作为一个块，防止数据丢失
            logger.warning(f"⚠️ Jieba 分句结果为空，回退为单块。文本前50字：{text[:50]}...")
            return [Document(page_content=text, metadata=metadata)]
            
        for i, sent in enumerate(sentences):
            doc = Document(
                page_content=sent,
                metadata={
                    **metadata,
                    "sentence_index": i,
                    "total_sentences_in_parent": len(sentences),
                    "chunk_method": "jieba_sentence"
                }
            )
            docs.append(doc)
            
        return docs

# ==========================================
# 3. 子分块器工厂 (Internal Factory for Decoupling)
# ==========================================
# 注意：这是一个纯逻辑工厂，不包含 settings 依赖，由外部 factories.py 调用

class ChildSplitterFactory:
    """
    子分块器创建工厂
    根据传入的策略字符串和参数创建具体的子分块器实例
    """
    
    @staticmethod
    def create(strategy: str, **kwargs) -> BaseChildSplitter:
        strategy = strategy.lower()
        
        if strategy == "sentence" or strategy == "sentence_window" or strategy == "jieba":
            return SentenceChildSplitter(
                min_sentence_len=kwargs.get('min_sentence_len', 10)
            )
        elif strategy == "recursive" or strategy == "fixed" or strategy == "char":
            return RecursiveChildSplitter(
                chunk_size=kwargs.get('chunk_size', 200),
                chunk_overlap=kwargs.get('chunk_overlap', 0),
                separators=kwargs.get('separators', ["\n\n", "\n", "。", "！", "？", " "])
            )
        else:
            logger.warning(f"⚠️ 未知子分块策略 '{strategy}'，降级使用 recursive")
            return RecursiveChildSplitter(
                chunk_size=kwargs.get('chunk_size', 200),
                chunk_overlap=kwargs.get('chunk_overlap', 0),
                separators=kwargs.get('separators', ["\n\n", "\n", "。", "！", "？", " "])
            )