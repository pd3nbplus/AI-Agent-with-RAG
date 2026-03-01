【Agent开发】第六阶段：RAG 深度优化实战 —— 父子索引与上下文窗口优化 -- pd的AI Agent开发笔记
---

[toc]


前置环境：当前环境是基于WSL2 + Ubuntu 24.04 + Docker Desktop构建的云原生开发平台，所有服务（MySQL、Redis、Qwen）均以独立容器形式运行并通过Docker Compose统一编排。如何配置请参考我的博客 [WSL2 + Ubuntu 24.04 + Docker Desktop 配置双内核环境](https://blog.csdn.net/weixin_52185313/article/details/158416250?spm=1011.2415.3001.5331) 并且补充了milvus相关的配置，如何配置请参考我的博客 [【Agent开发】第三阶段：RAG 实战 —— 赋予 Agent “外脑”](https://blog.csdn.net/weixin_52185313/article/details/158506104?spm=1011.2415.3001.5331)。 并且引入了ES检索，并且配置了ES服务，ES部分的配置请查看我的博客 [【Agent开发】第五阶段：RAG 深度优化实战 —— 从“可用”到“卓越”](https://blog.csdn.net/weixin_52185313/article/details/158506104?spm=1011.2415.3001.5331)。

> 核心目标：打破“小 Chunk 检索准但上下文缺失，大 Chunk 上下文全但检索噪声大”的僵局。
> 演进路线：Advanced RAG → Context-Aware RAG。
> 关键概念：Small-to-Big Retrieval (小查大返)。

# 第三部分：父子索引与上下文窗口优化—— 解决“检索准”与“上下文全”的矛盾 (Parent-Child Indexing)

## 1. 痛点分析：为什么需要父子索引？

❌ 当前困境
在之前的架构中，我们通常选择一个固定的 Chunk Size（例如 500 字）：
+ 如果 Chunk 太小 (200 字)：
  + ✅ 优点：向量嵌入非常精准，能匹配到具体的细节。
  + ❌ 缺点：丢失上下文。LLM 拿到片段后不知道“他”是谁，“这个公司”指哪家，导致回答断章取义。
+ 如果 Chunk 太大 (2000 字)：
  + ✅ 优点：上下文完整，LLM 能理解全局。
  + ❌ 缺点：向量被大量无关信息稀释，检索精度下降，容易匹配到错误文档。

**✅ 解决方案：父子索引 (Parent-Child Indexing)**
**核心思想：“用小块检索，用大块生成”。**
1. 子块 (Child Chunks)：小粒度分块（如 200 字），用于生成向量并建立索引。负责“被找到”。
2. 父块 (Parent Chunks)：大粒度分块（如 1000 字）或原始文档。负责“被阅读”。
3. 映射关系：每个子块记录其所属的父块 ID。
4. 检索流程：
   + 用户查询 → 匹配 子块。
   + 命中子块 → 通过 ID 找回对应的 父块。
   + 将 父块 发送给 LLM 生成答案。

## 架构设计：数据模型与流程

我们需要在现有的 Milvus 结构中增加 parent_id 字段，并调整入库和检索逻辑。

📂 新增/修改文件结构

```text
src/rag/
    ├── chunkers.py              # 👈 更新：增加父子分块策略
    ├── ingestion.py             # 👈 更新：建立父子映射并入库
    ├── strategies/
    │   └── retrievers/
    │       └── vector_text.py   # 👈 更新：支持“查子返父”逻辑
    └── pipeline.py              # (无需大改，接口保持一致)
```

**🗄️ Milvus Schema 变更**
需要在 Collection 中增加一个标量字段 parent_id (VarChar)，用于反向查找。

## 3. 实战编码

### 🛠️ 第一步：实现父子分块策略 (src/rag/chunkers.py)

我们将创建一个特殊的 Chunker，它先生成大块（父），再把大块切分成小块（子），并建立关联。

```py
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
    # 已有逻辑

class FixedChunker(BaseChunker):
    # 已有逻辑

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
```

### 🔄 第二步：更新入库逻辑 (src/rag/ingestion.py)

修改 DataIngestion 类，使其支持动态加载不同的 Chunker，并保持 process_file 逻辑不变（因为接口统一了）。

```py
    def process_file(self, file_path: str, category: str = "general"):
        """处理单个文件：加载 -> 分块 -> 增强 -> 入库"""
        # 1. 加载文档
        docs = self.load_document(file_path) 
        if not docs:
            return

        all_chunks = []
        
        # 2. 分块 (统一调用 split_documents，无需关心内部是简单还是父子)
        # 输入：List[Document], 输出：List[Document]
        for doc in docs:
            splits = self.text_splitter.split_documents([doc])
            all_chunks.extend(splits)
        
        logger.info(f"✂️ 分块完成，共生成 {len(all_chunks)} 个 chunks")

        # 3. 入库
        success_count = 0
        for i, chunk in enumerate(all_chunks):
            text = chunk.page_content
            
            # 跳过过短的块
            if len(text.strip()) < 5:
                continue

            # 元数据增强(可选：为了速度，生产环境可异步或批量处理)
            enhanced_meta = self.enhance_metadata(text, chunk.metadata.get("source", ""))
            summary_str = enhanced_meta.get("summary", "")
            questions_str = enhanced_meta.get("questions", "")

            # 合并元数据：保留分块器产生的 metadata (如 parent_id, parent_text)
            final_metadata = {
                **chunk.metadata,  # 👈 关键：保留 parent_id 和 parent_text
                "source": os.path.basename(file_path),
                "page": chunk.metadata.get("page", 0),
                "category": category,
                "summary": summary_str,
                "questions": questions_str
            }
            
            # 生成唯一 ID
            doc_id = f"{os.path.basename(file_path)}_{i}_{uuid.uuid4().hex[:6]}"
            
            # A. 存入 Milvus (向量化的是 chunk.page_content 即小子块)
            try:
                self.milvus.insert_data(
                    id=doc_id,
                    text=text,
                    metadata=final_metadata # metadata 里现在包含了 parent_text
                )
                success_count += 1
            except Exception as e:
                logger.error(f"❌ Milvus 插入失败：{e}")

            # B. 存入 Elasticsearch (关键词库) - 双管齐下
            if self.es.is_available():
                if questions_str:
                    self.es.indexing_question(
                        doc_id=doc_id,
                        questions=questions_str,
                        summary=summary_str,
                        text=text,
                        metadata=final_metadata
                    )
                
                if summary_str:
                    self.es.indexing_summary(
                        doc_id=doc_id,
                        summary=summary_str,
                        text=text,
                        metadata=final_metadata
                    )

        logger.info(f"✅ 文件 {file_path} 处理完毕，成功入库 {success_count}/{len(all_chunks)} 条记录")
```

### 🔍 第三步：更新检索策略 (src/rag/strategies/retrievers)

+ vector_text.py
+ vector_rewritten.py

确保检索时能利用 parent_text 实现“查子返父”。这两个文件的改法类似
```python
# src/rag/strategies/retrievers/vector_rewritten.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from src.core.milvus_client import get_milvus_client
from src.rag.rewriter import rewriter_instance # 复用之前的重写器
from src.core.config import settings
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class VectorRewrittenRetriever(BaseRetrievalStrategy):
    """
    插件 2: 变体向量检索
    策略：先让 LLM 重写查询 (Query Rewriting)，再用重写后的句子进行向量搜索。
    """
    def __init__(self):
        self.milvus = get_milvus_client()
        self.rewriter = rewriter_instance
        # 从配置读取是否启用父子上下文
        self.use_parent_context = settings.rag_offline.chunk_strategy == "parent_child"
        logger.info(f"🔌 [Plugin] 加载变体向量检索插件 (Rewritten) 检索模式: ({'查子反父' if self.use_parent_context else 'normal'})")

    def search(self, query: str, top_k: int, filter_expr: Optional[str] = None, **kwargs) -> List[SearchResult]:
        # 1. 生成变体查询
        try:
            rewritten_query = self.rewriter.rewrite(query)
            if rewritten_query == query:
                logger.debug("⚠️ [Vector-Rewritten] 重写后无变化，跳过此路以避免重复")
                return []
            logger.info(f"🔄 [Vector-Rewritten] 变体查询：{rewritten_query}")
        except Exception as e:
            logger.error(f"❌ [Vector-Rewritten] 重写失败：{e}")
            return []
        
        hits = self.milvus.search(
            query=query,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=["text", "metadata"]
        )

        results = []
        for hit in hits:
            meta = hit['metadata'] or {}
            original_text = hit['text']
            
            final_text = original_text
            # source_tag = "vector_rewritten"
            source_tag = os.path.splitext(os.path.basename(__file__))[0]
            # 👇 核心逻辑：查子返父
            if self.use_parent_context and meta.get("parent_text"):
                final_text = meta["parent_text"]
                source_tag = source_tag + "_parent"
                # 可选：在 metadata 中记录原始子块文本，方便调试
                meta['_matched_child_text'] = original_text

        results.append(SearchResult(
                text=final_text, # 返回大块文本给 LLM
                score=hit['score'], # 分数基于小子块匹配 (精准)
                metadata=meta,
                source_field=source_tag
            ))

        return results
```

+ es_summaries.py
+ es_questions.py

这两个文件的改法也很类似
```python
class ESSummariesRetriever(BaseRetrievalStrategy):
    """
    插件 3: ES 关键词检索 (Questions 字段)
    """
    def __init__(self):
        self.es = es_client_instance
        self.index_name = settings.db.es_index_summaries
        # 从配置读取是否启用父子上下文
        self.use_parent_context = settings.rag_offline.chunk_strategy == "parent_child"
        if self.es.is_available():
            logger.info(f"🔌 [Plugin] ES 检索插件{self.index_name}已就绪 检索模式: ({'查子反父' if self.use_parent_context else 'normal'})")
        else:
            logger.warning("🔌 [Plugin] ES 不可用，此插件将自动跳过")

    def search(self, query: str, top_k: int, filter_expr: Optional[str] = None, **kwargs) -> List[SearchResult]:
        if not self.es.is_available():
            return []
            
        # 调用封装好的搜索方法
        # 注意：ES 原生不支持复杂的 JSON 过滤表达式 (如 Milvus 语法)，这里暂不实现 filter_expr
        # 如果需要，可以在 ES 查询中添加 term 过滤
        hits = self.es.search_summaries(query, top_k=top_k)
        
        results = []
        for hit in hits:
            meta = hit['metadata'] or {}
            original_text = hit['text']
            
            final_text = original_text
            # source_tag = "es_summaries"
            source_tag = os.path.splitext(os.path.basename(__file__))[0]
            # 👇 核心逻辑：查子返父
            if self.use_parent_context and meta.get("parent_text"):
                final_text = meta["parent_text"]
                source_tag = source_tag + "_parent"
                # 可选：在 metadata 中记录原始子块文本，方便调试
                meta['_matched_child_text'] = original_text

            results.append(SearchResult(
                text=final_text,
                score=hit['score'],
                metadata=meta,
                source_field=source_tag
            ))
        return results
```
