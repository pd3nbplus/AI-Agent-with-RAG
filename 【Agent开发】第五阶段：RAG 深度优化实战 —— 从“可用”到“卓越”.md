【Agent开发】第五阶段：RAG 深度优化实战 —— 从“可用”到“卓越” -- pd的AI Agent开发笔记
---

[toc]


前置环境：当前环境是基于WSL2 + Ubuntu 24.04 + Docker Desktop构建的云原生开发平台，所有服务（MySQL、Redis、Qwen）均以独立容器形式运行并通过Docker Compose统一编排。如何配置请参考我的博客 [WSL2 + Ubuntu 24.04 + Docker Desktop 配置双内核环境](https://blog.csdn.net/weixin_52185313/article/details/158416250?spm=1011.2415.3001.5331) 并且补充了milvus相关的配置，如何配置请参考我的博客 [【Agent开发】第三阶段：RAG 实战 —— 赋予 Agent “外脑”](https://blog.csdn.net/weixin_52185313/article/details/158506104?spm=1011.2415.3001.5331)。

> 核心目标：激活沉睡在 metadata 中的摘要（Summary）和假设性问题（Hypothetical Questions），从“单一路径检索”升级为“混合过滤与匹配”。
> 痛点解决：
>    1. 语义鸿沟：用户问“怎么报销”，文档写“报销流程”，虽然向量能匹配，但如果有直接存储的“问题”匹配会更准。
>    2. 领域混淆：用户问“年假”，如果不限制范围，可能会搜到“产品手册”里提到的“假期促销”。
>    3. 摘要浪费：之前生成的 summary 只存在那，没在检索时发挥作用。


# 第一部分：元数据与摘要的深度利用 (Metadata & Summary Deep Dive)

## 0. 痛点分析与解决方案

**❌ 当前痛点**
1. “存而不用”：我们花费算力生成了 summary 和 questions，但检索时只用了 text 字段。
2. “语义鸿沟”：用户问“年假几天”，文档写“入职满一年...”，虽然向量能匹配，但如果用户问的是文档里生成的那个“假设性问题”，匹配度会更高。
3. “缺乏精确控制”：无法强制限制“只查 HR 文档”或“只查 2024 年后的政策”。
4. “关键词失效”：向量检索对专有名词（如“Qwen-3-4B”、“Error-503”）不敏感，容易匹配到语义相近但无关的内容。

## 1. 解决方案矩阵


| 痛点编号 | 痛点描述 | 解决方案 | 技术手段 | 预期效果 |
|----------|-----------|------------|-------------|-------------|
| 1 | “存而不用”：summary 和 questions 未被有效利用 | 激活 metadata 中的摘要和假设性问题 | 使用 Elasticsearch 存储并检索 summary 和 questions 字段 | 提高检索相关性，充分利用预生成内容 |
| 2 | 语义鸿沟：用户查询与文档表述存在差异 | 实施 Query Rewriting 机制 | 利用 LLM 重写用户查询，使其更贴近文档表述 | 缩小语义差距，提高匹配准确率 |
| 3 | 缺乏精确控制：无法限制检索范围 | 引入元数据过滤策略 | Milvus 支持标量字段过滤，实现精确检索控制 | 实现精准检索，减少无关结果 |
| 4 | 关键词失效：向量检索对专有名词不敏感 | 结合关键词检索补充向量检索 | Elasticsearch 提供关键词匹配能力 | 增强对专有名词的识别能力 |


**1. 激活 metadata 中的摘要和假设性问题**
- **目的**：解决“存而不用”的问题，使预先生成的摘要和假设性问题能够参与检索过程。
- **实现**：部署 Elasticsearch 并安装 IK 分词插件，创建专门的索引用于存储 `summary` 和 `questions` 字段。
- **优势**：通过关键词匹配增强检索的相关性和准确性。

**2. Query Rewriting 机制**
- **目的**：缓解语义鸿沟，让用户的问题更贴近文档的实际表述。
- **实现**：借助大型语言模型（LLM）对原始查询进行改写，生成更加规范化的查询语句后再进行向量检索。
- **优势**：显著提升跨领域或表达方式不同的查询匹配成功率。

**3. 元数据过滤**
- **目的**：赋予用户更强的检索控制权，能够在特定范围内查找信息。
- **实现**：利用 Milvus 支持的标量字段过滤功能，在检索时加入条件约束。
- **优势**：有效排除不相关领域的干扰信息，聚焦于目标内容。

**4. 关键词检索补充**
- **目的**：针对向量检索在处理专有名词方面的局限性提供补救措施。
- **实现**：整合 Elasticsearch 的强大文本搜索能力，特别是在处理精确词汇匹配方面。
- **优势**：确保即使面对高度专业化的术语也能获得良好的检索表现。


## 2. 架构设计：可插拔的检索策略

我们将重构 rag/pipeline.py，将其变为一个组装器，动态加载各种检索策略

📂 新增/修改文件结构
```text
src/rag/
├── strategies/                # 👈 新增：策略模式核心目录
│   ├── __init__.py
│   ├── base.py                # 基础策略接口
│   ├── metadata_filter.py     # 元数据过滤策略
│   ├── composer.py            # 检索器组装器
│   └── retrievers/            # 👈 新增：具体检索器插件目录
│       ├── __init__.py
│       ├── vector_text.py     # 插件 1: 向量检索 Text (主路)
│       ├── vector_rewritten.py# 插件 2: 变体向量检索 (Rewritten Query)
│       └── es_questions.py    # 插件 3: ES 关键词检索 (Questions) - *预留接口*
├── factories.py               # 更新：增加策略工厂
├── pipeline.py                # 更新：组装策略
└── ... (其他文件不变)
```

## 2.5 ES配置

### docker配置
修改docker-compose.yml，添加 ES 配置。
```yaml
# ================= Elasticsearch 搜索引擎 =================
  # 端口：9200 (API), 9300 (节点通信)
  # 注意：ES 非常吃内存，建议给 Docker 分配至少 4GB+ 内存
  
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: es-local
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false  # 开发环境关闭密码验证，方便调试
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g" # 限制 JVM 内存，防止撑爆
      - TZ=Asia/Shanghai
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - ./data/es:/usr/share/elasticsearch/data
    # 👇 新增：启动时自动安装 IK 插件
    command: >
      bash -c '
        echo "🚀 开始安装 IK 分词插件..." &&
        bin/elasticsearch-plugin install --batch https://release.infinilabs.com/analysis-ik/stable/elasticsearch-analysis-ik-8.11.0.zip &&
        echo "✅ IK 插件安装完成，启动 ES..." &&
        /usr/local/bin/docker-entrypoint.sh
      '
    networks:
      - ai-net
    restart: "no"
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5

  # ================= ElasticHD 图形化界面 =================
  # 访问地址：http://localhost:9800
  # 连接地址：http://elasticsearch:9200 (容器内部名称)
  
  elastichd:
    image: containerize/elastichd:latest
    container_name: elastichd-local
    ports:
      - "9800:9800"
    environment:
      - ELASTICSEARCH_URL=http://elasticsearch:9200 # 指向 ES 容器名
    depends_on:
      - elasticsearch
    networks:
      - ai-net
    restart: "no"
```

💡 启动命令：
```bash
docker-compose up -d elasticsearch elastichd
```

图形化管理：访问 http://localhost:9800 查看 ES 数据和索引状态。也可以不装elastichd，用vscode中的连接插件即可。

### python配置

python也需要安装ES依赖
```bash
# 保证跟elasticsearch的版本一致
pip install elasticsearch==8.11.0
```
注意这里依赖下面修改的config.py
🛠️ 新建 `src/core/es_client.py`

这是一个单例客户端，负责初始化连接、索引创建和数据同步辅助功能。
```python
# src/core/es_client.py
from elasticsearch import Elasticsearch, helpers
from src.core.config import settings
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class ESClient:
    def __init__(self):
        self.host = settings.db.es_host
        self.index_questions = settings.db.es_index_questions
        self.index_summaries = settings.db.es_index_summaries
        self.client: Optional[Elasticsearch] = None
        self._connect()

    def _connect(self):
        """初始化 ES 连接"""
        try:
            hosts = [self.host]
            
            # 尝试连接
            self.client = Elasticsearch(
                hosts=hosts,
                basic_auth=(settings.db.es_user, settings.db.es_password) if settings.db.es_user else None,
                request_timeout=30
            )
            
            # 测试连通性
            info = self.client.info()
            logger.info(f"✅ Elasticsearch 连接成功：{info['version']['number']}")
            
            # 初始化索引 (如果不存在)
            self._init_indices()
            
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch 连接失败或不可用：{e}")
            self.client = None

    def _init_indices(self):
        """创建必要的索引映射"""
        if not self.client:
            return
            
        indices = [
            {
                "name": self.index_questions,
                "mapping": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "questions": {"type": "text", "analyzer": "ik_max_word"}, # 中文分词
                        "text": {"type": "text"},
                        "metadata": {"type": "object"}
                    }
                }
            },
            {
                "name": self.index_summaries,
                "mapping": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "text": {"type": "text", "analyzer": "ik_max_word"},
                        "summaries": {"type": "text"},
                        "metadata": {"type": "object"}
                    }
                }
            }
        ]
        
        for idx in indices:
            if not self.client.indices.exists(index=idx["name"]):
                self.client.indices.create(index=idx["name"], mappings=idx["mapping"])
                logger.info(f"📑 创建 ES 索引：{idx['name']}")
            else:
                logger.debug(f"ℹ️ ES 索引已存在：{idx['name']}")

    def is_available(self) -> bool:
        return self.client is not None

    def indexing_question(self, doc_id: str, questions: str, text: str, metadata: Dict):
        """将生成的假设性问题存入 ES"""
        if not self.client:
            return
            
        document = {
            "doc_id": doc_id,
            "questions": questions,
            "text": text, # 冗余存储原文，方便返回
            "metadata": metadata
        }
        
        try:
            self.client.index(
                index=self.index_questions,
                id=doc_id,
                document=document
            )
            logger.debug(f"📝 ES 写入问题索引：{doc_id}")
        except Exception as e:
            logger.error(f"❌ ES 写入失败：{e}")

    def indexing_summary(self, doc_id: str, summary: str, text: str, metadata: Dict):
        """将摘要存入 ES 专用索引"""
        if not self.client or not summary:
            return
            
        document = {
            "doc_id": doc_id,
            "summary": summary,
            "text": text, # 冗余存储原文，便于直接返回
            "metadata": metadata
        }
        
        try:
            self.client.index(
                index=self.index_summaries,
                id=f"sum_{doc_id}", # 加前缀防止 ID 冲突
                document=document
            )
            logger.debug(f"📝 ES 写入摘要索引：{doc_id}")
        except Exception as e:
            logger.error(f"❌ ES 摘要写入失败：{e}")
    def search_questions(self, query: str, top_k: int = 5) -> List[Dict]:
        """在问题索引中搜索"""
        if not self.client:
            return []
            
        es_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["questions^2", "text"], # questions 字段权重加倍
                    "type": "best_fields"
                }
            },
            "size": top_k
        }
        
        try:
            response = self.client.search(index=self.index_questions, body=es_query)
            hits = response['hits']['hits']
            
            results = []
            for hit in hits:
                source = hit['_source']
                results.append({
                    "text": source.get('text', ''),
                    "score": hit['_score'],
                    "metadata": source.get('metadata', {}),
                    "source_field": "es_questions"
                })
            return results
        except Exception as e:
            logger.error(f"❌ ES 搜索失败：{e}")
            return []
    
    def search_summaries(self, query: str, top_k: int = 5) -> List[Dict]:
        """在摘要索引中搜索"""
        if not self.client:
            return []
            
        es_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["summary^2", "text"], # summary 字段权重加倍
                    "type": "best_fields"
                }
            },
            "size": top_k
        }
        
        try:
            response = self.client.search(index=self.index_summaries, body=es_query)
            hits = response['hits']['hits']
            
            results = []
            for hit in hits:
                source = hit['_source']
                results.append({
                    "text": source.get('text', ''),
                    "score": hit['_score'],
                    "metadata": source.get('metadata', {}),
                    "source_field": "es_summaries"
                })
            return results
        except Exception as e:
            logger.error(f"❌ ES 摘要搜索失败：{e}")
            return []

# 单例
es_client_instance = ESClient()
```

## 3. 实战编码：构建可插拔策略模块

### 第一步：配置驱动 (src/core/config.py)

由于配置项越来越多，很多配置都混用，例如重排序前要使用rewrite、检索插件要使用rewrite，因此需要将配置项进行分类，便于管理，修改了配置后，代码中相应(使用settings)的部分也要进行修改。

其中`model_config = SettingsConfigDict(env_prefix="SEARCH_")` 是 Pydantic Settings（特别是 pydantic-settings 库）中的一个关键配置，用于控制环境变量如何映射到 Pydantic 模型的字段上。

它的意思是：
+ 这个类的所有字段，会优先从以 SEARCH_ 开头的环境变量中读取值。

假设你的 .env 文件或系统环境中有如下变量：
```bash
SEARCH_ENABLE_HYBRID_SEARCH=true
SEARCH_PLUGIN_ES_QUESTIONS=true
SEARCH_ES_HOST=http://my-es:9200
SEARCH_RRF_K=50
```

那么当你创建 SearchStrategySettings() 实例时：
Pydantic 会自动将：
+ SEARCH_ENABLE_HYBRID_SEARCH → 映射到 enable_hybrid_search 字段（转为布尔值 True）
+ SEARCH_PLUGIN_ES_QUESTIONS → 映射到 plugin_es_questions（True）
+ SEARCH_ES_HOST → 映射到 es_host（字符串）
+ SEARCH_RRF_K → 映射到 rrf_k（整数 50）
> 💡 注意：Pydantic 会自动将环境变量名 转为小写 + 下划线命名 来匹配字段名。
> 即：SEARCH_PLUGIN_ES_QUESTIONS → plugin_es_questions


```python
import os
from typing import Optional, List, Dict, Any
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# 加载 .env 文件 (兼容旧习惯，Pydantic-settings 也会自动加载，但显式调用更稳妥)
load_dotenv()

class AppSettings(BaseSettings):
    """应用基础信息"""
    name: str = Field("AI Agent", description="应用名称")
    version: str = Field("0.0.1", description="应用版本")
    description: str = Field("基于 RAG 构建的 AI 智能体", description="应用描述")
    
    model_config = SettingsConfigDict(env_prefix="APP_")

class LLMSettings(BaseSettings):
    """LLM 模型配置"""
    model_name: str = Field("qwen-3-4b", description="模型名称")
    base_url: str = Field("http://localhost:7575/v1", description="API 地址")
    api_key: str = Field("not-needed", description="API Key")
    temperature: float = Field(0.05, ge=0.0, le=1.0, description="生成温度")
    rewrite_temperature: float = Field(0.3, ge=0.0, le=1.0, description="重写温度")
    
    model_config = SettingsConfigDict(env_prefix="LLM_")

class DatabaseSettings(BaseSettings):
    """数据库连接配置 (Redis, MySQL, Milvus)"""
    # Redis
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    
    # MySQL
    database_url: str = "mysql+pymysql://root:mysql2002@localhost:3307/agent_memory"
    
    # Milvus
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_collection: str = "knowledge_base"
    
    # Milvus 索引参数
    milvus_index_type: str = "HNSW"
    milvus_metric_type: str = "COSINE"
    milvus_index_m: int = 8
    milvus_index_ef_construction: int = 200
    milvus_search_ef: int = 64
    
    # --- Elasticsearch (新增) ---
    es_host: str = "http://localhost:9200"       # 本地访问地址
    es_index_questions: str = "knowledge_questions"
    es_index_documents: str = "knowledge_documents"
    es_user: Optional[str] = None                 # 如果开启安全认证
    es_password: Optional[str] = None
    
    model_config = SettingsConfigDict(env_prefix="") # 使用具体前缀在字段级或类级定义

class EmbeddingSettings(BaseSettings):
    """Embedding 模型配置"""
    # 推荐中文：BAAI/bge-small-zh-v1.5, shibing624/text2vec-base-chinese
    # 推荐英文：sentence-transformers/all-MiniLM-L6-v2
    model_name: str = "BAAI/bge-small-zh-v1.5"
    
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

class RagOfflineSettings(BaseSettings):
    """RAG 检索前 (Offline) 配置：分块与入库"""
    # 分块策略选择: 'recursive' (递归), 'fixed' (固定长度), 'sentence' (按句子)
    chunk_strategy: str = "recursive" # recursive, fixed, sentence
    # 分块参数
    chunk_size: int = 500
    chunk_overlap: int = 50
    # 分隔符配置 (仅 recursive 策略有效)
    # 可以使用逗号分隔多个字符，例如 "\n\n,\n,。"
    chunk_separators: str = "\n\n,\n,。,!,?， ,"
    
    @property
    def separators_list(self) -> List[str]:
        return self.chunk_separators.split(',') if self.chunk_separators else ["\n\n"]

class RagOnlineSettings(BaseSettings):
    """RAG 检索后 (Online) 配置：重排、阈值"""
    
    # 重排序 (Re-ranker)
    enable_rerank: bool = True
    rerank_model_name: str = "BAAI/bge-reranker-base"
    rerank_device: str = "cpu"
    torch_num_threads: int = 4
    rerank_max_length: Optional[int] = None
    
    # 召回与过滤
    rough_top_k: int = 8
    final_top_k: int = 3
    score_threshold: float = 0.5
    
    # 动态重排策略
    rerank_dynamic_threshold: float = 0.10

class SearchStrategySettings(BaseSettings):
    """检索策略 (Plugins) 配置：对应 RetrieverComposer"""
    # 总开关：是否启用多路混合检索
    enable_hybrid_search: bool = False
    
    # 插件开关
    plugin_rewritten_query: bool = True       # 对应 VectorRewrittenRetriever 重写
    plugin_es_questions: bool = False         # 对应 ESQuestionsRetriever ES 检索
    
    # 融合算法参数
    rrf_k: int = 60
    
    # 权重配置 (JSON 字符串自动解析为 Dict)
    # 示例：{"text": 0.6, "questions": 0.3, "summary": 0.1}
    field_weights: Dict[str, float] = {"text": 0.6, "questions": 0.3, "summary": 0.1}
    
    # 元数据过滤默认值
    default_filter_category: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="SEARCH_")

class Settings(BaseSettings):
    """
    根配置类：聚合所有模块
    使用时：settings = Settings()
    访问：settings.llm.model_name, settings.rag.online.final_top_k
    """
    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rag_offline: RagOfflineSettings = Field(default_factory=RagOfflineSettings)
    rag_online: RagOnlineSettings = Field(default_factory=RagOnlineSettings)
    search: SearchStrategySettings = Field(default_factory=SearchStrategySettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False, # 环境变量不区分大小写
        extra="ignore"        # 忽略 .env 中未定义的变量
    )

# 实例化全局配置对象
settings = Settings()
```


### 第二步：创建策略基础模块 (src/rag/strategies/)

新建目录 src/rag/strategies/ 并添加 `__init__.py`。

**A. 基础接口 (src/rag/strategies/base.py)**
```py
# src/rag/strategies/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class SearchResult:
    """统一检索结果对象"""
    text: str
    score: float
    metadata: Dict[str, Any]
    source_field: str = "text"
    rank: int = 0 # 用于 RRF 计算
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "source_field": self.source_field
        }

class BaseRetrievalStrategy(ABC):
    """检索策略基类"""
    @abstractmethod
    def search(self, query: str, top_k: int, filter_expr: Optional[str] = None, **kwargs) -> List[SearchResult]:
        pass

class BaseFilterStrategy(ABC):
    """过滤策略基类"""
    @abstractmethod
    def build_expr(self, **kwargs) -> Optional[str]:
        pass
```

**B. 元数据过滤器 (src/rag/strategies/metadata_filter.py)**

这是最快见效的优化。Milvus 支持强大的标量字段过滤。(这一步要配合milvus客户端的更新，详见4)

```py
# src/rag/strategies/metadata_filter.py
from .base import BaseFilterStrategy
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class MetadataFilterBuilder(BaseFilterStrategy):
    def build_expr(self, **kwargs) -> Optional[str]:
        """
        构建 Milvus 标量过滤表达式
        支持：category, source, min_page
        """
        filters = []
        
        # 1. 分类过滤
        category = kwargs.get("category")
        if category:
            # Milvus JSON 字段过滤语法：metadata["key"] == "value"
            filters.append(f'(metadata["category"] == "{category}")')
            
        # 2. 来源文件过滤
        source = kwargs.get("source")
        if source:
            filters.append(f'(metadata["source"] == "{source}")')
            
        # 3. 页码过滤
        min_page = kwargs.get("min_page")
        if min_page is not None:
            filters.append(f'(metadata["page"] >= {int(min_page)})')
            
        if not filters:
            return None
            
        expr = " and ".join(filters)
        logger.info(f"🔒 [Filter] 应用过滤策略：{expr}")
        return expr
```

**C. 检索器插件库 (src/rag/strategies/retrievers)**

所有插件都继承 BaseRetrievalStrategy，确保输入输出一致。

**1. 主路：向量检索 Text (vector_text.py)**

```python
# src/rag/strategies/retrievers/vector_text.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from src.core.milvus_client import get_milvus_client
import logging
from typing import List

logger = logging.getLogger(__name__)

class VectorTextRetriever(BaseRetrievalStrategy):
    """
    主路: 标准向量检索
    目标字段：text
    输入查询：原始查询
    """
    def __init__(self):
        self.milvus = get_milvus_client()
        logger.info("🔌 加载插件：VectorTextRetriever (原文向量检索)")

    def search(self, query: str, top_k: int, filter_expr: str = None, **kwargs) -> List[SearchResult]:
        logger.debug(f"🔍 [VectorText] 检索：{query}")
        hits = self.milvus.search(
            query=query,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=["text", "metadata"]
        )
        
        return [
            SearchResult(
                text=h['text'],
                score=h['score'],
                metadata=h['metadata'],
                source_field="vector_text"
            ) for h in hits
        ]
```

**2. 变体路：向量检索重写 Query (vector_rewritten.py)**
```python
# src/rag/strategies/retrievers/vector_rewritten.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from src.core.milvus_client import get_milvus_client
from src.rag.rewriter import rewriter_instance # 复用之前的重写器
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
        logger.info("🔌 [Plugin] 加载变体向量检索插件 (Rewritten)")

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
        
        return [
            SearchResult(
                text=h['text'],
                score=h['score'],
                metadata=h['metadata'],
                source_field="vector_text"
            ) for h in hits
        ]
```

**3. 变体路：ES questions关键词检索 (es_questions.py)**
```python
# src/rag/strategies/retrievers/es_questions.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from core.es_client import es_client_instance
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class ESQuestionsRetriever(BaseRetrievalStrategy):
    """
    插件 3: ES 关键词检索 (Questions 字段)
    """
    def __init__(self):
        self.es = es_client_instance
        if self.es.is_available():
            logger.info("🔌 [Plugin] ES 检索插件已就绪")
        else:
            logger.warning("🔌 [Plugin] ES 不可用，此插件将自动跳过")

    def search(self, query: str, top_k: int, filter_expr: Optional[str] = None, **kwargs) -> List[SearchResult]:
        if not self.es.is_available():
            return []
            
        # 调用封装好的搜索方法
        # 注意：ES 原生不支持复杂的 JSON 过滤表达式 (如 Milvus 语法)，这里暂不实现 filter_expr
        # 如果需要，可以在 ES 查询中添加 term 过滤
        raw_hits = self.es.search_questions(query, top_k=top_k)
        
        results = []
        for hit in raw_hits:
            results.append(SearchResult(
                text=hit['text'],
                score=hit['score'],
                metadata=hit['metadata'],
                source_field="es_questions"
            ))
        return results
```

**4. 变体路：ES summeries关键词检索 (es_summaries.py)**

结构几乎与questions一样，不再赘述

```python
# src/rag/strategies/retrievers/es_summaries.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from src.core.es_client import es_client_instance
from core.config import settings
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class ESSummariesRetriever(BaseRetrievalStrategy):
    """
    插件 3: ES 关键词检索 (Questions 字段)
    """
    def __init__(self):
        self.es = es_client_instance
        self.index_name = settings.db.es_index_summaries
        if self.es.is_available():
            logger.info(f"🔌 [Plugin] ES 检索插件{self.index_name}已就绪")
        else:
            logger.warning("🔌 [Plugin] ES 不可用，此插件将自动跳过")

    def search(self, query: str, top_k: int, filter_expr: Optional[str] = None, **kwargs) -> List[SearchResult]:
        if not self.es.is_available():
            return []
            
        # 调用封装好的搜索方法
        # 注意：ES 原生不支持复杂的 JSON 过滤表达式 (如 Milvus 语法)，这里暂不实现 filter_expr
        # 如果需要，可以在 ES 查询中添加 term 过滤
        raw_hits = self.es.search_summaries(query, top_k=top_k)
        
        results = []
        for hit in raw_hits:
            results.append(SearchResult(
                text=hit['text'],
                score=hit['score'],
                metadata=hit['metadata'],
                source_field="es_questions"
            ))
        return results
```

**D.  数据入库联动 (src/rag/ingestion.py)**

在数据入库时，同步将 questions、summeray 写入 ES。

```python
# src/rag/ingestion.py
    # 只需要更新process_file方法即可
    def process_file(self, file_path: str, category: str = "general"):
        """处理单个文件：加载 -> 分块 -> 增强 -> 入库"""
        docs = self.load_document(file_path)
        if not docs:
            return

        all_chunks = []
        
        # 分块
        for doc in docs:
            # doc.page_content 是文本，doc.metadata 包含页码等信息
            splits = self.text_splitter.split_documents([doc])
            all_chunks.extend(splits)
        
        logger.info(f"✂️ 分块完成，共生成 {len(all_chunks)} 个 chunks")

        # 入库
        success_count = 0
        for i, chunk in enumerate(all_chunks):
            text = chunk.page_content
            
            # 跳过过短的块（可能是噪点）
            if len(text.strip()) < 20:
                continue

            # 元数据增强 (可选：为了速度，生产环境可异步或批量处理)
            enhanced_meta = self.enhance_metadata(text, chunk.metadata.get("source", ""))
            summary_str = enhanced_meta.get("summary", "")
            questions_str = enhanced_meta.get("questions", "")

            final_metadata = {
                "source": os.path.basename(file_path),
                "page": chunk.metadata.get("page", 0),
                "category": category,
                "summary": summary_str,
                "questions": questions_str
            }
            
            doc_id = f"{os.path.basename(file_path)}_{i}_{uuid.uuid4().hex[:6]}"
            
            # 2. 存入 Milvus (向量库)
            try:
                self.milvus.insert_data(
                    id=doc_id,
                    text=text,
                    metadata=final_metadata
                )
                success_count += 1
            except Exception as e:
                logger.error(f"❌ Milvus 插入失败：{e}")

            # 3. 存入 Elasticsearch (关键词库) - 双管齐下
            if es_client_instance.is_available():
                # A. 写入问题索引 (包含 summary)
                if questions_str:
                    es_client_instance.indexing_question(
                        doc_id=doc_id,
                        questions=questions_str,
                        text=text,
                        metadata=final_metadata
                    )
                
                # B. 写入摘要索引 (新增！)
                if summary_str:
                    es_client_instance.indexing_summary(
                        doc_id=doc_id,
                        summary=summary_str,
                        text=text,
                        metadata=final_metadata
                    )

        logger.info(f"✅ 文件 {file_path} 处理完毕，成功入库 {success_count}/{len(all_chunks)} 条记录")
```

### 第三步：创建融合引擎 (src/rag/fusion/)

新建目录 src/rag/fusion/ 并添加 `__init__.py`。

RRF 融合器 (src/rag/fusion/rrf.py)
```python
# src/rag/fusion/rrf.py
from src.rag.strategies.base import SearchResult
from typing import List, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class RRFFusionEngine:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, result_lists: List[List[SearchResult]], top_k: int) -> List[SearchResult]:
        """
        执行倒数排名融合 (RRF)
        """
        if not result_lists:
            return []
            
        score_map: Dict[str, SearchResult] = {} # Key: text (简化去重)
        
        for list_idx, results in enumerate(result_lists):
            for rank, res in enumerate(results):
                # RRF 分数 = 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank)
                
                # 使用 text 前 60 字符作为唯一键 (生产环境建议用 doc_id)
                key = res.text[:60]
                
                if key in score_map:
                    # 累加分数
                    score_map[key].score += rrf_score
                    # 保留最高原始分
                    if res.score > score_map[key].score: # 注意：这里 score 被复用为 RRF 总分了，需小心
                         # 实际上我们应该有个 separate field for original score, 但为了简单，我们只更新 RRF 分
                         pass
                else:
                    # 克隆对象，分数初始化为 RRF 分
                    new_res = SearchResult(
                        text=res.text,
                        score=rrf_score, # 初始为 RRF 分
                        metadata=res.metadata.copy(),
                        source_field="fused"
                    )
                    # 保存原始分数供参考
                    new_res.metadata['_original_score'] = res.score
                    score_map[key] = new_res
        
        # 按 RRF 总分降序排列
        final_results = sorted(score_map.values(), key=lambda x: x.score, reverse=True)
        
        logger.info(f"✨ [RRF] 融合完成：{len(result_lists)} 路 -> {len(final_results)} 条 (Top-{top_k})")
        return final_results[:top_k]
```


### 第四步： 重构组装器 (src/rag/strategies/composer.py)

这是新的核心。它不再是一个具体的检索器，而是一个导演，负责根据配置实例化插件，并行执行，然后融合。

为了实现多路召回时的异步并行 ，我们需要做三个小改动：
1. 将 search 方法改为 async 异步方法。
2. 将循环调用改为 asyncio.gather 并发执行。
3. 关键：由于你的底层库（`pymilvus`, `elasticsearch-py`）是同步阻塞的，直接 `await` 会卡死事件循环。我们需要用 `asyncio.to_thread()` 将它们包裹起来，放到线程池中运行，从而实现真正的“非阻塞并行”。


```py
# src/rag/strategies/composer.py
from src.rag.strategies.base import BaseRetrievalStrategy, SearchResult
from src.rag.strategies.retrievers.vector_text import VectorTextRetriever
from src.rag.strategies.retrievers.vector_rewritten import VectorRewrittenRetriever
from src.rag.strategies.retrievers.es_questions import ESQuestionsRetriever
from src.rag.strategies.retrievers.es_summaries import ESSummariesRetriever # 导入新插件
from src.rag.fusion.rrf import RRFFusionEngine
from src.core.config import settings
from typing import List, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)

class RetrieverComposer:
    """
    检索器组装器
    职责：根据配置动态加载多个检索插件，并行执行，并使用 RRF 融合结果。
    """
    def __init__(self):
        self.retrievers: List[BaseRetrievalStrategy] = []
        self.rrf_engine = RRFFusionEngine(k=settings.search.rrf_k)
        
        self._load_plugins()

    def _load_plugins(self):
        """根据配置动态加载插件"""
        # 1. 主路：永远加载
        self.retrievers.append(VectorTextRetriever())
        logger.info("✅ [Composer] 已加载主路：VectorText")
        
        # 2. 变体路：如果开启混合检索
        if settings.search.enable_hybrid_search:
            # 2. 改写路：如果开启改写
            if settings.search.plugin_rewritten_query:
                self.retrievers.append(VectorRewrittenRetriever())
                logger.info("✅ [Composer] 已加载变体路：VectorRewritten")
            
            # 3. ES 路：如果配置了 ES
            if settings.db.es_host:
                # 3. ES - Questions 路
                if settings.search.plugin_es_questions:
                    es_retriever = ESQuestionsRetriever()
                    if es_retriever.es.is_available(): # 只有连接成功才加入
                        self.retrievers.append(es_retriever)
                        logger.info("✅ [Composer] 已加载 ES - Questions 路：ESQuestions")
                # 4. ES - Summaries 路
                if settings.search.plugin_es_summaries:
                    es_retriever = ESSummariesRetriever()
                    if es_retriever.es.is_available(): # 只有连接成功才加入
                        self.retrievers.append(es_retriever)
                        logger.info("✅ [Composer] 已加载 ES - Summaries 路：ESSummaries")

        async def search(self, query: str, rough_top_k: int, filter_expr: Optional[str] = None, **kwargs) -> List[SearchResult]:
        """
        异步并行执行多路检索
        """
        logger.info(f"🚀 [Composer] 开始异步多路检索 ({len(self.retrievers)} 路)...")
        
        # 定义一个内部异步包装器，用于在线程池中运行同步的 retriever.search
        async def run_retriever(retriever: BaseRetrievalStrategy) -> List[SearchResult]:
            try:
                # asyncio.to_thread 将阻塞的同步代码放入线程池，避免阻塞主事件循环
                results = await asyncio.to_thread(
                    retriever.search, 
                    query, 
                    rough_top_k, 
                    filter_expr=filter_expr, 
                    **kwargs
                )
                if results:
                    logger.debug(f"✅ [{retriever.__class__.__name__}] 完成，召回 {len(results)} 条")
                    return results
                return []
            except Exception as e:
                logger.error(f"❌ [Composer] 插件 {retriever.__class__.__name__} 执行失败：{e}")
                return []

        # 1. 创建所有检索任务
        tasks = [run_retriever(r) for r in self.retrievers]
        
        # 2. 并行执行 (gather)
        # return_exceptions=True 确保某个插件崩溃不会导致整个 gather 失败
        all_results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 3. 过滤掉异常对象和空列表，只保留有效的结果列表
        valid_results_lists = [
            res for res in all_results_lists 
            if isinstance(res, list) and len(res) > 0
        ]
        
        if not valid_results_lists:
            logger.warning("⚠️ [Composer] 所有检索路均无结果或失败")
            return []
            
        # 4. 如果只有一路有效，直接返回并排序 (跳过 RRF 以节省时间)
        if len(valid_results_lists) == 1:
            final_results = sorted(valid_results_lists[0], key=lambda x: x.score, reverse=True)
            logger.info(f"✨ [Composer] 单路有效检索完成，共 {len(final_results)} 条")
            return final_results[:rough_top_k]
        
        # 5. 多路结果，执行 RRF 融合
        logger.info(f"🔄 [Composer] 执行 RRF 融合 ({len(valid_results_lists)} 路输入)")
        fused_results = self.rrf_engine.fuse(valid_results_lists, rough_top_k)
        return fused_results

# 单例
composer_instance = RetrieverComposer()
```

## 🔧 4. 升级 Milvus 客户端 (src/core/milvus_client.py)

修改 `src/core/milvus_client.py` 中的 `search` 方法，添加过滤和输出字段支持。
```py
    def search(self, query: str, top_k: int = 3, filter_expr: str = None, output_fields: List[str] = None) -> List[dict]:
        """支持元数据过滤和指定输出字段的搜索"""
        query_vector = self.embed_text(query)
        
        search_params = {
            "metric_type": settings.MILVUS_METRIC_TYPE,
            "params": {"ef": settings.MILVUS_SEARCH_EF}
        }
        
        # 默认输出
        if not output_fields:
            output_fields = ["text", "metadata"]
            
        try:
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,  # 👈 支持过滤
                output_fields=output_fields
            )
        except Exception as e:
            logger.error(f"❌ Milvus 搜索失败 (Expr: {filter_expr}): {e}")
            return []
        
        hits = []
        for hit in results[0]:
            # 注意：这里先不过滤阈值，交给上层策略处理，以便 RRF 能看到更多候选
            hits.append({
                "text": hit.entity.get("text"),
                "score": hit.score,
                "metadata": hit.entity.get("metadata")
            })
        return hits
```

## 5. 重构 pipeline.py：组装策略

现在 Pipeline 变得非常干净，只需要使用这样的完整流程即可 Query -> filter（元数据过滤组件） -> 多路Search（composer） -> Result。由于多路召回组件用了异步,所以这里的run也要修改为异步.

```py
# src/rag/pipeline.py
from src.rag.rewriter import rewriter_instance
from src.rag.factories import RerankerFactory # 👈 新增导入
from src.rag.strategies.metadata_filter import MetadataFilterBuilder
from src.rag.strategies.composer import composer_instance # 导入多路召回组件
from src.rag.strategies.base import SearchResult # 👈 导入 SearchResult 类
from src.core.config import settings
from src.utils.xml_parser import remove_think_and_n
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    def __init__(self):
        # 导入元数据过滤组件
        self.filter_builder = MetadataFilterBuilder()
        self.default_filter_category = settings.search.default_filter_category
        # 导入多路召回组件
        self.composer = composer_instance
        # 👇 修改点：使用工厂获取重排器
        self.reranker = RerankerFactory.get_reranker()
        logger.info(f"⚙️ Pipeline 初始化完成 (重排器：{'已加载' if self.reranker else '未加载'})")
        
        # 读取配置
        self.rough_top_k = settings.rag_online.rough_top_k
        self.final_top_k = settings.rag_online.final_top_k
        self.dynamic_threshold = settings.rag_online.score_threshold
        
        logger.info(f"⚙️ Pipeline 初始化：粗排Top{self.rough_top_k}, 动态阈值={self.dynamic_threshold}")

    def _should_trigger_rerank(self, candidates: List[SearchResult]) -> bool:
        """
        动态判断是否需要重排 (支持 SearchResult 对象)
        策略：如果第 1 名和第 2 名的分数差距很小，说明难以抉择，需要重排。
        """
        if not self.reranker:
            return False
        if len(candidates) < 2:
            return False # 只有一个结果，没必要重排
        
        score_1 = candidates[0].score
        score_2 = candidates[1].score
        gap = score_1 - score_2
        
        logger.debug(f"📊 粗排分数分析：Top1={score_1:.4f}, Top2={score_2:.4f}, 差距={gap:.4f}")
        
        # 如果差距小于阈值，触发重排
        if gap <= self.dynamic_threshold:
            logger.info("⚡ 分数差距较小，触发重排序...")
            return True
        else:
            logger.info("✅ 分数差距明显，跳过重排序 (节省资源)")
            return False
    async def run(self, query: str, top_k: int = 3, category: Optional[str] = None) -> List[SearchResult]:
        """
        执行完整的检索流程 (Advanced RAG Online Flow)
        Flow: Query -> filter -> 多路Search -> Result
        """
        final_query = query

        # step 1: 构建过滤表达式
        filter_category = category or self.default_filter_category
        filter_expr = self.filter_builder.build_expr(category=filter_category)
        

        # Step 2: 执行多路检索与融合
        rough_results = await self.composer.search(
            query=query, 
            rough_top_k=self.rough_top_k, 
            filter_expr=filter_expr
        )
        logger.info(f"🔍 [Pipeline] 多路检索与融合：{final_query} (召回 {len(rough_results)} 条)")
        
        if not rough_results:
            return []
        
        # Step 4: ⚡ 动态重排决策 (Re-ranking)
        if self._should_trigger_rerank(rough_results):
            # 触发重排
            rough_results = self.reranker.rerank(final_query, rough_results, top_k=top_k)
        
        # Step 5: 阈值过滤
        filtered_results = []
        for r in rough_results:
            if r.score >= self.dynamic_threshold:
                filtered_results.append(r)
            else:
                logger.debug(f"🚫 [Pipeline] 分数 {r.score:.2f} 低于阈值 {self.dynamic_threshold}, 过滤")
        
        return filtered_results[:top_k]

# 单例
pipeline_instance = RetrievalPipeline()
```

由于有一个数据结构`SearchResult`的修改,`src/rag/reranker.py`也要修改

```py
from src.rag.strategies.base import SearchResult # 👈 导入 SearchResult 类
...[其他部分不变]
    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 3) -> List[SearchResult]:
        if not candidates:
            return []
        
        # 准备输入对
        pairs = [[query, cand.text] for cand in candidates]
        
        logger.debug(f"⚖️ 开始对 {len(pairs)} 个候选项进行重排序...")
        
        try:
            # 批量计算分数
            scores = self.reranker.predict(
                pairs, 
                convert_to_numpy=True, 
                show_progress_bar=False
            )
            
            # 确保 scores 是列表 (某些版本单元素可能返回 float)
            if isinstance(scores, (int, float)):
                scores = [scores] * len(candidates)
            else:
                scores = scores.tolist()

        except Exception as e:
            logger.error(f"❌ 重排序计算失败：{e}")
            return candidates[:top_k]
        
        # 回填分数
        for i, cand in enumerate(candidates):
            cand.rerank_score = scores[i]
        
        ranked_candidates = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)
        final_results = ranked_candidates[:top_k]
        
        if final_results:
            logger.info(f"✨ 重排序完成：最高分 {final_results[0].rerank_score:.4f}")
        
        # 转换为字典格式返回
        return [
            SearchResult(
                text=cand.text,
                score=cand.rerank_score,
                source_field=cand.source_field,
                metadata=cand.metadata,
                rank = i,
            ) for i, cand in enumerate(final_results)
        ]
```

## 6. 测试结果

### ES的存量同步

因为之前测试时，还没用上ES，只存了Milvus，但是Milvus中的metadata也保存着question和summery。利用 Milvus 中已有的 metadata（包含 questions 和 summary）直接同步到 ES，可以避免重新加载原始文档、重新分块、重新调用 LLM 生成摘要和问题，极大地节省了时间和算力。

在 `src/core/es_client.py` 中新增了一个 sync_from_milvus 方法。它会：
1. 连接 Milvus。
2. 分批拉取所有数据（包含 text, metadata）。
3. 解析 metadata 中的 questions 和 summary。
4. 批量写入 ES 的对应索引。
5. 提供进度日志和去重/更新逻辑（使用相同的 doc_id，ES 会自动覆盖更新）。

⚠️ 重要前置步骤：扩展 `milvus_client.py`,由于 Milvus 的 search 需要向量且不适合全量扫描，需要在 src/core/milvus_client.py 中补充这个辅助方法：
```py
# 在 src/core/milvus_client.py
    def scan_collection(self, limit: int = 500, offset: int = 0) -> List[Dict]:
        """
        扫描集合中的数据 (用于全量同步)
        注意：Milvus 深度分页性能有限，仅适用于离线同步任务。
        生产环境建议使用 Milvus 2.3+ 的 Iterator API。
        """
        try:
            # 使用空表达式查询所有，配合 limit 和 offset
            # output_fields 必须包含 id, text, metadata
            results = self.collection.query(
                expr="",  # 空表达式表示所有
                output_fields=["text", "metadata"],
                limit=limit,
                offset=offset
            )
            return results
        except Exception as e:
            logger.error(f"❌ Milvus 扫描失败：{e}")
            return []
```

扩展 `es_client.py`：
```py
# src/core/es_client.py

from elasticsearch import Elasticsearch, helpers

    def sync_from_milvus(self, batch_size: int = 500):
        # 在函数里导入，避免循环依赖
        from src.core.milvus_client import get_milvus_client
        """
        【存量同步】从 Milvus 拉取已有数据，同步到 Elasticsearch
        利用 Milvus metadata 中已存储的 questions 和 summary，无需重新计算。
        """
        if not self.is_available():
            logger.error("❌ ES 不可用，无法执行同步")
            return

        logger.info("🚀 开始从 Milvus 同步数据到 Elasticsearch...")
        milvus = get_milvus_client()
        
        try:
            # 这里我们采用迭代游标的方式，直到取不到数据为止
            offset = 0
            total_synced = 0
            total_questions = 0
            total_summaries = 0
            
            # 准备批量写入 ES 的动作列表
            actions_questions = []
            actions_summaries = []
            
            while True:
                # 2. 分批从 Milvus 检索数据
                # 🚀 这里提供一个基于 `milvus_client` 扩展的建议实现：
                hits = milvus.scan_collection(limit=batch_size, offset=offset)
                
                if not hits:
                    break
                
                for hit in hits:
                    doc_id = hit.get('id') # 假设返回了 id
                    text = hit.get('text')
                    metadata = hit.get('metadata', {})
                    
                    questions = metadata.get('questions', '')
                    summary = metadata.get('summary', '')
                    
                    # 准备 Questions 索引数据
                    if questions:
                        actions_questions.append({
                            "_index": settings.db.es_index_questions,
                            "_id": doc_id,
                            "_source": {
                                "doc_id": doc_id,
                                "questions": questions,
                                "summary": summary,
                                "text": text,
                                "metadata": metadata
                            }
                        })
                        total_questions += 1
                    
                    # 准备 Summaries 索引数据
                    if summary:
                        actions_summaries.append({
                            "_index": settings.db.es_index_summaries,
                            "_id": f"sum_{doc_id}",
                            "_source": {
                                "doc_id": doc_id,
                                "summary": summary,
                                "text": text,
                                "metadata": metadata
                            }
                        })
                        total_summaries += 1
                    
                    total_synced += 1

                # 批量写入 ES
                if actions_questions:
                    helpers.bulk(self.client, actions_questions)
                    actions_questions = []
                if actions_summaries:
                    helpers.bulk(self.client, actions_summaries)
                    actions_summaries = []
                    
                logger.info(f"⏳ 已处理 {total_synced} 条...")
                offset += batch_size
                
                # 如果返回数量小于 batch_size，说明到头了
                if len(hits) < batch_size:
                    break

            # 写入剩余数据
            if actions_questions:
                helpers.bulk(self.client, actions_questions)
            if actions_summaries:
                helpers.bulk(self.client, actions_summaries)

            logger.info(f"✅ 同步完成！共处理 {total_synced} 条记录。")
            logger.info(f"   -> 问题索引写入：{total_questions} 条")
            logger.info(f"   -> 摘要索引写入：{total_summaries} 条")

        except Exception as e:
            logger.error(f"❌ 同步过程中发生错误：{e}")
            import traceback
            traceback.print_exc()
```

建一个临时脚本 `src/test/sync_es.py` 运行一次即可：
```python
from src.core.es_client import es_client_instance

if __name__ == "__main__":
    if not es_client_instance.is_available():
        print("❌ ES 不可用，请检查配置")
        exit(1)
    
    print("🚀 开始存量同步...")
    es_client_instance.sync_from_milvus(batch_size=500)
    print("✅ 同步结束")
```

运行该脚本
```bash
python -m src.test.sync_es
```

可以在ElasticHD 看到：es_index_summaries和es_index_questions这两个索引都有17条数据，正好对应了之前添加的文档。

### 召回测试

编写一个测试脚本 `src/test/recall_test.py` ：
```py
# src/test/recall_test.py
# python -m src.test.recall_test
import asyncio
import logging
from typing import List, Dict, Any
from src.rag.pipeline import pipeline_instance
from src.core.config import settings
from src.rag.strategies.base import SearchResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RecallEval")

# ==========================================
# 1. 测试数据集定义
# ==========================================

TEST_CASES = [
    {
        "category": "精确关键词匹配 (Easy)",
        "questions": [
            "根据《奇葩星球有限公司员工差旅与报销标准手册》，乘坐高铁的票价超过多少元将被视为“奢侈出行”？",
            "在《小模型训练奇谭》中，推荐的“摸鱼Transformer”模型有多少参数？",
            "奇葩星球公司禁止报销哪种咖啡？",
        ],
        "keywords": [
            ["高铁", "奢侈出行", "10元"],
            ["摸鱼Transformer", "参数", "42000"],
            ["禁止报销", "咖啡", "星巴克"]
        ]
    },
    {
        "category": "语义理解与推理 (Medium)",
        "questions": [
            "如果一名员工想通过步行出差获得最高荣誉，他每走一公里能拿到多少补贴？",
            "在预算极其有限的情况下，《小模型训练奇谭》认为真正的智能取决于什么？",
            "《小模型训练奇谭》推荐哪些非传统的数据来源用于训练小模型？",
        ],
        "keywords": [
            ["步行", "补贴", "50元"],
            ["智能", "脑洞密度", "参数量"],
            ["数据来源", "微信群", "分手短信", "流浪猫"]
        ]
    },
    {
        "category": "多跳推理 (Hard)",
        "questions": [
            "一名员工因工作压力在出差途中哭泣，根据公司规定，他如何能将情绪转化为实际的经济补偿？",
            "《小模型训练奇谭》中提到的“穷人版分布式训练”是如何利用社交工具实现的？",
        ],
        "keywords": [
            ["哭泣", "眼泪", "情感天平", "补贴"],
            ["穷人版分布式训练", "微信群", "梯度", "红包"]
        ]
    },
    {
        "category": "区分相似概念 (Confusion)",
        "questions": [
            "“泡面”在《奇葩星球有限公司员工差旅与报销标准手册》和《小模型训练奇谭》中分别扮演什么角色？",
            "两份文件都提到了“老板”，它们各自的态度或处理方式有何核心区别？",
        ],
        "keywords": [
            ["泡面", "觉醒补贴", "泡面RNN", "电热水壶"],
            ["老板", "黑暗料理", "画饼", "老板探测器"]
        ]
    }
]

# ==========================================
# 2. 评估逻辑
# ==========================================

def check_hit(results: List[SearchResult], keywords: List[str]) -> Dict[str, Any]:
    """
    检查检索结果是否命中包含关键词的文档
    返回：{'hit': bool, 'rank': int, 'source_plugin': str, 'snippet': str}
    """
    for rank, res in enumerate(results):
        text = res.text.lower()
        # 只要命中任意一个关键词组中的部分词（简化逻辑：命中2个以上关键词即算命中）
        # 或者更严格：必须命中所有关键词？这里采用宽松策略：命中至少 2 个关键概念
        match_count = sum(1 for kw in keywords if kw.lower() in text)
        
        # 简单策略：如果关键词列表中有任意一个词出现在文本中，且该词长度>2（避免停用词）
        # 这里为了演示，我们检查是否包含列表中的任意一个长词
        is_hit = False
        for kw in keywords:
            if len(kw) > 2 and kw.lower() in text:
                is_hit = True
                break
        
        if is_hit:
            return {
                "hit": True,
                "rank": rank + 1,
                "source_plugin": res.source_field,
                "snippet": text[:100] + "...",
                "score": res.score
            }
    
    return {
        "hit": False,
        "rank": -1,
        "source_plugin": "None",
        "snippet": "No relevant chunk found.",
        "score": 0
    }

async def run_single_test(question: str, keywords: List[str], top_k: int = 5) -> Dict[str, Any]:
    """运行单个问题的测试"""
    logger.info(f"\n❓ 测试问题：{question}")
    
    try:
        # 调用异步 Pipeline 现在返回 List[SearchResult]
        results = await pipeline_instance.run(query=question, top_k=top_k)
        
        if not results:
            logger.warning("⚠️ 无检索结果")
            return {"question": question, "hit": False, "rank": -1, "total_results": 0, "rerank_triggered": False}
        
        # 评估命中率
        eval_result = check_hit(results, keywords)
        
        # 记录是否触发重排 (通过日志或结果推断，这里简单通过结果数量判断，实际可加 flag)
        # 注意：pipeline 内部日志已经输出了是否触发重排，这里主要关注结果
        
        logger.info(f"✅ 评估结果：Hit={eval_result['hit']}, Rank={eval_result['rank']}, Source={eval_result['source_plugin']}")
        logger.debug(f"   片段：{eval_result['snippet']}")
        
        return {
            "question": question,
            "hit": eval_result["hit"],
            "rank": eval_result["rank"],
            "source_plugin": eval_result["source_plugin"],
            "total_results": len(results),
            "top_score": results[0].score if results else 0
        }
        
    except Exception as e:
        logger.error(f"❌ 执行出错：{e}")
        return {"question": question, "hit": False, "rank": -1, "error": str(e)}

async def main():
    print("="*80)
    print("🚀 开始 RAG 召回能力全面评估")
    print(f"⚙️ 配置：Hybrid={settings.search.enable_hybrid_search}, Rerank={settings.rag_online.enable_rerank}")
    print("="*80)
    
    all_results = []
    category_stats = {}
    
    for case in TEST_CASES:
        cat_name = case["category"]
        logger.info(f"\n--- 类别：{cat_name} ---")
        category_stats[cat_name] = {"total": 0, "hit": 0, "ranks": []}
        
        for q, kws in zip(case["questions"], case["keywords"]):
            res = await run_single_test(q, kws, top_k=5)
            res["category"] = cat_name
            all_results.append(res)
            
            category_stats[cat_name]["total"] += 1
            if res.get("hit"):
                category_stats[cat_name]["hit"] += 1
                category_stats[cat_name]["ranks"].append(res["rank"])
    
    # ==========================================
    # 3. 生成报告
    # ==========================================
    print("\n" + "="*80)
    print("📊 评估报告总结")
    print("="*80)
    
    total_q = len(all_results)
    total_hit = sum(1 for r in all_results if r.get("hit"))
    overall_hit_rate = total_hit / total_q if total_q > 0 else 0
    
    # 计算 MRR (Mean Reciprocal Rank)
    reciprocal_ranks = [1.0/r["rank"] for r in all_results if r.get("hit") and r["rank"] > 0]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    print(f"\n🏆 总体表现:")
    print(f"   - 总问题数：{total_q}")
    print(f"   - 命中数 (Hit@5): {total_hit}")
    print(f"   - 命中率 (Hit Rate): {overall_hit_rate:.2%}")
    print(f"   - 平均倒数排名 (MRR): {mrr:.4f} (越接近 1 越好)")
    
    print(f"\n📈 分类别表现:")
    print(f"{'类别':<30} | {'命中率':<10} | {'MRR':<10} | {'平均排名':<10}")
    print("-" * 70)
    
    for cat, stats in category_stats.items():
        hit_rate = stats["hit"] / stats["total"] if stats["total"] > 0 else 0
        ranks = stats["ranks"]
        cat_mrr = sum([1.0/r for r in ranks]) / len(ranks) if ranks else 0
        avg_rank = sum(ranks) / len(ranks) if ranks else 0
        
        print(f"{cat:<30} | {hit_rate:>6.2%}     | {cat_mrr:>6.4f}     | {avg_rank:>6.2f}")
    
    print("\n💡 建议:")
    if overall_hit_rate < 0.6:
        print("   ⚠️ 召回率较低，建议检查 ES 索引是否同步完成，或调整 IK 分词配置。")
    if mrr < 0.5:
        print("   ⚠️ 排名靠后，建议检查 RRF 融合权重或重排序阈值。")
    if settings.search.enable_hybrid_search == False:
        print("   💡 当前未开启混合检索，尝试设置 SEARCH_ENABLE_HYBRID_SEARCH=True 以提升效果。")
    
    print("="*80)

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
```

运行该脚本
```bash
python -m src.test.recall_test
```

运行结果:

![alt text](./img/01recall测试.png)

![alt text](./img/02recall测试.png)

![alt text](./img/03recall测试.png)

结果那么高的原因可能有两个
1. 数据量太少了几乎没有confuse的关键词
2. 我们的模型太强了,哈哈哈


### 可能报错

**报错1：**

```text
ERROR: Elasticsearch did not exit normally - check the logs at /usr/share/elasticsearch/logs/docker-cluster.log
ERROR: Elasticsearch exited unexpectedly, with exit code 1
Mar 01, 2026 3:51:24 PM sun.util.locale.provider.LocaleProviderAdapter <clinit>
WARNING: COMPAT locale provider will be removed in a future release
ERROR: Elasticsearch did not exit normally - check the logs at /usr/share/elasticsearch/logs/docker-cluster.log
ERROR: Elasticsearch exited unexpectedly, with exit code 1
Mar 01, 2026 3:52:27 PM sun.util.locale.provider.LocaleProviderAdapter <clinit>
WARNING: COMPAT locale provider will be removed in a future release
```

这通常意味着 ES 已完成初始化、成为 master 节点，并开始处理集群元数据。但另一方面，容器却退出了（exit code 1），并反复报错：这说明：Elasticsearch 进程在启动后不久就崩溃或被终止了。

**🔍 可能原因分析**
✅ 1. JVM 内存不足（最可能！）

将内存调高到至少 1GB：
```yaml
environment:
  - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
```

ES服务healthy，但是 ElisticHD 显示链接不上
```text
info": "Get http://127.0.0.1:9200///_search: dial tcp 127.0.0.1:9200: getsockopt: connection refused"
```

ElasticHD 的 Web 界面允许用户在浏览器中输入 ES 地址。如果在页面上填的 http://127.0.0.1:9200 或 http://localhost:9200，

这就跟他自己内部的docker容器不同了，正常来说他就是要访问 `http://elasticsearch:9200` ，只是我使用了NAT模式将所有端口都转发到宿主机了，宿主机可以访问 http://127.0.0.1:9200 但不代表他ElisticHD这个docker容器也可以访问，所以只需将链接修改为 `http://elasticsearch:9200` 就可以了


**报错2：**
```text
BadRequestError(400, 'mapper_parsing_exception', 'Failed to parse mapping: analyzer [ik_max_word] has not been configured in mappings')
```

+ `ik_max_word` 不是 Elasticsearch 自带的分词器。
+ 含义：它是中文分词插件 IK Analyzer 提供的一种模式，会将文本进行最细粒度的拆分（例如：“清华大学”会被拆分为“清华”、“华大”、“大学”、“清华大学”），非常适合搜索索引，能最大化召回率。
+ 报错原因：你的 ES 容器是官方纯净版 (docker.elastic.co/elasticsearch/elasticsearch:8.11.0)，里面没有安装 IK 插件。所以当你试图在 mapping 中配置 analyzer: "ik_max_word" 时，ES 不认识它，直接报错。

方案 A：安装 IK 插件（推荐用于生产/正式测试）
你需要修改 docker-compose.yml，让 ES 容器在启动时自动安装插件。
```yaml
elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    ...保持不变
    # 👇 新增：启动时自动安装 IK 插件
    command: >
      bash -c '
        echo "🚀 开始安装 IK 分词插件..." &&
        bin/elasticsearch-plugin install --batch https://release.infinilabs.com/analysis-ik/stable/elasticsearch-analysis-ik-8.11.0.zip &&
        echo "✅ IK 插件安装完成，启动 ES..." &&
        /usr/local/bin/docker-entrypoint.sh
      '
```


## 🏗️ 项目架构全景图与演进规划

随着 Elasticsearch (IK 分词)、异步并行检索、多路召回策略 以及 存量同步工具 的加入，你的系统已经从一个“高级 RAG”进化为一个企业级混合检索智能体平台。

现在的架构具备以下核心能力：
1. 混合检索 (Hybrid Search)：向量语义 (Milvus) + 关键词精确匹配 (ES)。
2. 多路召回 (Multi-Retrieval)：原文、重写查询、假设性问题、摘要四路并行。
3. 异步高并发：基于 asyncio.gather 的并行执行，大幅降低延迟。
4. 模块化插件：所有检索策略均可插拔、可配置。
5. 数据双写与同步：支持实时入库和存量数据一键同步。

### 📂 当前项目结构解析

```text
src/
├── main.py                      # 🚀 应用入口 (启动 Agent 或 CLI)
│
├── core/                        # 🧱 基础设施层 (Infrastructure)
│   ├── config.py                # ⚙️ 配置中心 (Pydantic 模块化配置)
│   ├── redis_client.py          # 💾 短期记忆 (LangGraph Checkpointer)
│   ├── db_session.py            # 🗄️ MySQL 连接池 (ORM 基础)
│   ├── models.py                # 📑 MySQL 数据模型 (User Profiles)
│   ├── milvus_client.py         # 🔭 向量数据库客户端 (含 scan_collection 同步支持)
│   └── es_client.py             # 🔎 ES 客户端 (含 IK 分词、双索引、sync_from_milvus)
│
├── rag/                         # 🧠 RAG 引擎层 (Modular RAG Core) ⭐ 核心进化区
│   ├── ingestion.py             # 📥 数据摄入管道 (双写 Milvus + ES)
│   ├── pipeline.py              # 🔄 异步检索执行流 (Async/Await)
│   ├── rewriter.py              # ✍️ 查询重写器
│   ├── reranker.py              # ⚖️ 重排序模块 (CPU 优化 + 动态开关)
│   ├── chunkers.py              # ✂️ 分块策略集
│   ├── factories.py             # 🏭 工厂模块
│   │
│   ├── strategies/              # 🧩 策略模式核心目录
│   │   ├── base.py              # 基础接口 (SearchResult, BaseRetriever)
│   │   ├── composer.py          # 🎼 检索器组装器 (Asyncio Gather + RRF)
│   │   ├── metadata_filter.py   # 🔒 元数据过滤构建器
│   │   └── retrievers/          # 🔌 检索插件库 (可插拔)
│   │       ├── vector_text.py   # 插件 1: 主路向量检索 (Text)
│   │       ├── vector_rewritten.py # 插件 2: 变体向量检索 (Rewritten)
│   │       ├── es_questions.py  # 插件 3: ES 问题检索 (Questions)
│   │       └── es_summaries.py  # 插件 4: ES 摘要检索 (Summaries)
│   │
│   └── fusion/                  # 🔗 融合算法库
│       └── rrf.py               # 倒数排名融合 (RRF)
│
├── Mini_Agent/                  # 🤖 Agent 编排层 (LangGraph Logic)
│   ├── state.py                 # 📦 状态定义
│   ├── graph.py                 # 🕸️ 工作流图
│   └── tools/                   # 🛠️ 工具集
│       ├── base_tools.py        # ⏰ 基础工具
│       ├── memory_tools.py      # 📒 长期记忆工具
│       └── rag_tools.py         # 📚 知识库工具 (调用 Async Pipeline)
│
├── utils/                       # 🔧 通用工具类
│   └── xml_parser.py            # 📝 XML 格式解析
│
└── test/                        # 🧪 测试验证集
    ├── sync_es.py               # 🔄 存量数据同步脚本 (Milvus -> ES)
    ├── recall_test.py           # 📊 召回率测试
    ├── test_dynamic_rerank.py   # ⚡ 动态重排测试
    └── ...
```

### 🌊 系统数据流向架构图

![](./img/03离线数据管道.png)

![](./img/03AIagent架构图.png)

**🔑 关键模块职责说明**

| 模块层级 | 核心组件 | 职责与特性 |
| :--- | :--- | :--- |
| 配置层 | `config.py` | 大脑。Pydantic 模块化配置，控制所有插件开关、阈值、连接信息。 |
| 存储层 | `Milvus` + `ES` | 双引擎。Milvus 负责语义模糊匹配，ES (IK 分词) 负责关键词精确匹配。 |
| 策略层 | `retrievers/` | 插件库。4 大检索插件并行工作，互不干扰，易于扩展。 |
| 调度层 | `composer.py` | 指挥官。使用 `asyncio.gather` 并行发起请求，使用 `RRF` 融合结果。 |
| 优化层 | `reranker.py` | 守门员。动态决策是否重排，CPU 线程限制，防止资源耗尽。 |
| 工具层 | `sync_es.py` | 搬运工。一键将 Milvus 存量数据同步至 ES，无需重新计算。 |

