import os
from typing import Optional, List, Dict, Any
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# 加载 .env 文件 (兼容旧习惯，Pydantic-settings 也会自动加载，但显式调用更稳妥)
load_dotenv()

# HuggingFace 离线模式开关：
# 默认启用离线，避免模型加载时频繁 HEAD/GET 请求。
# 如需临时联网，可在环境变量中设置 HF_OFFLINE=0。
if os.getenv("HF_OFFLINE", "1").lower() in ("1", "true", "yes", "on"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
    # 稍微增加创造性以扩展同义词
    rewrite_temperature: float = Field(0.3, ge=0.0, le=1.0, description="重写温度")
    hyde_temperature: float = Field(0.7, ge=0.0, le=1.0, description="HyDE 温度")
    
    model_config = SettingsConfigDict(env_prefix="LLM_")

class DatabaseSettings(BaseSettings):
    """数据库连接配置 (Redis, MySQL, Milvus)"""
    # Redis
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    
    # MySQL
    database_url: str = "mysql+pymysql://root:mysql2002@localhost:3307/agent_memory"

    # PostgreSQL (评估数据等场景使用，一行 URL 配置即可) 
    postgres_database_url: str = "postgresql+psycopg2://agent_user:postgresql_2002@127.0.0.1:5433/agent_dev"
    pool_pre_ping: bool = True
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

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
    es_index_questions: str = "knowledge_questions" # 存储假设性问题的索引
    es_index_summaries: str = "knowledge_summaries"  # 存储总结的索引
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
    # 分块策略选择: 'recursive' (递归), 'fixed' (固定长度), 'parent_child' (父子)
    chunk_strategy: str = "recursive" # recursive, fixed, parent_child
    # 分块参数
    chunk_size: int = 500
    chunk_overlap: int = 50
    # 分隔符配置 
    # 可以使用逗号分隔多个字符，例如 "\n\n,\n,。"
    chunk_separators: str = "\n\n,\n,。,!,?， ,"
    
    # 父子分块特有参数
    # 子分块器配置
    child_chunk_size: int = 80
    child_chunk_overlap: int = 0
    child_splitter_strategy: str = "recursive"  # 可选：recursive, sentence
    min_sentence_length: int = 10
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
    enable_hybrid_search: bool = True
    
    # 插件开关
    plugin_rewritten_query: bool = True       # 对应 VectorRewrittenRetriever 重写
    plugin_es_questions: bool = False         # 对应 ESQuestionsRetriever ES 检索
    plugin_es_summaries: bool = False         # 对应 ESSummariesRetriever ES 摘要检索
    plugin_rewritten_hyde: bool = False       # 对应 VectorRewrittenRetriever HyDE 重写

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
