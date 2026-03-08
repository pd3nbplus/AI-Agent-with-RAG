# src/core/es_client.py
from elasticsearch import Elasticsearch, helpers
from src.core.config import settings
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ESClientConfig:
    host: str = settings.db.es_host
    index_questions: str = settings.db.es_index_questions
    index_summaries: str = settings.db.es_index_summaries
    user: Optional[str] = settings.db.es_user
    password: Optional[str] = settings.db.es_password
    request_timeout: int = 30

    @classmethod
    def from_any(cls, value: Optional["ESClientConfig | Dict[str, Any]"]) -> "ESClientConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            payload = {k: v for k, v in value.items() if k in allowed}
            return cls(**payload)
        raise TypeError("ES config must be None, ESClientConfig or dict")


class ESClient:
    def __init__(self, config: Optional[ESClientConfig | Dict[str, Any]] = None):
        cfg = ESClientConfig.from_any(config)
        self.host = cfg.host
        self.index_questions = cfg.index_questions
        self.index_summaries = cfg.index_summaries
        self.user = cfg.user
        self.password = cfg.password
        self.request_timeout = cfg.request_timeout
        self.client: Optional[Elasticsearch] = None
        self._connect()

    def _connect(self):
        """初始化 ES 连接"""
        try:
            hosts = [self.host]
            
            # 尝试连接
            self.client = Elasticsearch(
                hosts=hosts,
                basic_auth=(self.user, self.password) if self.user else None,
                request_timeout=self.request_timeout
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
                            "_index": self.index_questions,
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
                            "_index": self.index_summaries,
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

# 单例（默认配置）与按配置缓存实例
es_client_instance = ESClient()
es_client_instances: Dict[tuple, ESClient] = {}


def get_es_client(config: Optional[ESClientConfig | Dict[str, Any]] = None) -> ESClient:
    if config is None:
        return es_client_instance

    cfg = ESClientConfig.from_any(config)
    cache_key = (
        cfg.host,
        cfg.index_questions,
        cfg.index_summaries,
        cfg.user,
        cfg.password,
        cfg.request_timeout,
    )
    if cache_key not in es_client_instances:
        es_client_instances[cache_key] = ESClient(cfg)
    return es_client_instances[cache_key]
