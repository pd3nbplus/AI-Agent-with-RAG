from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from src.core.config import settings
from src.core.embedding_client import get_sentence_transformer

logger = logging.getLogger(__name__)

@dataclass
class MilvusClientConfig:
    host: str = settings.db.milvus_host
    port: str = settings.db.milvus_port
    collection_name: str = settings.db.milvus_collection
    metric_type: str = settings.db.milvus_metric_type
    index_type: str = settings.db.milvus_index_type
    index_m: int = settings.db.milvus_index_m
    index_ef_construction: int = settings.db.milvus_index_ef_construction
    search_ef: int = settings.db.milvus_search_ef
    embedding_model_name: str = settings.embedding.model_name

    @classmethod
    def from_any(cls, value: Optional["MilvusClientConfig | Dict[str, Any]"]) -> "MilvusClientConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            payload = {k: v for k, v in value.items() if k in allowed}
            return cls(**payload)
        raise TypeError("Milvus config must be None, MilvusClientConfig or dict")


class MilvusClient:
    def __init__(self, config: Optional[MilvusClientConfig | Dict[str, Any]] = None):
        cfg = MilvusClientConfig.from_any(config)

        self.host = cfg.host
        self.port = cfg.port
        self.collection_name = cfg.collection_name
        self.metric_type = cfg.metric_type
        self.index_type = cfg.index_type
        self.index_m = cfg.index_m
        self.index_ef_construction = cfg.index_ef_construction
        self.search_ef = cfg.search_ef

        self.collection: Optional[Collection] = None

        self.model_name = cfg.embedding_model_name

        try:
            self.embedding_model = get_sentence_transformer(self.model_name)
            self.dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info("✅ 模型加载完成，向量维度：%s", self.dim)
        except Exception as e:
            logger.error("❌ 加载 Embedding 模型失败：%s", e)
            raise

        self._connect()
        self._init_collection()

    def _connect(self) -> None:
        try:
            connections.connect(host=self.host, port=self.port)
            logger.info("✅ 成功连接 Milvus (%s:%s)", self.host, self.port)
        except Exception as e:
            logger.error("❌ 连接 Milvus 失败：%s", e)
            raise

    def _init_collection(self) -> None:
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info("ℹ️ 集合 %s 已存在，已加载", self.collection_name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, "RAG Knowledge Base for AI Agent")

        self.collection = Collection(self.collection_name, schema)

        index_params = {
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "params": {"M": self.index_m, "efConstruction": self.index_ef_construction},
        }
        self.collection.create_index("vector", index_params)
        self.collection.load()
        logger.info("✅ 集合 %s 创建成功，索引类型：%s", self.collection_name, self.index_type)

    def embed_text(self, text: str) -> List[float]:
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()

    def insert_data(self, id: str, text: str, metadata: Optional[dict] = None) -> None:
        if self.collection is None:
            raise RuntimeError("Milvus collection is not initialized")

        vector = self.embed_text(text)
        data = [{"id": id, "vector": vector, "text": text, "metadata": metadata or {}}]
        self.collection.insert(data)
        logger.debug("📦 插入数据：%s", id)

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[dict]:
        if self.collection is None:
            raise RuntimeError("Milvus collection is not initialized")

        query_vector = self.embed_text(query)
        search_params = {"metric_type": self.metric_type, "params": {"ef": self.search_ef}}

        if not output_fields:
            output_fields = ["text", "metadata"]

        try:
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields,
            )
        except Exception as e:
            logger.error("❌ Milvus 搜索失败 (Expr: %s): %s", filter_expr, e)
            return []

        hits: List[dict] = []
        for hit in results[0]:
            hits.append(
                {
                    "text": hit.entity.get("text"),
                    "score": hit.score,
                    "metadata": hit.entity.get("metadata"),
                }
            )
        return hits

    def drop_collection(self) -> None:
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.warning("🗑️ 集合 %s 已删除", self.collection_name)

    def scan_collection(self, limit: int = 500, offset: int = 0) -> List[Dict]:
        if self.collection is None:
            raise RuntimeError("Milvus collection is not initialized")

        try:
            results = self.collection.query(
                expr="",
                output_fields=["text", "metadata"],
                limit=limit,
                offset=offset,
            )
            return results
        except Exception as e:
            logger.error("❌ Milvus 扫描失败：%s", e)
            return []


milvus_client_instance = None
milvus_client_instances: Dict[tuple, MilvusClient] = {}


def get_milvus_client(config: Optional[MilvusClientConfig | Dict[str, Any]] = None) -> MilvusClient:
    global milvus_client_instance

    if config is None:
        if milvus_client_instance is None:
            milvus_client_instance = MilvusClient()
        return milvus_client_instance

    cfg = MilvusClientConfig.from_any(config)
    cache_key = (
        cfg.host,
        cfg.port,
        cfg.collection_name,
        cfg.embedding_model_name,
        cfg.metric_type,
        cfg.index_type,
        cfg.index_m,
        cfg.index_ef_construction,
        cfg.search_ef,
    )
    if cache_key not in milvus_client_instances:
        milvus_client_instances[cache_key] = MilvusClient(cfg)
    return milvus_client_instances[cache_key]


if __name__ == "__main__":
    client = get_milvus_client()
