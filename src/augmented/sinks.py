# augmented/sinks.py
# 存储模块：将评估样本批量 upsert 到 PostgreSQL。
from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from src.core.postgres_client import get_postgres_client
from src.core.models import RagEvalSample

class PostgresSink:
    def __init__(self) -> None:
        self.client = get_postgres_client()
        # 启动时做一次连通性与表结构检查，避免运行中才暴露配置问题。
        self._test_connection()
        self._ensure_table()

    def _test_connection(self) -> None:
        with self.client.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def _ensure_table(self) -> None:
        RagEvalSample.__table__.create(bind=self.client.engine, checkfirst=True)
        # 兼容已存在旧表：补齐新增字段（不会覆盖历史数据）。
        with self.client.engine.begin() as conn:
            conn.execute(text("ALTER TABLE rag_eval_samples ADD COLUMN IF NOT EXISTS model_name VARCHAR(128)"))
            conn.execute(text("ALTER TABLE rag_eval_samples ADD COLUMN IF NOT EXISTS batch_id INTEGER DEFAULT 1"))

    def save(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        # 仅做字段归一化，数据库写入采用单次 upsert。
        payload: List[Dict[str, Any]] = []
        for row in rows:
            payload.append(
                {
                    "id": row["id"],
                    "category": row.get("category", "general"),
                    "difficulty": row["difficulty"],
                    "query": row["query"],
                    "ground_truth_context": row["ground_truth_context"],
                    "ground_truth_answer": row["ground_truth_answer"],
                    "source_document": row.get("source_document"),
                    "model_name": row.get("model_name"),
                    "metadata": row.get("metadata", {}),
                    "source_chunk_index": row["source_chunk_index"],
                    "source_backend": row.get("source_backend", "milvus"),
                    "created_at": row["created_at"],
                    "batch_id": int(row.get("batch_id", 1)),
                }
            )

        upsert_stmt = insert(RagEvalSample.__table__).values(payload)
        upsert_stmt = upsert_stmt.on_conflict_do_update(
            index_elements=[RagEvalSample.id],
            set_={
                "category": upsert_stmt.excluded.category,
                "difficulty": upsert_stmt.excluded.difficulty,
                "query": upsert_stmt.excluded.query,
                "ground_truth_context": upsert_stmt.excluded.ground_truth_context,
                "ground_truth_answer": upsert_stmt.excluded.ground_truth_answer,
                "source_document": upsert_stmt.excluded.source_document,
                "model_name": upsert_stmt.excluded.model_name,
                "metadata": upsert_stmt.excluded.metadata,
                "source_chunk_index": upsert_stmt.excluded.source_chunk_index,
                "source_backend": upsert_stmt.excluded.source_backend,
                "created_at": upsert_stmt.excluded.created_at,
                "batch_id": upsert_stmt.excluded.batch_id,
            },
        )

        with self.client.get_session() as session:
            session.execute(upsert_stmt)
            session.commit()
