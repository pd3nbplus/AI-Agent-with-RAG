# augmented/sinks.py
# 存储模块：将评估样本批量 upsert 到 PostgreSQL。
from typing import Any, Dict, List

from sqlalchemy.dialects.postgresql import insert

from src.core.models import RagEvalSample
from src.core.postgres_client import get_postgres_client


class PostgresSink:
    def __init__(self) -> None:
        self.client = get_postgres_client()

    def save(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        # 仅做字段归一化，数据库写入采用单次 upsert。
        # 插入载荷：统一按 canonical 字段入库。
        payload: List[Dict[str, Any]] = []
        for row in rows:
            payload.append(
                {
                    "id": row["id"],
                    "category": row.get("category", "general"),
                    "difficulty": row["difficulty"],
                    "question": row["question"],
                    "ground_truth_contexts": row["ground_truth_contexts"],
                    "ground_truth": row["ground_truth"],
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
        # 冲突更新载荷：这是 PostgreSQL upsert 语义所需，不是别名映射。
        upsert_stmt = upsert_stmt.on_conflict_do_update(
            index_elements=[RagEvalSample.id],
            set_={
                "category": upsert_stmt.excluded.category,
                "difficulty": upsert_stmt.excluded.difficulty,
                "question": upsert_stmt.excluded.question,
                "ground_truth_contexts": upsert_stmt.excluded.ground_truth_contexts,
                "ground_truth": upsert_stmt.excluded.ground_truth,
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
