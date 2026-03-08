# augmented/sources.py
# 数据源模块：当前仅从 Milvus 扫描 chunk 作为评估题生成输入。
from typing import Any, Dict, List

from src.core.milvus_client import get_milvus_client


class MilvusSource:
    def __init__(self) -> None:
        self.client = get_milvus_client()

    def load_chunks(self, limit: int) -> List[Dict[str, Any]]:
        # 使用 offset 分页拉取，直到达到 limit 或无更多数据。
        offset = 0
        batch_size = min(limit, 200)
        chunks: List[Dict[str, Any]] = []
        while len(chunks) < limit:
            rows = self.client.scan_collection(limit=batch_size, offset=offset)
            if not rows:
                break
            for row in rows:
                text = row.get("text", "")
                metadata = row.get("metadata", {}) or {}
                if text:
                    chunks.append({"text": text, "metadata": metadata})
                    if len(chunks) >= limit:
                        break
            if len(rows) < batch_size:
                break
            offset += batch_size
        return chunks
