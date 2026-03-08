from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from ragas.embeddings.base import BaseRagasEmbedding
from sentence_transformers import SentenceTransformer

from src.core.config import settings

# Global singleton cache: one SentenceTransformer instance per model name.
_st_model_cache: Dict[str, SentenceTransformer] = {}
# Global singleton cache: one ragas embedding adapter per model name.
_ragas_embedding_cache: Dict[str, "SharedSentenceTransformerEmbedding"] = {}


class SharedSentenceTransformerEmbedding(BaseRagasEmbedding):
    """Ragas embedding adapter backed by a shared SentenceTransformer instance."""

    def __init__(self, sentence_transformer_model: SentenceTransformer):
        super().__init__()
        self._model = sentence_transformer_model

    def embed_text(self, text: str, **kwargs) -> List[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    async def aembed_text(self, text: str, **kwargs) -> List[float]:
        return await asyncio.to_thread(self.embed_text, text, **kwargs)

    # Compatibility methods for ragas/lc-style embedding calls.
    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    async def aembed_query(self, text: str) -> List[float]:
        return await self.aembed_text(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.gather(*[self.aembed_text(t) for t in texts])


def get_sentence_transformer(model_name: Optional[str] = None) -> SentenceTransformer:
    model_name = model_name or settings.embedding.model_name
    if model_name not in _st_model_cache:
        _st_model_cache[model_name] = SentenceTransformer(model_name)
    return _st_model_cache[model_name]


def get_ragas_shared_embedding(model_name: Optional[str] = None) -> SharedSentenceTransformerEmbedding:
    model_name = model_name or settings.embedding.model_name
    if model_name not in _ragas_embedding_cache:
        _ragas_embedding_cache[model_name] = SharedSentenceTransformerEmbedding(
            get_sentence_transformer(model_name)
        )
    return _ragas_embedding_cache[model_name]

