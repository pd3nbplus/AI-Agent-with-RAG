"""
RAG evaluator (sync).
- Read generated eval samples from PostgreSQL.
- Reuse existing retrieval pipeline to produce answer + contexts.
- Run RAGAS metrics and return a pandas DataFrame.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from datasets import Dataset as HFDataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

from src.core.config import settings
from src.core.embedding_client import get_ragas_shared_embedding
from src.core.models import RagEvalResult, RagEvalSample
from src.core.postgres_client import get_postgres_client
from src.rag.pipeline import RetrievalPipeline
from src.utils.xml_parser import remove_think_and_n
from sqlalchemy import func

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluator aligned with current architecture."""

    def __init__(self, top_k: int = 3, composer_config: Optional[Dict[str, Any]] = None):
        self.top_k = top_k
        self.pg_client = get_postgres_client()

        # Independent retrieval pipeline for evaluation experiments.
        self.pipeline = RetrievalPipeline(composer_config=composer_config)

        # 关键节点1：初始化两套 LLM
        # - answer_llm: 生成最终回答（RAG 回答阶段）
        # - ragas_llm: 供 RAGAS 指标打分使用（评估阶段）
        self.answer_llm = self._build_llm_from_global_config()
        self.ragas_llm = self._build_ragas_llm()
        # 关键节点2：复用全局 embedding 单例，避免重复加载模型。
        self.ragas_embeddings = get_ragas_shared_embedding(
            self.pipeline.composer.milvus_client.model_name
        )

        self.answer_prompt = ChatPromptTemplate.from_template(
            """
你是一个严谨的问答助手。请只依据给定上下文回答问题。
如果上下文无法支持答案，请明确回答：根据提供的上下文无法回答。
问题：{question}

上下文：
{contexts}
"""
        )
        self._ensure_eval_tables()

    @staticmethod
    def _run_async(coro):
        """Run async pipeline in sync context."""
        try:
            asyncio.get_running_loop()
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        except RuntimeError:
            return asyncio.run(coro)

    def _build_llm_from_global_config(self) -> ChatOpenAI:
        logger.info("Evaluator uses global LLM: %s", settings.llm.model_name)
        return ChatOpenAI(
            base_url=settings.llm.base_url,
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            temperature=float(settings.llm.temperature),
        )

    def _build_ragas_llm(self):
        """
        Ragas collections metrics require InstructorLLM.
        Build it from the same global model endpoint.
        """
        client = OpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
        )
        return llm_factory(
            model=settings.llm.model_name,
            provider="openai",
            client=client,
        )

    def _ensure_eval_tables(self) -> None:
        """启动时检查评估结果表结构，避免运行中才暴露建表问题。"""
        RagEvalResult.__table__.create(bind=self.pg_client.engine, checkfirst=True)

    def load_samples_from_postgres(
        self, limit: int = 100, batch_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """按批次加载评测样本；batch_ids 为空时回退为全量随机抽样。"""
        with self.pg_client.get_session() as session:
            q = session.query(RagEvalSample)
            if batch_ids:
                q = q.filter(RagEvalSample.batch_id.in_(batch_ids))
            rows: List[RagEvalSample] = q.order_by(func.random()).limit(limit).all()

        samples: List[Dict[str, Any]] = []
        for r in rows:
            samples.append(
                {
                    "id": r.id,
                    "batch_id": int(r.batch_id or 1),
                    "query": r.query,
                    "ground_truth_answer": r.ground_truth_answer,
                    "ground_truth_context": r.ground_truth_context or [],
                }
            )
        logger.info(
            "Loaded %s eval samples from PostgreSQL (batch_ids=%s)",
            len(samples),
            batch_ids or "ALL",
        )
        return samples

    def run_pipeline_for_query(self, query: str) -> Dict[str, Any]:
        """Retrieve contexts then generate answer."""
        results = self._run_async(self.pipeline.run(query=query, top_k=self.top_k))
        contexts = [r.text for r in results] if results else []

        chain = self.answer_prompt | self.answer_llm
        resp = chain.invoke({"question": query, "contexts": "\n\n".join(contexts)})
        # 清理部分模型返回的 <think> 推理片段，避免污染评估与报告展示。
        answer = remove_think_and_n(getattr(resp, "content", "") or "")

        return {"contexts": contexts, "answer": answer}

    def evaluate_dataset(self, test_samples: List[Dict[str, Any]]) -> pd.DataFrame:
        """Run pipeline and compute RAGAS metrics."""
        if not test_samples:
            return pd.DataFrame()

        logger.info("Start evaluation, sample count=%s", len(test_samples))
        rows_for_ragas: List[Dict[str, Any]] = []

        # 关键节点3：逐条样本执行“检索+回答”，并转换到 ragas 所需字段。
        for s in test_samples:
            out = self.run_pipeline_for_query(s["query"])
            rows_for_ragas.append(
                {
                    # 样本追踪字段：用于写回评估结果表与诊断表。
                    "sample_id": s["id"],
                    "sample_batch_id": int(s.get("batch_id", 1)),
                    # New ragas schema keys.
                    "user_input": s["query"],
                    "response": out["answer"],
                    "retrieved_contexts": out["contexts"],
                    "reference": s["ground_truth_answer"],
                    "reference_contexts": s.get("ground_truth_context", []),
                    # Keep compatibility aliases for older tooling/exports.
                    "question": s["query"],
                    "answer": out["answer"],
                    "contexts": out["contexts"],
                    "ground_truth": s["ground_truth_answer"],
                }
            )

        hf_dataset = HFDataset.from_list(rows_for_ragas)

        # 关键节点4：实例化评估指标对象（不是函数），并注入 llm/embedding 依赖。
        metrics = [
            Faithfulness(llm=self.ragas_llm),
            AnswerRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings),
            ContextPrecision(llm=self.ragas_llm),
            ContextRecall(llm=self.ragas_llm),
        ]

        # 关键节点5：执行 ragas evaluate，返回结构化 DataFrame 结果。
        result = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            show_progress=True,
        )
        return result.to_pandas()

    def save_eval_results(self, df_results: pd.DataFrame, eval_run_id: str) -> int:
        """将本次评估明细写入 rag_eval_results。"""
        if df_results is None or df_results.empty:
            return 0

        rows: List[Dict[str, Any]] = []
        for _, row in df_results.iterrows():
            rows.append(
                {
                    "eval_run_id": eval_run_id,
                    "sample_id": str(row.get("sample_id", "")),
                    "sample_batch_id": int(row.get("sample_batch_id", 1) or 1),
                    "question": str(row.get("question", row.get("user_input", ""))),
                    "answer": str(row.get("answer", row.get("response", ""))),
                    "contexts": row.get("contexts", row.get("retrieved_contexts", [])) or [],
                    "ground_truth": str(row.get("ground_truth", row.get("reference", ""))),
                    "ground_truth_contexts": row.get(
                        "ground_truth_contexts", row.get("reference_contexts", [])
                    )
                    or [],
                    "faithfulness": float(row.get("faithfulness", 0.0) or 0.0),
                    "answer_relevancy": float(row.get("answer_relevancy", 0.0) or 0.0),
                    "context_precision": float(row.get("context_precision", 0.0) or 0.0),
                    "context_recall": float(row.get("context_recall", 0.0) or 0.0),
                    "created_at": datetime.now(UTC),
                }
            )

        with self.pg_client.get_session() as session:
            session.bulk_insert_mappings(RagEvalResult, rows)
            session.commit()
        return len(rows)

    def evaluate_from_postgres(
        self, limit: int = 100, batch_ids: Optional[List[int]] = None
    ) -> tuple[pd.DataFrame, str]:
        """按批次执行评估并返回结果及 eval_run_id。"""
        samples = self.load_samples_from_postgres(limit=limit, batch_ids=batch_ids)
        eval_run_id = f"eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        return self.evaluate_dataset(samples), eval_run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluator = RAGEvaluator(top_k=3)
    df, _ = evaluator.evaluate_from_postgres(limit=10)
    print(df.head())
