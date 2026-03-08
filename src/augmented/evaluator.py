"""
RAG evaluator (sync):
1) 从 PostgreSQL 读取评测样本
2) 通过 RetrievalPipeline 执行检索并生成回答
3) 调用 RAGAS 指标评估并返回 DataFrame
"""

from __future__ import annotations

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
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from sqlalchemy import func

from src.augmented.utils import deep_merge, run_async
from src.core.config import settings
from src.core.embedding_client import get_ragas_shared_embedding
from src.core.models import RagEvalResult, RagEvalSample
from src.core.postgres_client import get_postgres_client
from src.rag.pipeline import RetrievalPipeline
from src.schema.augmented_schema import EvalInputSample, EvalResultSample
from src.utils.xml_parser import remove_think_and_n

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """评测器：以可配置 pipeline 运行检索评估。"""

    def __init__(
        self,
        pipeline_config: Optional[Dict[str, Any]] = None,
    ):
        self.pg_client = get_postgres_client()
        self.pipeline = RetrievalPipeline()
        self.pipeline_config = self._build_pipeline_config(pipeline_config=pipeline_config)

        self.answer_llm = self._build_llm_from_global_config()
        self.ragas_llm = self._build_ragas_llm()
        self.ragas_embeddings = get_ragas_shared_embedding(settings.embedding.model_name)

        self.answer_prompt = ChatPromptTemplate.from_template(
            """
你是一个严谨的问答助手。请只依据给定上下文回答问题。
如果上下文无法支持答案，请明确回答：根据提供的上下文无法回答。
问题：{question}

上下文：
{contexts}
"""
        )

    @staticmethod
    def _build_pipeline_config(
        pipeline_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base_config: Dict[str, Any] = {
            "retrieval": {"top_k": 3, "rough_top_k": 8},
            "online": {
                "enable_rerank": settings.rag_online.enable_rerank,
                "dynamic_threshold": settings.rag_online.score_threshold,
            },
            "filter": {},
            "composer": {},
        }
        if pipeline_config:
            base_config = deep_merge(base_config, pipeline_config)
        return base_config

    def _build_llm_from_global_config(self) -> ChatOpenAI:
        logger.info("Evaluator uses global LLM: %s", settings.llm.model_name)
        return ChatOpenAI(
            base_url=settings.llm.base_url,
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            temperature=float(settings.llm.temperature),
        )

    @staticmethod
    def _build_ragas_llm():
        client = OpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
        )
        return llm_factory(
            model=settings.llm.model_name,
            provider="openai",
            client=client,
        )

    def load_samples_from_postgres(
        self, limit: int = 100, batch_ids: Optional[List[int]] = None
    ) -> List[EvalInputSample]:
        with self.pg_client.get_session() as session:
            query = session.query(RagEvalSample)
            if batch_ids:
                query = query.filter(RagEvalSample.batch_id.in_(batch_ids))
            rows: List[RagEvalSample] = query.order_by(func.random()).limit(limit).all()

        samples: List[EvalInputSample] = []
        for row in rows:
            samples.append(
                EvalInputSample(
                    sample_id=row.id,
                    sample_batch_id=int(row.batch_id or 1),
                    question=row.question,
                    ground_truth=row.ground_truth,
                    ground_truth_contexts=row.ground_truth_contexts or [],
                )
            )
        logger.info(
            "Loaded %s eval samples from PostgreSQL (batch_ids=%s)",
            len(samples),
            batch_ids or "ALL",
        )
        return samples

    def _retrieve_contexts(self, question: str) -> List[str]:
        results = run_async(self.pipeline.run(query=question, config=self.pipeline_config))
        return [item.text for item in results] if results else []

    def _answer_question(self, question: str, contexts: List[str]) -> str:
        chain = self.answer_prompt | self.answer_llm
        response = chain.invoke({"question": question, "contexts": "\n\n".join(contexts)})
        return remove_think_and_n(getattr(response, "content", "") or "")

    def run_pipeline_for_question(self, question: str) -> Dict[str, Any]:
        contexts = self._retrieve_contexts(question)
        answer = self._answer_question(question, contexts)
        return {"contexts": contexts, "answer": answer}

    @staticmethod
    def _to_ragas_row(sample: EvalInputSample, answer: str, contexts: List[str]) -> Dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "sample_batch_id": sample.sample_batch_id,
            "user_input": sample.question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": sample.ground_truth,
            "reference_contexts": sample.ground_truth_contexts,
            "question": sample.question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": sample.ground_truth,
            "ground_truth_contexts": sample.ground_truth_contexts,
        }

    def evaluate_dataset(self, test_samples: List[EvalInputSample]) -> pd.DataFrame:
        if not test_samples:
            return pd.DataFrame()

        logger.info("Start evaluation, sample count=%s", len(test_samples))
        rows_for_ragas: List[Dict[str, Any]] = []
        for sample in test_samples:
            output = self.run_pipeline_for_question(sample.question)
            rows_for_ragas.append(self._to_ragas_row(sample, output["answer"], output["contexts"]))

        ragas_dataset = HFDataset.from_list(rows_for_ragas)
        metrics = [
            Faithfulness(llm=self.ragas_llm),
            AnswerRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings),
            ContextPrecision(llm=self.ragas_llm),
            ContextRecall(llm=self.ragas_llm),
        ]
        result = evaluate(dataset=ragas_dataset, metrics=metrics, show_progress=True)
        return result.to_pandas()

    def save_eval_results(self, df_results: pd.DataFrame, eval_run_id: str) -> int:
        if df_results is None or df_results.empty:
            return 0

        rows: List[Dict[str, Any]] = []
        for _, row in df_results.iterrows():
            sample = EvalResultSample(
                sample_id=str(row["sample_id"]),
                sample_batch_id=int(row.get("sample_batch_id", 1) or 1),
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=row.get("contexts", []) or [],
                ground_truth=str(row["ground_truth"]),
                ground_truth_contexts=row.get("ground_truth_contexts", []) or [],
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            rows.append(
                {
                    "eval_run_id": eval_run_id,
                    **sample.model_dump(),
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
        samples = self.load_samples_from_postgres(limit=limit, batch_ids=batch_ids)
        eval_run_id = f"eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        return self.evaluate_dataset(samples), eval_run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluator = RAGEvaluator(pipeline_config={"retrieval": {"top_k": 3}})
    df, _ = evaluator.evaluate_from_postgres(limit=10)
    print(df.head())
