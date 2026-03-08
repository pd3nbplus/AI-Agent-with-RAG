from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, Float, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(255), nullable=False, index=True)
    user_key = Column(String(100), nullable=False)
    user_value = Column(Text, nullable=False)
    confidence_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("thread_id", "user_key", name="uk_thread_key"),
        Index("idx_thread", "thread_id"),
    )

    def __repr__(self):
        return f"<UserProfile(thread_id={self.thread_id}, key={self.user_key}, value={self.user_value})>"


class RagEvalSample(Base):
    __tablename__ = "rag_eval_samples"

    id = Column(String(64), primary_key=True)
    category = Column(String(64), nullable=False, default="general")
    difficulty = Column(String(16), nullable=False)

    # Legacy columns.
    query = Column(Text, nullable=True)
    ground_truth_context = Column(JSON, nullable=True)
    ground_truth_answer = Column(Text, nullable=True)

    # Canonical columns.
    question = Column(Text, nullable=True)
    ground_truth_contexts = Column(JSON, nullable=True)
    ground_truth = Column(Text, nullable=True)

    source_document = Column(String(255), nullable=True)
    model_name = Column(String(128), nullable=True)
    meta = Column("metadata", JSON, nullable=False, default=dict)

    source_chunk_index = Column(Integer, nullable=False)
    source_backend = Column(String(32), nullable=False, default="milvus")
    created_at = Column(BigInteger, nullable=False)
    batch_id = Column(Integer, nullable=False, default=1)

    __table_args__ = (
        Index("idx_rag_eval_samples_created_at", "created_at"),
        Index("idx_rag_eval_samples_difficulty", "difficulty"),
        Index("idx_rag_eval_samples_category", "category"),
        Index("idx_rag_eval_samples_batch_id", "batch_id"),
    )


class RagEvalResult(Base):
    __tablename__ = "rag_eval_results"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    # 一次评估运行的唯一标识，便于串联结果与诊断。
    eval_run_id = Column(String(64), nullable=False)
    sample_id = Column(String(64), nullable=False, index=True)
    sample_batch_id = Column(Integer, nullable=False, default=1)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    contexts = Column(JSON, nullable=False, default=list)
    ground_truth = Column(Text, nullable=False)
    ground_truth_contexts = Column(JSON, nullable=False, default=list)
    faithfulness = Column(Float, nullable=False, default=0.0)
    answer_relevancy = Column(Float, nullable=False, default=0.0)
    context_precision = Column(Float, nullable=False, default=0.0)
    context_recall = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_rag_eval_results_eval_run_id", "eval_run_id"),
        Index("idx_rag_eval_results_batch", "sample_batch_id"),
    )


class RagEvalDiagnosis(Base):
    __tablename__ = "rag_eval_diagnoses"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    eval_run_id = Column(String(64), nullable=False, index=True)
    sample_id = Column(String(64), nullable=False, index=True)
    sample_batch_id = Column(Integer, nullable=False, default=1)
    bad_case_rank = Column(Integer, nullable=False)
    predicted_category = Column(String(128), nullable=False, default="")
    diagnosis_model = Column(String(128), nullable=False, default="")
    diagnosis = Column(Text, nullable=False)
    diagnosis_markdown = Column(Text, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    contexts = Column(JSON, nullable=False, default=list)
    ground_truth = Column(Text, nullable=False)
    ground_truth_contexts = Column(JSON, nullable=False, default=list)
    faithfulness = Column(Float, nullable=False, default=0.0)
    answer_relevancy = Column(Float, nullable=False, default=0.0)
    context_precision = Column(Float, nullable=False, default=0.0)
    context_recall = Column(Float, nullable=False, default=0.0)
    avg_score = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_rag_eval_diagnoses_eval_run_id", "eval_run_id"),
        Index("idx_rag_eval_diagnoses_batch", "sample_batch_id"),
        Index("idx_rag_eval_diagnoses_bad_case_rank", "bad_case_rank"),
    )
