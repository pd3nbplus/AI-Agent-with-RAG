# src/core/models.py
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, UniqueConstraint, Index, BigInteger, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = 'user_profiles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(255), nullable=False, index=True)  # 会话ID
    user_key = Column(String(100), nullable=False)               # 键：name, preference
    user_value = Column(Text, nullable=False)                    # 值：IronMan, Pizza
    confidence_score = Column(Float, default=1.0)                # 置信度
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 定义唯一约束：同一个 session 下，key 不能重复
    __table_args__ = (
        UniqueConstraint('thread_id', 'user_key', name='uk_thread_key'),
        Index('idx_thread', 'thread_id'),
    )

    def __repr__(self):
        return f"<UserProfile(thread_id={self.thread_id}, key={self.user_key}, value={self.user_value})>"


class RagEvalSample(Base):
    __tablename__ = "rag_eval_samples"

    id = Column(String(64), primary_key=True)
    category = Column(String(64), nullable=False, default="general")
    difficulty = Column(String(16), nullable=False)
    query = Column(Text, nullable=False)
    ground_truth_context = Column(JSON, nullable=False)  # 标准 Schema: string[]
    ground_truth_answer = Column(Text, nullable=False)
    source_document = Column(String(255), nullable=True)
    model_name = Column(String(128), nullable=True)  # 记录生成该样本的模型名
    meta = Column("metadata", JSON, nullable=False, default=dict)  # 标准 Schema: object
    # 示例数据
    # {
    #     "id": "q_001",
    #     "category": "policy", 
    #     "difficulty": "easy",
    #     "query": "奇葩星球公司对于迟到超过 30 分钟的处罚是什么？",
    #     "ground_truth_context": [
    #         "员工手册第三章：考勤管理。迟到 30 分钟以内扣除当日餐补；迟到 30 分钟至 2 小时，扣除半日工资并通报批评；迟到 2 小时以上视为旷工。"
    #     ],
    #     "ground_truth_answer": "迟到 30 分钟至 2 小时，扣除半日工资并通报批评；迟到 2 小时以上视为旷工。",
    #     "source_document": "employee_handbook.pdf",
    #     "metadata": {
    #         "generated_by": "gpt-4o-mini",
    #         "generation_date": "2026-03-06"
    #     }
    # }

    # 运行时追踪字段（用于回溯生成来源）
    source_chunk_index = Column(Integer, nullable=False)
    source_backend = Column(String(32), nullable=False, default="milvus")
    created_at = Column(BigInteger, nullable=False)
    # 评测样本批次号：用于按批次筛选评估集（如 [1,2,3]）。
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
    # 在本次评估运行中的坏案例排名（1 开始）。
    bad_case_rank = Column(Integer, nullable=False)
    predicted_category = Column(String(128), nullable=False, default="")
    diagnosis_model = Column(String(128), nullable=False, default="")
    diagnosis = Column(Text, nullable=False)
    # 从“问题”到“诊断”的完整 markdown，用于前端直接渲染。
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
