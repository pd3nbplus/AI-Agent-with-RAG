-- PostgreSQL: 评估结果/诊断结果建表与样本批次字段补齐
-- 执行方式示例：
-- psql "postgresql://agent_user:***@127.0.0.1:5433/agent_dev" -f sql/postgres_eval_results_and_diagnosis.sql

BEGIN;

-- 1) 给原始评测样本表补 batch_id 字段（按批次管理测试集）
ALTER TABLE IF EXISTS rag_eval_samples
    ADD COLUMN IF NOT EXISTS batch_id INTEGER NOT NULL DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_rag_eval_samples_batch_id
    ON rag_eval_samples(batch_id);

-- 将当前历史样本统一归为 batch_id=1
UPDATE rag_eval_samples
SET batch_id = 1
WHERE batch_id IS DISTINCT FROM 1;


-- 2) 评估结果表：保存每次评估的逐样本指标与问答内容
CREATE TABLE IF NOT EXISTS rag_eval_results (
    id BIGSERIAL PRIMARY KEY,
    eval_run_id VARCHAR(64) NOT NULL,
    sample_id VARCHAR(64) NOT NULL,
    sample_batch_id INTEGER NOT NULL DEFAULT 1,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    ground_truth TEXT NOT NULL,
    ground_truth_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    faithfulness DOUBLE PRECISION NOT NULL DEFAULT 0,
    answer_relevancy DOUBLE PRECISION NOT NULL DEFAULT 0,
    context_precision DOUBLE PRECISION NOT NULL DEFAULT 0,
    context_recall DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_eval_results_eval_run_id
    ON rag_eval_results(eval_run_id);

CREATE INDEX IF NOT EXISTS idx_rag_eval_results_batch
    ON rag_eval_results(sample_batch_id);

CREATE INDEX IF NOT EXISTS idx_rag_eval_results_sample_id
    ON rag_eval_results(sample_id);


-- 3) 诊断结果表：保存坏案例排名与完整 markdown 诊断内容
CREATE TABLE IF NOT EXISTS rag_eval_diagnoses (
    id BIGSERIAL PRIMARY KEY,
    eval_run_id VARCHAR(64) NOT NULL,
    sample_id VARCHAR(64) NOT NULL,
    sample_batch_id INTEGER NOT NULL DEFAULT 1,
    bad_case_rank INTEGER NOT NULL,
    predicted_category VARCHAR(128) NOT NULL DEFAULT '',
    diagnosis_model VARCHAR(128) NOT NULL DEFAULT '',
    diagnosis TEXT NOT NULL,
    diagnosis_markdown TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    ground_truth TEXT NOT NULL,
    ground_truth_contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
    faithfulness DOUBLE PRECISION NOT NULL DEFAULT 0,
    answer_relevancy DOUBLE PRECISION NOT NULL DEFAULT 0,
    context_precision DOUBLE PRECISION NOT NULL DEFAULT 0,
    context_recall DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_eval_diagnoses_eval_run_id
    ON rag_eval_diagnoses(eval_run_id);

CREATE INDEX IF NOT EXISTS idx_rag_eval_diagnoses_batch
    ON rag_eval_diagnoses(sample_batch_id);

CREATE INDEX IF NOT EXISTS idx_rag_eval_diagnoses_bad_case_rank
    ON rag_eval_diagnoses(bad_case_rank);

CREATE INDEX IF NOT EXISTS idx_rag_eval_diagnoses_sample_id
    ON rag_eval_diagnoses(sample_id);

COMMIT;
