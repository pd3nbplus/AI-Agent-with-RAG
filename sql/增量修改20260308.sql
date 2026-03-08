-- PostgreSQL migration: unify rag_eval_samples schema to canonical field names.
-- Run:
-- psql "postgresql://agent_user:***@127.0.0.1:5433/agent_dev" -f sql/postgres_augmented_schema_unify.sql

BEGIN;

-- 1) Ensure columns used by current runtime code exist.
ALTER TABLE IF EXISTS rag_eval_samples
    ADD COLUMN IF NOT EXISTS model_name VARCHAR(128);

ALTER TABLE IF EXISTS rag_eval_samples
    ADD COLUMN IF NOT EXISTS batch_id INTEGER DEFAULT 1;

-- 2) Canonical columns.
ALTER TABLE IF EXISTS rag_eval_samples
    ADD COLUMN IF NOT EXISTS question TEXT;

ALTER TABLE IF EXISTS rag_eval_samples
    ADD COLUMN IF NOT EXISTS ground_truth TEXT;

ALTER TABLE IF EXISTS rag_eval_samples
    ADD COLUMN IF NOT EXISTS ground_truth_contexts JSONB;

-- 3) Backfill canonical columns from legacy columns.
UPDATE rag_eval_samples
SET
    question = COALESCE(question, query),
    ground_truth = COALESCE(ground_truth, ground_truth_answer),
    ground_truth_contexts = COALESCE(
        ground_truth_contexts,
        CASE
            WHEN ground_truth_context IS NULL THEN '[]'::jsonb
            ELSE ground_truth_context::jsonb
        END
    )
WHERE question IS NULL
   OR ground_truth IS NULL
   OR ground_truth_contexts IS NULL;

-- 4) Normalize batch_id.
UPDATE rag_eval_samples
SET batch_id = 1
WHERE batch_id IS NULL;

ALTER TABLE rag_eval_samples
    ALTER COLUMN batch_id SET DEFAULT 1;

COMMIT;

