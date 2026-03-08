"""
将评估 CSV 导入 PostgreSQL 的 rag_eval_results 表。

用法：
python -m src.test.import_eval_csv_to_pg
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, List

import pandas as pd

from src.core.models import RagEvalResult
from src.core.postgres_client import get_postgres_client


CONFIG = {
    # 评估 CSV 路径（run_evaluation 旧流程输出的结果文件）
    "csv_path": "C:\\Users\\pdnbplus\\Documents\\python全系列\\AIAgent开发\\data\\eval\\recursive_result.csv",
    # 本次导入写入到哪个 eval_run_id
    "eval_run_id": f"import_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
    # 写入 sample_batch_id（当 CSV 内没有该字段时使用）
    "sample_batch_id": 1,
    # 若该 eval_run_id 已存在数据，是否先删除再重导
    "replace_if_exists": True,
}


def _to_text_list(value: Any) -> List[str]:
    """兼容 list、字符串化 list、普通字符串。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [s]
    return [str(value)]


def _pick(row: pd.Series, *keys: str, default=None):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default


def main():
    csv_path = Path(CONFIG["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV 为空，无需导入。")
        return

    eval_run_id = str(CONFIG["eval_run_id"])
    default_batch = int(CONFIG["sample_batch_id"])

    rows = []
    now = datetime.now(UTC)
    for i, row in df.iterrows():
        sample_id = str(_pick(row, "sample_id", default=f"csv_{default_batch}_{i + 1}"))
        sample_batch_id = int(_pick(row, "sample_batch_id", "batch_id", default=default_batch) or default_batch)

        rows.append(
            {
                "eval_run_id": eval_run_id,
                "sample_id": sample_id,
                "sample_batch_id": sample_batch_id,
                "question": str(_pick(row, "question", "user_input", default="") or ""),
                "answer": str(_pick(row, "answer", "response", default="") or ""),
                "contexts": _to_text_list(_pick(row, "contexts", "retrieved_contexts", default=[])),
                "ground_truth": str(_pick(row, "ground_truth", "reference", default="") or ""),
                "ground_truth_contexts": _to_text_list(
                    _pick(row, "ground_truth_contexts", "reference_contexts", default=[])
                ),
                "faithfulness": float(_pick(row, "faithfulness", default=0.0) or 0.0),
                "answer_relevancy": float(_pick(row, "answer_relevancy", default=0.0) or 0.0),
                "context_precision": float(_pick(row, "context_precision", default=0.0) or 0.0),
                "context_recall": float(_pick(row, "context_recall", default=0.0) or 0.0),
                "created_at": now,
            }
        )

    pg = get_postgres_client()
    RagEvalResult.__table__.create(bind=pg.engine, checkfirst=True)

    with pg.get_session() as session:
        if CONFIG.get("replace_if_exists", True):
            session.query(RagEvalResult).filter(RagEvalResult.eval_run_id == eval_run_id).delete()
        session.bulk_insert_mappings(RagEvalResult, rows)
        session.commit()

    print(f"导入完成: {len(rows)} 条 -> rag_eval_results")
    print(f"eval_run_id: {eval_run_id}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
