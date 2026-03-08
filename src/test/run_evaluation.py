"""
运行评估脚本（固定配置版）
直接修改下方 CONFIG 字典后执行：
python -m src.test.run_evaluation
"""

import logging
from datetime import datetime
from pathlib import Path

from src.augmented.analyst import RAGAnalyst


CONFIG = {
    "limit": 15,
    "batch_ids": [1],  # 按批次评估
    # 可选：指定某次评估运行ID；不填时默认使用最近一次 eval_run_id
    "eval_run_id": None,
    "bad_case_top_k": 3,  # 坏案例诊断数量
    "save_markdown_report": True,
    "report_dir": "C:\\Users\\pdnbplus\\Documents\\python全系列\\AIAgent开发\\data\\eval",
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    analyst = RAGAnalyst()
    result = analyst.analyze(
        limit=CONFIG["limit"],
        batch_ids=CONFIG.get("batch_ids"),
        eval_run_id=CONFIG.get("eval_run_id"),
        bad_case_top_k=CONFIG["bad_case_top_k"],
    )

    eval_run_id = result["eval_run_id"]
    print(f"\n=== 评估运行 ID ===\n{eval_run_id}")
    print(f"评估结果入库条数: {result['eval_count']}")
    print(f"诊断结果入库条数: {result['diagnosis_count']}")

    if CONFIG.get("save_markdown_report", False):
        report_dir = Path(CONFIG["report_dir"])
        report_name = f"bad_cases_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        report_path = report_dir / report_name
        saved_path = analyst.to_markdown(str(report_path))
        print(f"报告已输出: {saved_path}")


if __name__ == "__main__":
    main()
