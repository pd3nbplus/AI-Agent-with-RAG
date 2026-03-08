from __future__ import annotations

import json
import re
from typing import Any, Dict, Type, TypeVar

from langchain_core.prompts import ChatPromptTemplate

from src.self_rag.adapters.llm_router import LLMRouter

TModel = TypeVar("TModel")


class JudgeLLMAdapter:
    """Self-RAG 评判专用 LLM 适配器。

    说明：
    - 仅用于 judge_* 三个节点，走外部大模型 endpoint；
    - 生成/改写仍由主链路 llm_adapter 负责；
    - 支持 endpoint 降级，降低本地单模型拥塞影响。
    """

    def __init__(self, llm_json_path: str, llm_group: str = "analyst_llms"):
        self.router = LLMRouter(llm_json_path=llm_json_path, llm_group=llm_group)

    @staticmethod
    def _build_format_instructions() -> str:
        return (
            '仅输出一个 JSON 对象，包含字段：'
            '"score"(0~1浮点数), "reasoning"(字符串), "passed"(布尔值，可省略)。'
            "不要输出 markdown 代码块。"
        )

    @staticmethod
    def _extract_json_text(text: str) -> Dict[str, Any]:
        raw = text.strip()
        try:
            return json.loads(raw)
        except Exception:
            # 兼容模型输出前后带说明文本的情况，提取首个 JSON 对象。
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise ValueError(f"judge llm 返回非 JSON：{raw}")
            return json.loads(match.group(0))

    async def generate_structured(
        self,
        template: str,
        payload: Dict[str, Any],
        output_model: Type[TModel],
    ) -> TModel:
        prompt = ChatPromptTemplate.from_template(template)
        final_payload = {**payload, "format_instructions": self._build_format_instructions()}
        text, _model = await self.router.ainvoke(prompt=prompt, payload=final_payload)
        data = self._extract_json_text(text)

        score = float(data.get("score", 0.0))
        reasoning = str(data.get("reasoning", "")).strip()
        passed = bool(data.get("passed", False))
        return output_model(score=score, reasoning=reasoning, passed=passed)
