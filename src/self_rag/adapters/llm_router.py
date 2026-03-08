from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.utils.xml_parser import remove_think_and_n

logger = logging.getLogger(__name__)


@dataclass
class LLMEndpoint:
    url: str
    model: str
    api_key: str
    temperature: float = 0.2


class LLMRouter:
    """Self-RAG 专用 LLM 路由器：从 JSON 读取多个 endpoint 并按顺序降级。"""

    def __init__(self, llm_json_path: str, llm_group: Optional[str] = None) -> None:
        self.llm_json_path = llm_json_path
        self.llm_group = llm_group
        self.endpoints = self._load_endpoints(llm_json_path=llm_json_path, llm_group=llm_group)
        # 实例级降级游标：前面的 endpoint 连续失败后，下次直接从后面的开始。
        self._degrade_start_idx = 0

    @staticmethod
    def _pick_records(payload: Any, llm_group: Optional[str]) -> Any:
        # 支持三种形态：
        # 1) {"generator_llms": [...], "analyst_llms": [...], "llms": [...]}
        # 2) {"llms": [...]}
        # 3) [...]
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return None
        if llm_group and isinstance(payload.get(llm_group), list):
            return payload.get(llm_group)
        if isinstance(payload.get("llms"), list):
            return payload.get("llms")
        return None

    @staticmethod
    def _validate_record(item: Dict[str, Any], idx: int, json_path: str) -> None:
        missing = [k for k in ("url", "model") if not item.get(k)]
        if missing:
            raise ValueError(f"LLM 配置缺失字段 {missing} at index={idx}, file={json_path}")

    def _load_endpoints(self, llm_json_path: str, llm_group: Optional[str]) -> List[LLMEndpoint]:
        with open(llm_json_path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)

        records = self._pick_records(payload, llm_group=llm_group)
        if not isinstance(records, list) or not records:
            raise ValueError(f"LLM JSON 配置无效或为空: {llm_json_path}, group={llm_group}")

        endpoints: List[LLMEndpoint] = []
        for idx, item in enumerate(records):
            if not isinstance(item, dict):
                raise ValueError(f"LLM 配置项必须是对象: index={idx}, file={llm_json_path}")
            self._validate_record(item, idx, llm_json_path)
            endpoints.append(
                LLMEndpoint(
                    url=item["url"],
                    model=item["model"],
                    api_key=item.get("api_key", "not-needed"),
                    temperature=float(item.get("temperature", 0.2)),
                )
            )
        return endpoints

    async def ainvoke(self, prompt: ChatPromptTemplate, payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        last_error: Optional[Exception] = None
        for idx in range(self._degrade_start_idx, len(self.endpoints)):
            ep = self.endpoints[idx]
            try:
                llm = ChatOpenAI(
                    model=ep.model,
                    base_url=ep.url,
                    api_key=ep.api_key if ep.api_key else "not-needed",
                    temperature=ep.temperature,
                )
                chain = prompt | llm
                res = await chain.ainvoke(payload)
                text = remove_think_and_n(getattr(res, "content", "") or "")
                if text:
                    return text, ep.model
            except Exception as e:
                last_error = e
                self._degrade_start_idx = max(self._degrade_start_idx, idx + 1)
                logger.warning("self-rag judge llm 调用失败，降级到下一 endpoint。model=%s err=%s", ep.model, e)
                continue

        raise RuntimeError(f"所有 LLM endpoint 调用失败: {last_error}")
