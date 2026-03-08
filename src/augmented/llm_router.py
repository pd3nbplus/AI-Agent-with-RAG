# augmented/llm_router.py
# LLM 路由模块：从 JSON 加载多端点并按顺序降级调用。
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.augmented.config import GeneratorConfig
from src.utils.xml_parser import remove_think_and_n

logger = logging.getLogger(__name__)


@dataclass
class LLMEndpoint:
    url: str
    model: str
    api_key: str
    temperature: float = 0.7


class LLMRouter:
    def __init__(self, config: GeneratorConfig, llm_group: Optional[str] = None) -> None:
        self.config = config
        self.llm_group = llm_group
        self.endpoints = self._load_endpoints(config.llm_json_path, llm_group=llm_group)
        # 实例级降级游标：第 k 个失败后，下次从 k+1 开始。
        self._degrade_start_idx = 0

    def _load_endpoints(self, json_path: str, llm_group: Optional[str] = None) -> List[LLMEndpoint]:
        # 支持形态：
        # 1) {"generator_llms": [...], "analyst_llms": [...], "llms": [...]} 
        # 2) {"llms": [...]} 
        # 3) [...]
        # Use utf-8-sig to be compatible with JSON files saved with BOM.
        with open(json_path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            if llm_group and isinstance(payload.get(llm_group), list):
                records = payload[llm_group]
            elif isinstance(payload.get("generator_llms"), list):
                records = payload["generator_llms"]
            else:
                records = payload.get("llms", payload)
        else:
            records = payload

        if not isinstance(records, list) or not records:
            raise ValueError(f"LLM JSON 配置无效或为空: {json_path}, group={llm_group}")

        endpoints: List[LLMEndpoint] = []
        for item in records:
            endpoints.append(
                LLMEndpoint(
                    url=item["url"],
                    model=item["model"],
                    api_key=item.get("api_key", "not-needed"),
                    temperature=float(item.get("temperature", 0.7)),
                )
            )
        return endpoints

    def invoke(self, prompt: ChatPromptTemplate, payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
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
                res = chain.invoke(payload)
                text = remove_think_and_n(getattr(res, "content", "") or "")
                if text:
                    return text, ep.model
            except Exception as e:
                last_error = e
                self._degrade_start_idx = max(self._degrade_start_idx, idx + 1)
                logger.warning("LLM 调用失败，降级到下一 endpoint。model=%s err=%s", ep.model, e)
                continue

        raise RuntimeError(f"所有 LLM endpoint 调用失败: {last_error}")
