from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type, TypeVar

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.core.config import settings
from src.utils.xml_parser import remove_think_and_n

TModel = TypeVar("TModel", bound=BaseModel)


class LLMAdapter:
    """统一封装 LLM 调用与结构化解析。"""

    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=settings.llm.base_url,
            model=settings.llm.model_name,
            api_key=settings.llm.api_key,
            temperature=float(settings.llm.temperature),
        )

    @staticmethod
    def load_prompt(prompt_path: str | Path) -> str:
        path = Path(prompt_path)
        with path.open("r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def _clean_content(message: Any) -> str:
        content = message.content if hasattr(message, "content") else str(message)
        return remove_think_and_n(content)

    async def generate_text(self, template: str, payload: Dict[str, Any]) -> str:
        prompt = ChatPromptTemplate.from_template(template)
        clean_chain = RunnableLambda(self._clean_content)
        chain = prompt | self.llm | clean_chain
        return await chain.ainvoke(payload)

    async def generate_structured(
        self,
        template: str,
        payload: Dict[str, Any],
        output_model: Type[TModel],
    ) -> TModel:
        parser = PydanticOutputParser(pydantic_object=output_model)
        prompt = ChatPromptTemplate.from_template(template)
        clean_chain = RunnableLambda(self._clean_content)
        chain = prompt | self.llm | clean_chain | parser
        final_payload = {**payload, "format_instructions": parser.get_format_instructions()}
        return await chain.ainvoke(final_payload)
