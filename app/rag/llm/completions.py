from enum import StrEnum
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core import settings


class ModelAPIType(StrEnum):
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"


RESULT_PROMPT = ChatPromptTemplate.from_template(
    """回答以下问题，基于提供的上下文信息。如果无法从上下文中找到答案，请说"我不知道"。

上下文: {context}
问题: {question}

回答:"""
)


class LLMParams(BaseModel):
    temperature: float = 0
    max_tokens: int | None = None
    timeout: int | None = None
    max_retries: int = 3
    streaming: bool = True
    stream_usage: bool = True
    api_type: ModelAPIType = ModelAPIType.OPENAI
    model_name: str = settings.CHAT_MODEL
    api_key: str | None = settings.OPENAI_API_KEY
    base_url: str | None = settings.OPENAI_API_BASE


class RAGLLMPrompt(BaseModel):
    prompt: ChatPromptTemplate = RESULT_PROMPT
    context: str
    question: str


class LLM:
    def __init__(self) -> None:
        pass

    # mypy: disable-error-code="call-arg"
    def generate(self, prompt: RAGLLMPrompt, params: LLMParams) -> RunnableSerializable[dict[Any, Any], BaseMessage]:
        prompt_vars = prompt.prompt.input_variables
        if "context" not in prompt_vars or "question" not in prompt_vars:
            raise ValueError("Prompt must have context and question variables.")

        llm: BaseChatModel | None = None
        match params.api_type:
            case ModelAPIType.OPENAI:
                llm = ChatOpenAI(
                    model=params.model_name,
                    api_key=params.api_key,  # type: ignore[arg-type]
                    base_url=params.base_url,
                    temperature=params.temperature,
                    max_tokens=params.max_tokens,
                    timeout=params.timeout,
                    max_retries=params.max_retries,
                    streaming=params.streaming,
                    stream_usage=params.stream_usage,
                )
            case ModelAPIType.OLLAMA:
                llm = ChatOllama(
                    model=params.model_name,
                    base_url=params.base_url,
                    temperature=params.temperature,
                    max_tokens=params.max_tokens,
                    timeout=params.timeout,
                    max_retries=params.max_retries,
                    streaming=params.streaming,
                    stream_usage=params.stream_usage,
                )

            case _:
                raise ValueError("Unsupported api type.")

        chain = prompt.prompt | llm
        # chain.invoke(prompt, config=config)
        return chain
