from enum import StrEnum

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core import settings


class ModelAPIType(StrEnum):
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"


class LLMParams(BaseModel):
    prompt: str
    temperature: float = 0.5
    max_tokens: int | None = None
    timeout: int | None = None
    max_retries: int = 3
    streaming: bool = True
    stream_usage: bool = True
    api_type: ModelAPIType = ModelAPIType.OPENAI
    model_name: str = settings.CHAT_MODEL
    api_key: str | None = settings.OPENAI_API_KEY
    base_url: str | None = settings.OPENAI_API_BASE


class LLM:
    def __init__(self) -> None:
        pass

    def generate(self, prompt: LanguageModelInput, params: LLMParams) -> AIMessage:
        match params.api_type:
            case ModelAPIType.OPENAI:
                llm = ChatOpenAI(
                    model=params.model_name,
                    api_key=params.api_key,
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

        return llm.invoke(prompt)
