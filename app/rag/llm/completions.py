from enum import StrEnum
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable, RunnableWithFallbacks
from langchain_core.tools import Tool
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
问题: {input}

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
    input: str


class LLM:
    def __init__(self) -> None:
        pass

    def _create_llm_instance(self, params: LLMParams) -> BaseChatModel:
        match params.api_type:
            case ModelAPIType.OPENAI:
                return ChatOpenAI(
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
                return ChatOllama(
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
                raise ValueError(f"Unsupported api type: {params.api_type}")

    # mypy: disable-error-code="call-arg"
    def generate(self, prompt: RAGLLMPrompt, params: LLMParams) -> RunnableSerializable[dict[Any, Any], BaseMessage]:
        prompt_vars = prompt.prompt.input_variables
        if "context" not in prompt_vars or "input" not in prompt_vars:
            raise ValueError("Prompt must have context and input variables.")

        llm = self._create_llm_instance(params)

        chain = prompt.prompt | llm
        # chain.invoke(prompt, config=config)
        return chain

    def get_llm(self, params: LLMParams) -> BaseChatModel:
        """获取原始的LLM对象，适用于需要直接使用LLM的场景"""
        return self._create_llm_instance(params)

    def _create_llm_fallback(
        self,
        model_names: list[str],
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
        timeout: int | None = None,
        tools: list[Tool] | None = None,
        exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,),
        exception_key: str | None = None,
    ) -> RunnableWithFallbacks:
        if not isinstance(model_names, list) or len(model_names) < 2:
            raise ValueError("model_names must be a list of at least 2 models")

        _model = ChatOpenAI(
            model=model_names[0],
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if tools:
            _model.bind_tools(tools)
        return _model.with_fallbacks([_model], exceptions_to_handle=exceptions_to_handle, exception_key=exception_key)

    def get_llm_fallback(
        self,
        model_names: list[str],
        api_key: str | None = settings.OPENAI_API_KEY,
        base_url: str | None = settings.OPENAI_API_BASE,
        temperature: float = 0,
        max_tokens: int | None = None,
        timeout: int | None = None,
        tools: list[Tool] | None = None,
        exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,),
        exception_key: str | None = None,
    ) -> RunnableWithFallbacks:
        """获取带有备选模型的LLM对象，适用于需要使用备选模型的场景"""
        return self._create_llm_fallback(
            model_names=model_names,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            tools=tools,
            exceptions_to_handle=exceptions_to_handle,
            exception_key=exception_key,
        )
