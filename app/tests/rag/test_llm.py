from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig

from app.rag.llm.completions import LLM, LLMParams, ModelAPIType, RAGLLMPrompt


def test_llm_chat() -> None:
    prompt = RAGLLMPrompt(context="", question="你好")
    params = LLMParams(
        api_type=ModelAPIType.OPENAI,
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        streaming=False,
        stream_usage=False,
    )

    llm: Runnable[dict[str, Any], AIMessage] = LLM().generate(
        prompt,
        params,
    )
    response: AIMessage = llm.invoke(
        prompt.model_dump(exclude={"prompt"}),
        config=RunnableConfig(callbacks=[]),
    )
    print(response.content)
    print(response.usage_metadata)
