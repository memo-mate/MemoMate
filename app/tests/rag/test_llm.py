from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from app.rag.llm.completions import LLM, LLMParams, ModelAPIType, RAGLLMPrompt


def test_llm_chat() -> None:
    prompt = RAGLLMPrompt(context="", question="你好")
    params = LLMParams(
        api_type=ModelAPIType.OPENAI,
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        streaming=False,
        stream_usage=False,
    )

    llm = LLM().generate(prompt, params)
    response: BaseMessage = llm.invoke(
        prompt.model_dump(exclude={"prompt"}),
        config=RunnableConfig(callbacks=[]),
    )
    print(f"content: {response.content}")
    print(f"usage_metadata: {getattr(response, 'usage_metadata', None)}")
