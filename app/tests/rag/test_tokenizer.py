from app.core import logger
from app.rag.llm.tokenizers import TokenCounter, TokenizerType


def test_tokenizer() -> None:
    text = "你好，世界！Hello World!"
    # OpenAI 模型
    counter = TokenCounter(TokenizerType.TIKTOKEN, "gpt-3.5-turbo")
    token_count = counter.count_tokens(text)
    assert token_count > 0, "OpenAI tokenizer 测试失败"
    estimated_count = counter.estimate_tokens(text)
    assert estimated_count > 0, "Estimate tokenizer 快速估算测试失败"
    logger.info(f"estimated_count: {estimated_count}")

    # HuggingFace 远程加载
    # 通义千问模型
    qwen_counter = TokenCounter(TokenizerType.HUGGINGFACE, "Qwen/Qwen-7B")
    qwen_token_count = qwen_counter.count_tokens(text)
    assert qwen_token_count > 0, "Qwen tokenizer 测试失败"

    # DeepSeek 模型
    deepseek_counter = TokenCounter(TokenizerType.HUGGINGFACE, "deepseek-ai/DeepSeek-R1")
    deepseek_token_count = deepseek_counter.count_tokens(text)
    assert deepseek_token_count > 0, "DeepSeek tokenizer 测试失败"
