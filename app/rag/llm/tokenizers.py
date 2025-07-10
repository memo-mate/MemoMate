from enum import StrEnum

import tiktoken
from transformers import AutoTokenizer


class TokenizerType(StrEnum):
    TIKTOKEN = "tiktoken"
    TRANSFORMERS = "transformers"
    HUGGINGFACE = "huggingface"


class TokenCounter:
    def __init__(self, tokenizer_type: TokenizerType, model_name: str = "gpt-3.5-turbo"):
        self.tokenizer_type = tokenizer_type
        self.model_name = model_name
        self.tokenizer = self._init_tokenizer()

    def _init_tokenizer(self) -> tiktoken.Encoding | AutoTokenizer:
        match self.tokenizer_type:
            case TokenizerType.TIKTOKEN:
                return tiktoken.encoding_for_model(self.model_name)
            case TokenizerType.TRANSFORMERS:
                return AutoTokenizer.from_pretrained(self.model_name)
            case TokenizerType.HUGGINGFACE:
                return AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            case _:
                raise ValueError(f"不支持的tokenizer类型: {self.tokenizer_type}")

    def count_tokens(self, text: str) -> int:
        match self.tokenizer_type:
            case TokenizerType.TIKTOKEN:
                return len(self.tokenizer.encode(text))
            case TokenizerType.TRANSFORMERS:
                return len(self.tokenizer.encode(text))
            case TokenizerType.HUGGINGFACE:
                return len(self.tokenizer.encode(text))
            case _:
                raise ValueError(f"不支持的tokenizer类型: {self.tokenizer_type}")

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """简单估算 token 数量（适用于快速估算）"""
        # 英文约每4个字符1个token
        # 中文约每1.5个字符1个token
        en_chars = sum(1 for c in text if ord(c) < 128)
        zh_chars = len(text) - en_chars
        return int(en_chars / 4 + zh_chars / 1.5)


if __name__ == "__main__":
    from rich.live import Live
    from rich.table import Table

    text = "你好，世界！Hello World!"

    table = Table(title="Token Count")
    table.add_column("Tokenizer Type", justify="center")
    table.add_column("Token Count", justify="center")

    table.add_row("Test Text", text)
    table.add_row("-" * len(text), "-" * len(text))

    with Live(table, refresh_per_second=10) as live:
        # print(f"test text: {text}")
        # OpenAI 模型
        counter = TokenCounter(TokenizerType.TIKTOKEN, "gpt-3.5-turbo")
        token_count = counter.count_tokens(text)
        table.add_row("gpt-3.5-turbo", str(token_count))
        live.update(table)

        # print(f"token_count: {token_count}")

        # 使用通义千问模型计算token
        qwen_counter = TokenCounter(TokenizerType.HUGGINGFACE, "Qwen/Qwen-7B")
        token_count = qwen_counter.count_tokens(text)
        table.add_row("Qwen/Qwen-7B", str(token_count))
        live.update(table)
        # print(f"qwen_token_count: {token_count}")

        # 使用DeepSeek模型计算token
        deepseek_counter = TokenCounter(TokenizerType.HUGGINGFACE, "deepseek-ai/DeepSeek-R1")
        token_count = deepseek_counter.count_tokens(text)
        table.add_row("deepseek-ai/DeepSeek-R1", str(token_count))
        live.update(table)
        # print(f"deepseek_token_count: {token_count}")

        # 快速估算
        estimated_count = counter.estimate_tokens(text)
        table.add_row("Estimated", str(estimated_count))
        live.update(table)
        # print(f"estimated_count: {estimated_count}")
