"""嵌入模型模块

使用 OpenAI 兼容 API，支持 vLLM、Ollama 等嵌入服务。

使用方式:
    from app.rag.embedding import get_embeddings

    # 自动从环境变量获取配置
    embeddings = get_embeddings()
    vectors = await embeddings.aembed_documents(["text1", "text2"])

配置示例（环境变量）:
    # vLLM 服务（默认端口 8000）
    EMBEDDING_API_BASE=http://localhost:8000/v1
    EMBEDDING_MODEL=BAAI/bge-m3
    EMBEDDING_API_KEY=dummy

    # Ollama 服务（默认端口 11434）
    EMBEDDING_API_BASE=http://localhost:11434/v1
    EMBEDDING_MODEL=bge-m3
    EMBEDDING_API_KEY=dummy

    # OpenAI 官方
    EMBEDDING_API_BASE=https://api.openai.com/v1
    EMBEDDING_MODEL=text-embedding-3-large
    EMBEDDING_API_KEY=sk-xxx
"""

from langchain_core.embeddings import Embeddings

from app.core.log_adapter import logger

# 延迟导入，避免循环依赖
_embeddings_instance: Embeddings | None = None


def get_embeddings() -> Embeddings:
    """获取嵌入模型实例

    从环境变量读取配置，连接 OpenAI 兼容的嵌入服务。

    Returns:
        Embeddings: 嵌入模型实例

    Examples:
        # 在任何需要使用嵌入的地方
        embeddings = get_embeddings()
        vectors = await embeddings.aembed_documents(["text1"])
    """
    global _embeddings_instance

    if _embeddings_instance is not None:
        return _embeddings_instance

    from app.rag.embedding.embeeding_model import EmbeddingFactory

    _embeddings_instance = EmbeddingFactory.get()

    return _embeddings_instance


def reset_embeddings() -> None:
    """重置嵌入模型实例（主要用于测试）"""
    global _embeddings_instance
    _embeddings_instance = None
    logger.debug("Embeddings instance reset")


__all__ = ["get_embeddings", "reset_embeddings"]
