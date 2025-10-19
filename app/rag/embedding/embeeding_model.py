from __future__ import annotations

from threading import Lock
from typing import NotRequired, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core import settings
from app.core.log_adapter import logger


class EmbeddingConfig(TypedDict):
    """嵌入模型配置

    使用 OpenAI 兼容 API，支持：
    - vLLM 嵌入服务
    - Ollama 嵌入服务
    - OpenAI 官方 API
    - 其他 OpenAI 兼容服务
    """

    model: str
    api_key: NotRequired[str]
    base_url: NotRequired[str]


class EmbeddingFactory:
    """嵌入模型工厂类（单例模式）

    统一使用 OpenAI 兼容 API，支持 vLLM、Ollama 等服务。

    使用示例:
        # 方式1：显式初始化（推荐）
        EmbeddingFactory.init({
            "model": "BAAI/bge-m3",
            "api_key": "dummy",
            "base_url": "http://localhost:8000/v1"
        })
        embeddings = EmbeddingFactory.get()

        # 方式2：自动从环境变量初始化
        embeddings = EmbeddingFactory.get()  # 自动读取配置
    """

    _instance: Embeddings | None = None
    _config: EmbeddingConfig | None = None
    _lock = Lock()

    def __new__(cls):
        raise RuntimeError("EmbeddingFactory is a singleton. Use EmbeddingFactory.get() instead")

    @classmethod
    def init(cls, config: EmbeddingConfig) -> None:
        """初始化嵌入模型

        Args:
            config: 嵌入模型配置

        Raises:
            ValueError: 配置不合法时抛出
        """
        with cls._lock:
            # 验证配置
            cls._validate_config(config)

            # 检查配置是否变化
            if cls._config == config and cls._instance is not None:
                logger.debug("Embedding configuration unchanged, reusing existing instance")
                return

            logger.info(
                "Initializing embedding model",
                model=config["model"],
                base_url=config.get("base_url", "default"),
            )

            # 保存配置并创建实例
            cls._config = config
            cls._create_instance(config)

    @classmethod
    def get(cls) -> Embeddings:
        """获取嵌入模型实例

        如果未初始化，会从环境变量自动初始化。

        Returns:
            Embeddings: 嵌入模型实例
        """
        # 快速路径：已经初始化
        if cls._instance is not None:
            return cls._instance

        # 慢速路径：需要初始化（加锁）
        with cls._lock:
            # 双重检查锁定
            if cls._instance is not None:
                return cls._instance

            # 从环境变量读取配置并初始化
            logger.info("EmbeddingFactory auto-initializing from environment variables")

            config = cls._get_config_from_env()
            cls._config = config
            cls._create_instance(config)

            return cls._instance

    @classmethod
    def _get_config_from_env(cls) -> EmbeddingConfig:
        """从环境变量读取配置

        Returns:
            EmbeddingConfig: 嵌入模型配置
        """
        logger.info(
            "Loading embedding config",
            base_url=settings.EMBEDDING_API_BASE,
            model=settings.EMBEDDING_MODEL,
        )

        config: EmbeddingConfig = {
            "model": settings.EMBEDDING_MODEL,
            "api_key": settings.EMBEDDING_API_KEY or "dummy",
            "base_url": settings.EMBEDDING_API_BASE,
        }

        return config

    @classmethod
    def _create_instance(cls, config: EmbeddingConfig) -> None:
        """创建嵌入模型实例（使用 OpenAI 兼容 API）

        Args:
            config: 嵌入模型配置

        Raises:
            ValueError: 创建失败时抛出
        """
        try:
            cls._instance = OpenAIEmbeddings(
                model=config["model"],
                api_key=config.get("api_key", "dummy"),
                base_url=config.get("base_url"),
            )

            logger.info(
                "Embedding model initialized successfully",
                model=config["model"],
                base_url=config.get("base_url"),
            )

        except Exception as e:
            logger.exception("Failed to initialize embedding model", exc_info=e, config=config)
            raise

    @classmethod
    def reset(cls) -> None:
        """重置工厂状态（主要用于测试）"""
        with cls._lock:
            cls._instance = None
            cls._config = None
            logger.debug("EmbeddingFactory reset")

    @classmethod
    def is_initialized(cls) -> bool:
        """检查是否已初始化

        Returns:
            bool: 是否已初始化
        """
        return cls._instance is not None

    @classmethod
    def _validate_config(cls, config: EmbeddingConfig) -> None:
        """验证配置合法性

        Args:
            config: 嵌入模型配置

        Raises:
            ValueError: 配置不合法时抛出
        """
        if not config.get("model"):
            raise ValueError("Missing required field: model")

        if not config.get("base_url"):
            raise ValueError("Missing required field: base_url")
