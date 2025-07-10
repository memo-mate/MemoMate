from threading import Lock
from typing import Literal, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.core import settings
from app.enums import EmbeddingDriverEnum


class MemoMateEmbeddings:
    """嵌入封装

    支持本地和OpenAI的嵌入模型
    args:
        model_name: 模型名称
        driver: 驱动类型
        normalize: 是否归一化

    PS: 目前采用 langchain 封装 embedding 对象，后续考虑使用自封装函数对象
    """

    @staticmethod
    def local_embedding(
        model_name: str = "BAAI/bge-large-zh-v1.5",
        driver: EmbeddingDriverEnum = EmbeddingDriverEnum.CPU,
        normalize: bool = True,
    ) -> Embeddings:
        """本地嵌入"""
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": normalize},
            model_kwargs={"device": driver.value},
        )
        return embedding_model

    @staticmethod
    def openai_embedding(
        api_key: str = settings.OPENAI_API_KEY,
        base_url: str = settings.OPENAI_API_BASE,
        model_name: str = "text-embedding-3-large",
    ) -> Embeddings:
        """OpenAI嵌入"""
        embedding_model = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
        )
        return embedding_model


class EmbeddingConfig(TypedDict):
    provider: Literal["openai", "huggingface"]
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    driver: EmbeddingDriverEnum | None = None
    normalize: bool | None = None


class EmbeddingFactory:
    _instance = None
    _config = None
    _lock = Lock()

    @classmethod
    def init(cls, config: EmbeddingConfig):
        with cls._lock:
            if cls._config != config:
                print("[EmbeddingFactory] Creating new embedding with config:", config)
                # 动态支持多模型
                if config.get("provider") == "openai":
                    cls._instance = MemoMateEmbeddings.openai_embedding(
                        model_name=config.get("model", "text-embedding-3-small"),
                        api_key=config.get("api_key", settings.OPENAI_API_KEY),
                        base_url=config.get("base_url", settings.OPENAI_API_BASE),
                    )
                elif config.get("provider") == "huggingface":
                    cls._instance = MemoMateEmbeddings.local_embedding(
                        model_name=config["model"],
                        driver=config.get("driver", EmbeddingDriverEnum.CPU),
                    )
                else:
                    raise ValueError("Unsupported embedding provider")
                cls._config = config

    @classmethod
    def get(cls) -> Embeddings:
        if cls._instance is None:
            raise RuntimeError("EmbeddingFactory not initialized. Call init(config) first.")
        return cls._instance
