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
        normalize: bool = True,
    ) -> Embeddings:
        """OpenAI嵌入"""
        embedding_model = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            # encode_kwargs={"normalize_embeddings": normalize},
            dimensions=1024,
        )
        return embedding_model
