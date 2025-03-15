"""
嵌入模块的向量搜索实现
"""

import os
from typing import Any, TypeVar, cast

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from app.core.config import settings

__all__ = ["HuggingFaceEmbeddings"]

# 定义类型变量
T = TypeVar("T")


class HuggingFaceEmbeddings(Embeddings, BaseModel):
    """使用HuggingFace模型的嵌入实现"""

    model_name: str = Field(settings.EMBEDDING_MODEL_PATH)
    cache_folder: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    multi_process: bool = Field(default=settings.EMBEDDING_MULTI_PROCESS)
    client: Any = Field(default=None)

    def __init__(self, **kwargs: Any) -> None:
        """初始化HuggingFaceEmbeddings"""
        super().__init__(**kwargs)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence_transformers包未安装，请使用 'pip install sentence-transformers' 安装")

        self.client = SentenceTransformer(
            model_name_or_path=self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """生成文档嵌入向量"""

        if self.multi_process:
            # 多进程模式
            from concurrent.futures import ProcessPoolExecutor

            # 将文本分成多个批次
            batch_size = max(len(texts) // (os.cpu_count() or 1), 1)
            batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

            # 使用多进程处理
            with ProcessPoolExecutor() as executor:
                embeddings = list(executor.map(self._embed_batch, batches))

            # 合并结果
            return [embedding for batch in embeddings for embedding in batch]
        else:
            # 单进程模式
            embeddings = self.client.encode(texts, normalize_embeddings=True, **self.encode_kwargs)
            # 确保返回的是List[List[float]]类型
            if hasattr(embeddings, "tolist"):
                return cast(list[list[float]], embeddings.tolist())
            return cast(list[list[float]], embeddings)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """嵌入一批文本（用于多进程）"""
        embeddings = self.client.encode(texts, normalize_embeddings=True, **self.encode_kwargs)
        # 确保返回的是List[List[float]]类型
        if hasattr(embeddings, "tolist"):
            return cast(list[list[float]], embeddings.tolist())
        return cast(list[list[float]], embeddings)

    def embed_query(self, text: str) -> list[float]:
        """生成查询嵌入向量"""
        embedding = self.client.encode(text, normalize_embeddings=True, **self.encode_kwargs)
        # 确保返回的是List[float]类型
        if hasattr(embedding, "tolist"):
            return cast(list[float], embedding.tolist())
        return cast(list[float], embedding)
