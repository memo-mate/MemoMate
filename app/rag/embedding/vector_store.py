import json
from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from app.core.config import settings
from app.rag.embedding.vector_search import HuggingFaceEmbeddings


class QdrantStore(BaseModel):
    """Qdrant向量存储实现"""

    collection_name: str = "documents"
    path: str | None = Field(default="app/database")
    url: str | None = Field(default=None)
    prefer_grpc: bool = Field(default=True)
    embedding_model: Embeddings | None = None
    client: Any = Field(default=None)
    embedding_dimension: int = Field(default=settings.EMBEDDING_DIMENSION)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        """初始化QdrantStore"""
        # 处理embeddings参数
        if "embeddings" in kwargs:
            kwargs["embedding_model"] = kwargs.pop("embeddings")

        # 确保embedding_model参数已提供
        if "embedding_model" not in kwargs or kwargs["embedding_model"] is None:
            print("警告: 未提供embedding_model参数或为None，使用默认模型")
            kwargs["embedding_model"] = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

        super().__init__(**kwargs)

        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("qdrant-client包未安装，请使用 'pip install qdrant-client' 安装")

        # 如果path和url都为None，则使用内存模式
        if self.path is None and self.url is None:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                path=self.path,
                url=self.url,
                prefer_grpc=self.prefer_grpc,
                force_disable_check_same_thread=True,
            )

        # 获取嵌入维度
        try:
            # 确保embedding_model不为None
            if self.embedding_model is None:
                print("警告: embedding_model为None，使用默认模型")
                self.embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

            self.embedding_dimension = len(self.embedding_model.embed_query("测试查询"))
            print(f"嵌入维度: {self.embedding_dimension}")
        except Exception as e:
            print(f"获取嵌入维度时出错: {str(e)}")
            # 使用默认维度
            self.embedding_dimension = settings.EMBEDDING_DIMENSION
            print(f"使用默认维度: {self.embedding_dimension}")

        # 检查集合是否存在，不存在则创建
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self) -> None:
        """如果集合不存在，则创建集合"""
        from qdrant_client.http import models as rest

        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                # 创建集合
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(
                        size=self.embedding_dimension,
                        distance=rest.Distance.COSINE,
                    ),
                )
                print(f"已创建集合: {self.collection_name}")
        except Exception as e:
            print(f"创建集合时出错: {str(e)}")
            import traceback

            traceback.print_exc()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """添加文本到向量存储"""
        import uuid

        from qdrant_client.http import models as rest

        # 确保embedding_model不为None
        if self.embedding_model is None:
            print("警告: embedding_model为None，使用默认模型")
            self.embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

        # 转换为列表以便多次使用
        texts_list = list(texts)

        # 生成嵌入
        embeddings = self.embedding_model.embed_documents(texts_list)

        # 准备元数据
        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        # 准备ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # 准备点
        points = [
            rest.PointStruct(
                id=id,
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": json.dumps(metadata),
                },
            )
            for id, text, metadata, embedding in zip(ids, texts_list, metadatas, embeddings, strict=False)
        ]

        # 添加点
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return ids

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """添加文档到向量存储"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)

    def get_collection_info(self) -> dict[str, Any]:
        """获取集合信息"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count if hasattr(collection_info, "vectors_count") else 0,
                "status": str(collection_info.status) if hasattr(collection_info, "status") else "unknown",
                "vector_size": collection_info.config.params.vectors.size if hasattr(collection_info, "config") else 0,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine") -> None:
        """创建集合"""
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": vector_size, "distance": distance},
        )
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """相似度搜索"""
        # 确保embedding_model不为None
        if self.embedding_model is None:
            print("警告: embedding_model为None，使用默认模型")
            self.embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

        # 生成查询嵌入
        query_embedding = self.embedding_model.embed_query(query)

        # 执行搜索
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=filter,
        )

        # 转换为文档
        documents = [
            Document(
                page_content=result.payload["text"],
                metadata=json.loads(result.payload["metadata"]),
            )
            for result in results
        ]

        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """带分数的相似度搜索"""
        # 确保embedding_model不为None
        if self.embedding_model is None:
            print("警告: embedding_model为None，使用默认模型")
            self.embedding_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_PATH)

        # 生成查询嵌入
        query_embedding = self.embedding_model.embed_query(query)

        # 执行搜索
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=filter,
            with_payload=True,
            with_vectors=False,
        )

        # 转换为文档和分数
        documents_with_scores = [
            (
                Document(
                    page_content=result.payload["text"],
                    metadata=json.loads(result.payload["metadata"]),
                ),
                result.score,
            )
            for result in results
        ]

        return documents_with_scores

    def delete(self, ids: list[str]) -> None:
        """删除向量"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )

    def delete_collection(self) -> None:
        """删除集合"""
        self.client.delete_collection(collection_name=self.collection_name)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "QdrantStore":
        """从文本创建向量存储"""
        store = cls(embedding_model=embedding, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "QdrantStore":
        """从文档创建向量存储"""
        store = cls(embedding_model=embedding, **kwargs)
        store.add_documents(documents=documents, ids=ids)
        return store
