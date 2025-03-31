import uuid
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.core.log_adapter import logger


class ScoreDocumentDict(TypedDict):
    score: float
    doc: Document


class QdrantDB:
    def __init__(
        self,
        embeddings: Embeddings,
        url: str,
        api_key: str | None = None,
    ):
        """
        初始化Qdrant客户端

        Args:
            embeddings: 嵌入模型
            url: Qdrant服务器URL
            api_key: Qdrant API密钥
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.embeddings = embeddings

        self.__content_page_key = "qdrant_content_page_key"
        self.__metadata_key = "qdrant_metadata_key"

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1024,
        distance: Distance = Distance.COSINE,
        force: bool = False,
    ) -> bool:
        """
        创建一个新的集合

        Args:
            collection_name: 集合名称
            vector_size: 向量大小
            distance: 距离计算方式
            force: 是否强制创建（如存在则先删除）

        Returns:
            bool: 是否成功创建
        """
        logger.info(f"创建集合 {collection_name}")
        if force:
            self.delete_collection(collection_name)
        else:
            # 检查集合是否存在
            if self.client.collection_exists(collection_name):
                logger.info(f"集合 {collection_name} 已存在")
                return False

        self.client.create_collection(
            collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=distance)
        )
        return True

    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"成功删除集合 {collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合 {collection_name} 失败", error=str(e))
            return False

    def list_collections(self) -> list[str]:
        collections = self.client.get_collections().collections
        return [collection.name for collection in collections]

    def add_vectors(
        self,
        collection_name: str,
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """
        将向量添加到集合中

        Args:
            collection_name: 集合名称
            vectors: 向量列表
            metadatas: 元数据列表
        Returns:
            list: 添加的向量ID列表
        """

        if metadatas is None:
            metadatas = [{} for _ in vectors]

        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload={**metadata})
            for vector, metadata in zip(vectors, metadatas, strict=True)
        ]

        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"向集合 {collection_name} 添加了 {len(vectors)} 条文本")

        return [str(point.id) for point in points]

    def add_documents(self, collection_name: str, documents: list[Document]) -> None:
        """
        将文档添加到集合中
        """
        vectors = [self.embeddings.embed_documents(document.page_content) for document in documents]
        metadatas = [document.metadata for document in documents]
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload={**metadata})
            for vector, metadata in zip(vectors, metadatas, strict=True)
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"向集合 {collection_name} 添加了 {len(documents)} 条文本")

    def delete_vectors(self, collection_name: str, ids: list[int | str]) -> bool:
        """
        从集合中删除向量

        Args:
            collection_name: 集合名称
            ids: 要删除的ID列表

        Returns:
            bool: 是否成功删除
        """
        try:
            self.client.delete(collection_name=collection_name, points_selector=ids)
            logger.info(f"从集合 {collection_name} 中删除了 {len(ids)} 条向量")
            return True
        except Exception as e:
            logger.error(f"从集合 {collection_name} 中删除向量失败", error=str(e))
            return False

    def search(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 5,
        filter: Filter | None = None,
    ) -> list[ScoreDocumentDict]:
        query_vector = self.embeddings.embed_query(query_text)
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter,
        )

        results: list[ScoreDocumentDict] = []
        for result in search_result:
            metadata = result.payload.get(self.__metadata_key) or {}
            metadata["_id"] = result.id
            metadata["_collection_name"] = collection_name
            results.append(
                {
                    "score": result.score,
                    "doc": Document(
                        id=result.id,
                        page_content=result.payload.get(self.__content_page_key, ""),
                        metadata=metadata,
                    ),
                }
            )

        return results

    def search_by_metadata(
        self,
        collection_name: str,
        metadatas: dict[str, Any],
        limit: int = 5,
    ) -> list[ScoreDocumentDict]:
        """
        通过元数据搜索

        Args:
            collection_name: 集合名称
            metadatas: 元数据
            limit: 返回结果数量

        Returns:
            list[ScoreDocumentDict]: 搜索结果
        """
        filter = Filter(
            must=[FieldCondition(key=key, match=MatchValue(value=value)) for key, value in metadatas.items()]
        )

        search_result = self.client.query_points(
            collection_name=collection_name,
            query_filter=filter,
            with_payload=True,
            limit=limit,
        ).points

        results: list[ScoreDocumentDict] = []
        for result in search_result:
            metadata = result.payload.get(self.__metadata_key) or {}
            metadata["_id"] = result.id
            metadata["_collection_name"] = collection_name
            results.append(
                {
                    "score": result.score,
                    "doc": Document(
                        id=result.id,
                        page_content=result.payload.get(self.__content_page_key, ""),
                        metadata=metadata,
                    ),
                }
            )

        return results

    def update_vector(
        self,
        collection_name: str,
        id: int,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        更新向量

        Args:
            collection_name: 集合名称
            id: 向量ID
            text: 新文本
            metadata: 新元数据

        Returns:
            bool: 是否成功更新
        """
        try:
            vector = self.embeddings.embed_query(text)
            point = PointStruct(id=id, vector=vector, payload=metadata)

            self.client.upsert(collection_name=collection_name, points=[point])

            logger.info(f"更新集合 {collection_name} 中ID为 {id} 的向量")
            return True
        except Exception as e:
            logger.error(f"更新集合 {collection_name} 中ID为 {id} 的向量失败", error=str(e))
            return False
