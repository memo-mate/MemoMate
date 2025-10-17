import uuid
from collections.abc import Callable, Iterable
from typing import Any, Literal, TypedDict

import numpy as np
from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.core.log_adapter import logger


class ScoreDocumentDict(TypedDict):
    score: float
    doc: Document


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str,
        embeddings: Embeddings,
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
        distance_type: Distance = Distance.COSINE,
        verily_distance: bool = True,
        vector_size: int = 1024,
        create_collection_if_not_exists: bool = True,
    ):
        """
        初始化Qdrant客户端


        Args:
            embeddings: 嵌入模型
            url: Qdrant服务器URL，如果为None，则使用path
            api_key: Qdrant API密钥
            path: Qdrant数据库路径，如果为None，则使用url
            distance_type: 距离类型
            verily_distance: 是否使用距离类型
            vector_size: 向量大小
            create_collection_if_not_exists: 是否创建集合
        """
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            raise ValueError("url and path can't be None at the same time.")
        logger.info(f"QdrantClient类型: {type(self.client)}")
        logger.info(f"QdrantClient.search: {QdrantClient.search}")
        self._embeddings = embeddings
        self.collection_name = collection_name
        self.distance_type = distance_type  # 距离类型
        self.vector_size = vector_size  # 向量大小
        # 检查集合是否存在
        if create_collection_if_not_exists:
            self.create_collection_if_not_exists(self.collection_name)
        # 获取当前集合的配置
        if verily_distance:
            collection_info = self.client.get_collection(self.collection_name)
            self.distance_type = collection_info.config.params.vectors.distance

        self.__content_page_key = "qdrant_content_page_key"
        self.__metadata_key = "qdrant_metadata_key"

    def create_collection_if_not_exists(
        self,
        collection_name: str,
        force: bool = False,
    ) -> None:
        """
        创建一个新的集合

        Args:
            collection_name: 集合名称
            distance: 距离计算方式
            force: 是否强制创建（如存在则先删除）
        """
        logger.info(f"创建集合 {collection_name}")
        is_exists = self.client.collection_exists(collection_name)
        if is_exists and not force:
            return
        if is_exists and force:
            self.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance_type),
        )

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
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
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
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        if metadatas is None:
            metadatas = [{} for _ in vectors]

        points = [
            PointStruct(id=id, vector=vector, payload={**metadata})
            for id, vector, metadata in zip(ids, vectors, metadatas, strict=True)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"向集合 {self.collection_name} 添加了 {len(vectors)} 条文本")

        return [str(point.id) for point in points]

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
        query: str,
        search_type: Literal["similarity", "similarity_score_threshold", "mmr"],
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return super().search(query, search_type, **kwargs)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """
        相似性搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            **kwargs: 额外参数，可包含filter等Qdrant搜索参数

        Returns:
            list[Document]: 相似文档列表
        """
        query_vector = self.embeddings.embed_query(query)

        # 从kwargs中提取filter参数
        filter_param = kwargs.get("filter")

        # 准备搜索参数
        search_params = {
            "collection_name": self.collection_name,
            "query": query_vector,
            "limit": k,
        }

        # 如果提供了filter，则添加到搜索参数中
        if filter_param is not None:
            search_params["query_filter"] = filter_param

        # 添加其他可能的搜索参数
        for key, value in kwargs.items():
            if key not in ["filter"] and value is not None:
                search_params[key] = value

        try:
            search_result = self.client.query_points(**search_params).points

            documents = []
            for result in search_result:
                # 提取和处理元数据
                metadata = result.payload.get(self.__metadata_key) or {}
                # 添加有用的元信息
                metadata["_id"] = result.id
                metadata["_collection_name"] = self.collection_name
                metadata["_score"] = result.score

                documents.append(
                    Document(
                        id=result.id,
                        page_content=result.payload.get(self.__content_page_key, ""),
                        metadata=metadata,
                    )
                )

            return documents

        except Exception as e:
            logger.exception("执行相似性搜索时出错", exc_info=e)
            return []

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[tuple[Document, float]]:
        """
        相似性搜索并返回结果及其得分

        Args:
            query: 查询文本
            k: 返回结果数量
            **kwargs: 额外参数，可包含filter等Qdrant搜索参数

        Returns:
            list[tuple[Document, float]]: 文档和相似度得分的元组列表
        """
        query_vector = self.embeddings.embed_query(query)

        # 从kwargs中提取filter参数
        filter_param = kwargs.get("filter")

        # 准备搜索参数
        search_params = {
            "collection_name": self.collection_name,
            "query": query_vector,
            "limit": k,
        }

        # 如果提供了filter，则添加到搜索参数中
        if filter_param is not None:
            search_params["query_filter"] = filter_param

        # 添加其他可能的搜索参数
        for key, value in kwargs.items():
            if key not in ["filter"] and value is not None:
                search_params[key] = value

        try:
            search_result = self.client.query_points(**search_params).points

            results: list[tuple[Document, float]] = []
            for result in search_result:
                # 提取和处理元数据
                metadata = result.payload.get(self.__metadata_key) or {}
                metadata["_id"] = result.id
                metadata["_collection_name"] = self.collection_name

                results.append(
                    (
                        Document(
                            id=result.id,
                            page_content=result.payload.get(self.__content_page_key, ""),
                            metadata=metadata,
                        ),
                        result.score,
                    )
                )

            return results

        except Exception as e:
            logger.exception("执行带分数的相似性搜索时出错", exc_info=e)
            return []

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        选择相关性得分函数，根据距离度量类型返回适当的得分转换函数

        当使用不同的距离度量方式时，需要不同的转换公式:
        - Cosine: 1.0 - distance/2 (余弦距离范围是[0,2])
        - Dot: 归一化的点积，保持原样
        - Euclidean: 使用高斯转换 exp(-distance)

        Returns:
            Callable[[float], float]: 相关性得分函数
        """
        try:
            # 根据距离类型返回合适的转换函数
            if self.distance_type == Distance.COSINE:
                return lambda distance: 1.0 - distance / 2
            elif self.distance_type == Distance.DOT:
                # 点积已经是相似度，保持不变
                return lambda distance: distance
            elif self.distance_type == Distance.EUCLID:
                # 欧几里得距离使用高斯转换
                return lambda distance: float(1.0 / (1.0 + distance))
            else:
                # 默认转换
                logger.warning(f"未知的距离类型: {self.distance_type}，使用默认转换")
                return lambda distance: 1.0 - distance

        except Exception as e:
            logger.exception("获取距离类型失败，使用默认转换", exc_info=e)
            # 默认假设使用余弦距离
            return lambda distance: 1.0 - distance

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Filter | None = None,
        search_params: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        consistency: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """
        最大边际相关性搜索，用于平衡相关性和多样性

        Args:
            query: 查询文本
            k: 返回结果数量
            fetch_k: 检索结果数量，用于计算MMR的候选集大小
            lambda_mult: 多样性参数，范围[0,1]，值越大表示越注重相关性，越小表示越注重多样性
            filter: 元数据过滤条件
            search_params: 额外的搜索参数
            score_threshold: 相似度阈值，低于此值的结果将被过滤
            consistency: 读取一致性设置
            **kwargs: 其他Qdrant搜索参数

        Returns:
            list[Document]: 最大边际相关性搜索结果
        """
        # 将查询转换为向量表示
        query_vector = self.embeddings.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            query_vector,
            k,
            fetch_k,
            lambda_mult,
            filter,
            search_params,
            score_threshold,
            consistency,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Filter | None = None,
        search_params: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        consistency: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """
        通过向量搜索
        """
        try:
            # 执行向量搜索，获取包含向量的结果
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                query_filter=filter,
                search_params=search_params,
                limit=fetch_k,
                with_payload=True,
                with_vectors=True,  # 确保返回向量
                score_threshold=score_threshold,
                consistency=consistency,
                **kwargs,
            ).points

            # 如果没有找到结果，返回空列表
            if not results:
                logger.info("没有找到匹配的文档")
                return []

            # 构建文档列表和向量列表
            docs_and_scores: list[tuple[Document, float]] = []
            embeddings: list[list[float]] = []

            for result in results:
                # 提取元数据，如果没有则使用空字典
                metadata = result.payload.get(self.__metadata_key) or {}
                # 添加有用的元信息
                metadata["_id"] = result.id
                metadata["_collection_name"] = self.collection_name
                metadata["_score"] = result.score

                # 创建文档对象
                doc = Document(
                    id=result.id,
                    page_content=result.payload.get(self.__content_page_key, ""),
                    metadata=metadata,
                )

                # 添加到结果列表
                docs_and_scores.append((doc, result.score))
                # 添加向量到向量列表
                embeddings.append(result.vector)

            # 使用MMR算法选择多样化的结果
            selected_indices = maximal_marginal_relevance(
                # query_embedding=embedding,
                query_embedding=np.array(embedding),
                embedding_list=embeddings,
                k=min(k, len(embeddings)),  # 确保k不超过可用文档数量
                lambda_mult=lambda_mult,
            )

            # 根据MMR选择的索引构建最终结果
            mmr_docs = []
            for idx in selected_indices:
                doc = docs_and_scores[idx][0]
                # 添加MMR排名信息
                doc.metadata["_mmr_rank"] = selected_indices.index(idx) + 1
                mmr_docs.append(doc)

            return mmr_docs

        except Exception as e:
            logger.exception("执行最大边际相关性搜索时出错", exc_info=e)
            return []

    def search_by_metadata(
        self,
        metadatas: dict[str, Any],
        limit: int = 5,
    ) -> list[Document]:
        """
        通过元数据搜索

        Args:
            metadatas: 元数据
            limit: 返回结果数量

        Returns:
            list[ScoreDocumentDict]: 搜索结果
        """
        try:
            filter = Filter(
                must=[FieldCondition(key=key, match=MatchValue(value=value)) for key, value in metadatas.items()]
            )

            # 使用scroll API进行纯过滤查询，而不是vector search
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter,
                limit=limit,
                with_payload=True,
            )[0]  # scroll返回(points, next_page_offset)元组，我们只需要points

            results: list[ScoreDocumentDict] = []
            for result in search_result:
                metadata = result.payload.get(self.__metadata_key) or {}
                metadata["_id"] = result.id
                metadata["_collection_name"] = self.collection_name
                # scroll API不返回score，所以我们设置一个默认值
                results.append(
                    Document(
                        id=result.id,
                        page_content=result.payload.get(self.__content_page_key, ""),
                        metadata=metadata,
                    )
                )

            return results
        except Exception as e:
            logger.exception("通过元数据搜索时出错", exc_info=e)
            return []

    def update_vector(
        self,
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

            self.client.upsert(collection_name=self.collection_name, points=[point])

            logger.info(f"更新集合 {self.collection_name} 中ID为 {id} 的向量")
            return True
        except Exception as e:
            logger.error(f"更新集合 {self.collection_name} 中ID为 {id} 的向量失败", error=str(e))
            return False

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            raise ValueError("Embeddings is not set")
        return self._embeddings

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete by vector ID or other criteria."""
        result = self.client.delete(collection_name=self.collection_name, points_selector=ids)
        return result.status == "completed"

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        """Get documents by their IDs."""
        result = self.client.retrieve(collection_name=self.collection_name, ids=ids, with_payload=True)
        return [
            Document(
                id=point.id,
                page_content=point.payload.get(self.__content_page_key, ""),
                metadata=point.payload.get(self.__metadata_key) or {},
            )
            for point in result
        ]

    def get_all(self, limit: int | None = None, with_vectors: bool = False) -> list[Document]:
        """
        获取集合中的所有文档

        Args:
            limit: 限制返回的文档数量，None 表示获取所有
            with_vectors: 是否包含向量数据

        Returns:
            list[Document]: 文档列表
        """
        try:
            all_points: list[PointStruct] = []
            offset = None
            batch_size = 100

            while True:
                result, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
                all_points.extend(result)

                # 如果设置了 limit 且已达到，则停止
                if limit and len(all_points) >= limit:
                    all_points = all_points[:limit]
                    break

                if next_offset is None:
                    break
                offset = next_offset

            # 转换为 Document 对象
            documents = [
                Document(
                    id=point.id,
                    page_content=point.payload.get(self.__content_page_key, ""),
                    metadata=point.payload.get(self.__metadata_key) or {},
                )
                for point in all_points
            ]

            logger.info(f"从集合 {self.collection_name} 获取了 {len(documents)} 个文档")
            return documents

        except Exception as e:
            logger.exception("获取所有文档时出错", exc_info=e)
            return []

    def add_texts(
        self, texts: Iterable[str], metadatas: list[dict] | None = None, *, ids: list[str] | None = None, **kwargs: Any
    ) -> list[str]:
        """
        添加文本到集合中

        Args:
            texts: 文本列表
            metadatas: 元数据列表
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        return self.add_vectors(
            vectors=self.embeddings.embed_documents(list(texts)),
            metadatas=metadatas,
            ids=ids,
        )

    @classmethod
    def from_texts(
        cls: type["QdrantVectorStore"],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "QdrantVectorStore":
        """
        从文本创建QdrantVectorStore实例

        Args:
            texts: 文本列表
            embedding: 嵌入模型
            metadatas: 元数据列表
            ids: ID列表
            **kwargs: 其他参数，包括collection_name, url, api_key等

        Returns:
            QdrantVectorStore: QdrantVectorStore
        """
        logger.info(f"从文本创建QdrantVectorStore实例: {kwargs}")
        # 提取必要的参数
        collection_name = kwargs.get("collection_name")
        if not collection_name:
            raise ValueError("collection_name 是必须的参数")

        url = kwargs.get("url")
        if not url:
            raise ValueError("url 是必须的参数")

        api_key = kwargs.get("api_key")
        distance_type = kwargs.get("distance_type", Distance.COSINE)
        vector_size = kwargs.get("vector_size", len(embedding.embed_query("test")))

        # 创建QdrantDB实例
        qdrant_db = cls(
            collection_name=collection_name,
            embeddings=embedding,
            url=url,
            api_key=api_key,
            distance_type=distance_type,
            verily_distance=kwargs.get("verily_distance", True),
            vector_size=vector_size,
            create_collection_if_not_exists=kwargs.get("create_collection_if_not_exists", True),
        )

        # 处理元数据
        if metadatas is None and texts:
            metadatas = [{} for _ in texts]

        # 添加文本
        if texts:
            qdrant_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        return qdrant_db
