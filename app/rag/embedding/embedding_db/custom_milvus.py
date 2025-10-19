"""Milvus 向量数据库封装

提供与 QdrantVectorStore 类似的接口，支持：
- 向量存储和检索
- 相似度搜索
- MMR 搜索
- 元数据过滤

使用示例:
    from app.rag.embedding.embedding_db.custom_milvus import MilvusVectorStore
    from app.rag.embedding import get_embeddings

    embeddings = get_embeddings()
    vector_store = MilvusVectorStore(
        collection_name="my_collection",
        embeddings=embeddings,
        connection_args={
            "host": "localhost",
            "port": 19530
        }
    )

    # 添加文档
    vector_store.add_texts(["text1", "text2"])

    # 搜索
    results = vector_store.similarity_search("query", k=4)
"""

import uuid
from collections.abc import Iterable
from typing import Any

import numpy as np
from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.core.log_adapter import logger


class MilvusVectorStore(VectorStore):
    """Milvus 向量数据库封装"""

    def __init__(
        self,
        collection_name: str,
        embeddings: Embeddings,
        connection_args: dict[str, Any] | None = None,
        embedding_dim: int = 1024,
        index_params: dict[str, Any] | None = None,
        search_params: dict[str, Any] | None = None,
        create_collection_if_not_exists: bool = True,
    ):
        """初始化 Milvus 向量数据库

        Args:
            collection_name: 集合名称
            embeddings: 嵌入模型
            connection_args: 连接参数，如 {"host": "localhost", "port": 19530}
            embedding_dim: 嵌入维度
            index_params: 索引参数
            search_params: 搜索参数
            create_collection_if_not_exists: 是否自动创建集合
        """
        self.collection_name = collection_name
        self._embeddings = embeddings
        self.embedding_dim = embedding_dim

        # 连接参数
        if connection_args is None:
            connection_args = {"host": "localhost", "port": 19530}

        # 连接 Milvus
        self.alias = f"milvus_{uuid.uuid4().hex[:8]}"
        connections.connect(alias=self.alias, **connection_args)
        logger.info(f"Connected to Milvus at {connection_args}")

        # 默认索引参数（HNSW 索引）
        if index_params is None:
            self.index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64},
            }
        else:
            self.index_params = index_params

        # 默认搜索参数
        if search_params is None:
            self.search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        else:
            self.search_params = search_params

        # 创建或加载集合
        if create_collection_if_not_exists:
            if not utility.has_collection(collection_name, using=self.alias):
                self._create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")
            else:
                logger.info(f"Collection {collection_name} already exists")

        # 加载集合到内存
        self.collection = Collection(collection_name, using=self.alias)
        self.collection.load()

    def _create_collection(self, collection_name: str) -> None:
        """创建集合"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        # 创建 schema
        schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name}")

        # 创建集合
        collection = Collection(name=collection_name, schema=schema, using=self.alias)

        # 创建索引
        collection.create_index(field_name="embedding", index_params=self.index_params)

        logger.info(f"Collection {collection_name} created with index")

    def delete_collection(self) -> bool:
        """删除集合"""
        try:
            utility.drop_collection(self.collection_name, using=self.alias)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.exception("Failed to delete collection", exc_info=e)
            return False

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """添加文本到集合

        Args:
            texts: 文本列表
            metadatas: 元数据列表
            ids: ID 列表
            **kwargs: 其他参数

        Returns:
            list[str]: 添加的文档 ID 列表
        """
        texts_list = list(texts)

        # 生成 ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # 生成元数据
        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        # 生成嵌入向量
        embeddings = self._embeddings.embed_documents(texts_list)

        # 准备插入数据
        entities = [
            ids,
            embeddings,
            texts_list,
            metadatas,
        ]

        # 插入数据
        self.collection.insert(entities)
        self.collection.flush()

        logger.info(f"Added {len(texts_list)} texts to collection {self.collection_name}")
        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            **kwargs: 其他参数，如 expr（过滤表达式）

        Returns:
            list[Document]: 相似文档列表
        """
        # 生成查询向量
        query_embedding = self._embeddings.embed_query(query)

        # 执行搜索
        search_params = self.search_params.copy()
        expr = kwargs.get("expr")

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["text", "metadata"],
        )

        # 转换结果
        documents = []
        for hits in results:
            for hit in hits:
                metadata = hit.entity.get("metadata") or {}
                metadata["_id"] = hit.id
                metadata["_score"] = hit.distance
                metadata["_collection_name"] = self.collection_name

                documents.append(
                    Document(
                        page_content=hit.entity.get("text", ""),
                        metadata=metadata,
                    )
                )

        return documents

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[tuple[Document, float]]:
        """相似度搜索并返回分数

        Args:
            query: 查询文本
            k: 返回结果数量
            **kwargs: 其他参数

        Returns:
            list[tuple[Document, float]]: 文档和分数的元组列表
        """
        query_embedding = self._embeddings.embed_query(query)

        search_params = self.search_params.copy()
        expr = kwargs.get("expr")

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["text", "metadata"],
        )

        documents_with_scores = []
        for hits in results:
            for hit in hits:
                metadata = hit.entity.get("metadata") or {}
                metadata["_id"] = hit.id
                metadata["_collection_name"] = self.collection_name

                doc = Document(
                    page_content=hit.entity.get("text", ""),
                    metadata=metadata,
                )

                documents_with_scores.append((doc, hit.distance))

        return documents_with_scores

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """MMR 搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            fetch_k: 初始检索数量
            lambda_mult: MMR 参数 (0-1)，越大越注重相关性，越小越注重多样性
            **kwargs: 其他参数

        Returns:
            list[Document]: MMR 搜索结果
        """
        # 生成查询向量
        query_embedding = self._embeddings.embed_query(query)

        # 执行搜索
        search_params = self.search_params.copy()
        expr = kwargs.get("expr")

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=fetch_k,
            expr=expr,
            output_fields=["text", "metadata", "embedding"],
        )

        if not results or not results[0]:
            return []

        # 提取文档和向量
        docs = []
        embeddings_list = []

        for hit in results[0]:
            metadata = hit.entity.get("metadata") or {}
            metadata["_id"] = hit.id
            metadata["_score"] = hit.distance
            metadata["_collection_name"] = self.collection_name

            doc = Document(
                page_content=hit.entity.get("text", ""),
                metadata=metadata,
            )
            docs.append(doc)
            embeddings_list.append(hit.entity.get("embedding"))

        # 应用 MMR
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding),
            embeddings_list,
            k=min(k, len(embeddings_list)),
            lambda_mult=lambda_mult,
        )

        # 返回选中的文档
        return [docs[i] for i in selected_indices]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        """删除文档

        Args:
            ids: 要删除的文档 ID 列表
            **kwargs: 其他参数

        Returns:
            bool: 是否删除成功
        """
        try:
            if ids:
                expr = f"id in {ids}"
                self.collection.delete(expr)
                self.collection.flush()
                logger.info(f"Deleted {len(ids)} documents from {self.collection_name}")
            return True
        except Exception as e:
            logger.exception("Failed to delete documents", exc_info=e)
            return False

    @property
    def embeddings(self) -> Embeddings:
        """获取嵌入模型"""
        return self._embeddings

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "MilvusVectorStore":
        """从文本创建 MilvusVectorStore

        Args:
            texts: 文本列表
            embedding: 嵌入模型
            metadatas: 元数据列表
            ids: ID 列表
            **kwargs: 其他参数

        Returns:
            MilvusVectorStore: MilvusVectorStore 实例
        """
        collection_name = kwargs.get("collection_name")
        if not collection_name:
            raise ValueError("collection_name is required")

        connection_args = kwargs.get("connection_args")
        embedding_dim = kwargs.get("embedding_dim", 1024)

        # 创建实例
        vector_store = cls(
            collection_name=collection_name,
            embeddings=embedding,
            connection_args=connection_args,
            embedding_dim=embedding_dim,
            index_params=kwargs.get("index_params"),
            search_params=kwargs.get("search_params"),
            create_collection_if_not_exists=True,
        )

        # 添加文本
        if texts:
            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        return vector_store

    def __del__(self):
        """清理连接"""
        try:
            connections.disconnect(alias=self.alias)
        except Exception:
            pass
