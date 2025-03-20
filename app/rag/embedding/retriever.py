import json
from pathlib import Path
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.core.config import settings
from app.rag.embedding.vector_search import HuggingFaceEmbeddings


class QdrantRetriever(BaseRetriever, BaseModel):
    """Qdrant检索器"""

    client: QdrantClient  # 使用 Pydantic 字段
    collection_name: str = "documents"
    embedding_model_path: str | None = None
    embedding_model: HuggingFaceEmbeddings | None = None
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs) -> None:
        """初始化QdrantRetriever"""
        super().__init__(**kwargs)

        # 如果提供了embedding_model_path但没有提供embedding_model，则创建嵌入模型
        if self.embedding_model is None and self.embedding_model_path is not None:
            try:
                self.embedding_model = HuggingFaceEmbeddings(model_name=str(Path(self.embedding_model_path)))
                print(f"已初始化嵌入模型: {self.embedding_model_path}")
            except Exception as e:
                print(f"初始化嵌入模型时出错: {str(e)}")
                # 使用默认路径
                default_path = settings.EMBEDDING_MODEL_PATH
                print(f"尝试使用默认路径: {default_path}")
                self.embedding_model = HuggingFaceEmbeddings(model_name=str(Path(default_path)))

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> list[Document]:
        """获取相关文档"""
        # 确保嵌入模型已初始化
        if self.embedding_model is None:
            embedding_model_path = self.embedding_model_path or settings.EMBEDDING_MODEL_PATH
            print(f"嵌入模型未初始化，使用路径: {embedding_model_path}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=str(Path(embedding_model_path)))

        # 生成查询向量
        try:
            print(f"生成查询向量: {query}")
            query_vector = self.embedding_model.embed_query(query)
            print(f"查询向量维度: {len(query_vector)}")
        except Exception as e:
            print(f"生成查询向量时出错: {str(e)}")
            import traceback

            traceback.print_exc()
            return []

        try:
            # 检查集合是否存在
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            print(f"可用集合: {collection_names}")
            print(f"当前集合名称: {self.collection_name}")
            print(f"客户端ID: {id(self.client)}")

            if self.collection_name not in collection_names:
                print(f"警告: '{self.collection_name}' 集合不存在，需要先创建集合并写入数据。")

                # 尝试创建集合
                try:
                    from qdrant_client.http import models as rest

                    print(f"尝试创建集合: {self.collection_name}")
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=rest.VectorParams(
                            size=len(query_vector),
                            distance=rest.Distance.COSINE,
                        ),
                    )
                    print(f"成功创建集合: {self.collection_name}")
                    # 重新检查集合
                    collections = self.client.get_collections().collections
                    collection_names = [collection.name for collection in collections]
                    print(f"创建后的可用集合: {collection_names}")

                    if self.collection_name not in collection_names:
                        print("集合创建失败，返回空结果。")
                        return []
                except Exception as e:
                    print(f"创建集合时出错: {str(e)}")
                    print("返回空结果。")
                    return []

            # 执行搜索
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.top_k,
            )

            print(f"搜索结果数量: {len(search_result)}")

            # 转换为文档
            return [
                Document(
                    page_content=result.payload["text"],
                    metadata=json.loads(result.payload["metadata"]),
                )
                for result in search_result
            ]
        except Exception as e:
            print(f"搜索时发生错误: {str(e)}")
            import traceback

            traceback.print_exc()
            print("返回空结果。")
            return []


class MultiQueryRetriever(BaseRetriever, BaseModel):
    """多查询检索器"""

    client: QdrantClient  # 使用 Pydantic 字段
    retriever: BaseRetriever
    llm: Any
    query_count: int = 3
    prompt: PromptTemplate | None = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """初始化MultiQueryRetriever"""
        super().__init__(**kwargs)

        # 如果没有提供prompt，使用默认的中文prompt
        if self.prompt is None:
            self.prompt = PromptTemplate(
                input_variables=["question"],
                template="""你是一个AI助手，你的任务是生成{query_count}个不同的搜索查询，这些查询与原始查询的含义相似，但使用不同的词语和表达方式。
                这些查询将用于检索文档，所以它们应该是独立的、多样化的，并且能够捕捉原始查询的不同方面。
                请直接返回这些查询，每行一个，不要有编号或其他文本。

                原始查询: {question}
                """.format(query_count=self.query_count, question="{question}"),
            )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> list[Document]:
        """获取相关文档"""
        # 生成多个查询
        queries = self._generate_queries(query)

        # 使用每个查询获取文档
        all_docs = []
        for q in queries:
            docs = self.retriever.invoke(q)
            all_docs.extend(docs)

        # 去重
        unique_docs = self._get_unique_documents(all_docs)

        return unique_docs

    def _generate_queries(self, query: str) -> list[str]:
        """生成多个查询"""
        # 使用LLM生成多个查询
        response = self.llm.invoke(self.prompt.format(question=query))

        # 解析响应
        queries = [query]  # 始终包含原始查询

        # 从响应中提取查询
        generated_queries = response.content.strip().split("\n")
        generated_queries = [q.strip() for q in generated_queries if q.strip()]

        # 添加生成的查询
        queries.extend(generated_queries)

        # 确保查询数量不超过预期
        if len(queries) > self.query_count + 1:
            queries = queries[: self.query_count + 1]

        print(f"生成的查询: {queries}")

        return queries

    def _get_unique_documents(self, documents: list[Document]) -> list[Document]:
        """获取唯一文档"""
        # 使用文档内容作为键去重
        unique_docs = {}
        for doc in documents:
            content = doc.page_content
            if content not in unique_docs:
                unique_docs[content] = doc

        return list(unique_docs.values())


class HybridRetriever(BaseRetriever, BaseModel):
    """混合检索器"""

    vector_retriever: BaseRetriever
    keyword_retriever: BaseRetriever | None = None
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> list[Document]:
        """获取相关文档"""
        # 向量检索
        vector_docs = self.vector_retriever.invoke(query)

        # 如果没有关键词检索器，直接返回向量检索结果
        if self.keyword_retriever is None:
            return vector_docs[: self.top_k]

        # 关键词检索
        keyword_docs = self.keyword_retriever.invoke(query)

        # 合并结果
        merged_docs = self._merge_results(vector_docs, keyword_docs)

        return merged_docs[: self.top_k]

    def _merge_results(self, vector_docs: list[Document], keyword_docs: list[Document]) -> list[Document]:
        """合并结果"""
        # 创建文档分数映射
        doc_scores = {}

        # 处理向量检索结果
        for i, doc in enumerate(vector_docs):
            content = doc.page_content
            score = self.vector_weight * (1.0 - i / len(vector_docs))
            doc_scores[content] = {"doc": doc, "score": score}

        # 处理关键词检索结果
        for i, doc in enumerate(keyword_docs):
            content = doc.page_content
            score = self.keyword_weight * (1.0 - i / len(keyword_docs))
            if content in doc_scores:
                doc_scores[content]["score"] += score
            else:
                doc_scores[content] = {"doc": doc, "score": score}

        # 按分数排序
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)

        return [item["doc"] for item in sorted_docs]
