import json
import os
from pathlib import Path

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from rich import print

from app.core.config import *  # noqa: F403


def test_multi_query():
    try:
        client = QdrantClient(path="app/database", prefer_grpc=True)

        question = "马梓康会开滴滴吗?"
        llm = ChatOpenAI(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.siliconflow.cn/v1",
            temperature=0,
            max_tokens=3200,
            timeout=None,
            max_retries=3,
        )

        # 使用正确的 QdrantRetriever 类
        retriever = QdrantRetriever(client=client)

        # 自定义中文 prompt
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""你是一个AI助手，你的任务是生成3个不同的搜索查询，这些查询与原始查询的含义相似，但使用不同的词语和表达方式。
            这些查询将用于检索文档，所以它们应该是独立的、多样化的，并且能够捕捉原始查询的不同方面。
            请直接返回这些查询，每行一个，不要有编号或其他文本。
            
            原始查询: {question}
            """,
        )

        # 配置日志 - 减少噪音
        import logging

        # 禁用所有其他模块的日志
        logging.basicConfig(level=logging.ERROR)

        # 只启用 langchain.retrievers.multi_query 的 INFO 级别日志
        multi_query_logger = logging.getLogger("langchain.retrievers.multi_query")
        multi_query_logger.setLevel(logging.INFO)

        # 创建一个控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建一个格式化器，只显示消息内容
        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)

        # 清除现有的处理器并添加新的处理器
        multi_query_logger.handlers = []
        multi_query_logger.addHandler(console_handler)

        # 禁用 OpenAI 和 httpx 的日志
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)

        # 禁用 sentence_transformers 的日志
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

        # 禁用 asyncio 的日志
        logging.getLogger("asyncio").setLevel(logging.ERROR)

        # 禁用 pydot 的日志
        logging.getLogger("pydot").setLevel(logging.ERROR)

        print("原始查询:", question)

        # 使用 MultiQueryRetriever 进行多查询
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=llm, prompt=prompt, parser_key="questions"
        )

        # 获取文档
        unique_docs = retriever_from_llm.get_relevant_documents(question)

        print(f"\n获取到 {len(unique_docs)} 个文档")
        if len(unique_docs) > 0:
            print("文档内容:")
            for i, doc in enumerate(unique_docs):
                print(f"文档 {i+1}:")
                print(f"内容: {doc.page_content}")
                print(f"元数据: {doc.metadata}")
                print("-" * 50)
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")


# 修改 QdrantRetriever 类
class QdrantRetriever(BaseRetriever):
    client: QdrantClient  # 使用 Pydantic 字段

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> list[Document]:
        from app.rag.embedding.vector_search import HuggingFaceEmbeddings

        # 将 Path 对象转换为字符串
        embedding = HuggingFaceEmbeddings(model_name=str(Path(r"D:\LLM\bge-large-zh-v1.5")))
        query_vector = embedding.embed_query(query)

        try:
            # 检查集合是否存在
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if "documents" not in collection_names:
                print("警告: 'documents' 集合不存在，需要先创建集合并写入数据。")
                print("返回空结果。")
                return []

            # 如果集合存在，则进行搜索
            search_result = self.client.search(
                collection_name="documents",
                query_vector=query_vector,
                limit=5,
            )
            return [
                Document(
                    page_content=result.payload["text"],
                    metadata=json.loads(result.payload["metadata"]),
                )
                for result in search_result
            ]
        except Exception as e:
            print(f"搜索时发生错误: {str(e)}")
            print("返回空结果。")
            return []


test_multi_query()
