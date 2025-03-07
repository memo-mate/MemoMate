import fitz
from typing import List
import uuid
from loguru import logger
from app.core.config import (
    MODEL_PATH,
    TABLE_NAME,
    TEXT_SPLITTER_CONFIG,
    EMBEDDING_CONFIG,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB  # noqa: F401
import lancedb  # type: ignore
from app.core.db import init_database
import json


class PdfLoader:
    def __init__(self, model_path=MODEL_PATH):
        self.embedding = HuggingFaceEmbeddings(
            model_name=str(model_path), **EMBEDDING_CONFIG
        )
        self.text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CONFIG)
        self.connection = init_database()

    def _read_pdf_file(self, file_path) -> List[str]:
        """提取PDF文本内容及页面元数据"""
        doc = fitz.open(file_path)
        return [
            {"text": page.get_text(), "metadata": {"page": page.number + 1}}
            for page in doc
        ]

    def _split_text(self, documents):
        """分割文本并保留元数据"""
        return self.text_splitter.split_documents(
            [
                Document(page_content=doc["text"], metadata=doc["metadata"])
                for doc in documents
            ]
        )

    def _vector_exists(self, vector_store, text):
        results = vector_store.similarity_search_with_score(
            text, k=1, score_threshold=0.99
        )
        return len(results) > 0

    def _write_to_lancedb(self, texts, source):
        connection = lancedb.connect(self.connection.uri)
        table = connection.open_table(TABLE_NAME)

        for text in texts:
            # 使用embedding模型获取向量
            vector = self.embedding.embed_query(text.page_content)

            # 准备数据
            data = {
                "id": str(uuid.uuid4()),
                "vector": vector,
                "text": text.page_content,
                "source": source,
                "metadata": json.dumps({"page": text.metadata["page"]}),
            }

            # 添加到表中
            table.add([data])
            logger.info(f"已写入PDF内容: {text.page_content[:30]}...")

    @logger.catch
    def process_file(self, file_path):
        pdf_content = self._read_pdf_file(file_path)
        split_docs = self._split_text(pdf_content)
        self._write_to_lancedb(split_docs, file_path)


if __name__ == "__main__":
    loader = PdfLoader()
    loader.process_file(r"D:\Test_Data\sample.pdf")
