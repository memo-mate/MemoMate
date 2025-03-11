import pandas as pd
import uuid
import json
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


class ExcelLoader:
    def __init__(self, model_path=MODEL_PATH):
        self.embedding = HuggingFaceEmbeddings(
            model_name=str(model_path), **EMBEDDING_CONFIG
        )
        self.text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CONFIG)
        self.connection = init_database()
        self.file_path = None

    def _read_excel_file(self, file_path):
        """读取Excel文件内容，并将每个工作表转换为文本"""
        excel_data = []

        # 读取所有工作表
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        for sheet_name in sheet_names:
            # 读取工作表
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # 将DataFrame转换为字符串
            sheet_text = f"工作表: {sheet_name}\n"
            sheet_text += df.to_string(index=False)

            # 添加到结果列表
            excel_data.append(
                {"text": sheet_text, "metadata": {"sheet_name": sheet_name}}
            )

        self.file_path = file_path

        return excel_data

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
                "metadata": json.dumps(
                    {
                        "sheet_name": text.metadata.get("sheet_name", ""),
                        "file_path": self.file_path,
                    }
                ),
            }

            # 添加到表中
            table.add([data])
        logger.info(f"文件{self.file_path}已写入数据库")

    @logger.catch
    def process_file(self, file_path):
        excel_content = self._read_excel_file(file_path)
        split_docs = self._split_text(excel_content)
        self._write_to_lancedb(split_docs, file_path)


if __name__ == "__main__":
    loader = ExcelLoader()
    loader.process_file(r"D:\Test_Data\sample.xlsx")
