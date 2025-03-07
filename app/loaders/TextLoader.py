import lancedb  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB  # type: ignore  # noqa: F401
from loguru import logger
from app.core.config import (
    MODEL_PATH,
    TEXT_SPLITTER_CONFIG,
    EMBEDDING_CONFIG,
    TABLE_NAME,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.db import init_database
import uuid


class TextLoader:
    def __init__(self, model_path=MODEL_PATH):
        self.embedding = HuggingFaceEmbeddings(
            model_name=str(model_path), **EMBEDDING_CONFIG
        )
        self.text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CONFIG)
        self.connection = init_database()

    def _read_txt_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _split_text(self, text):
        return self.text_splitter.create_documents([text])

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
            vector = self.embedding.embed_query(text)

            # 准备数据
            data = {
                "id": str(uuid.uuid4()),
                "vector": vector,
                "text": text,
                "source": source,
                "metadata": "",
            }

            # 添加到表中
            table.add([data])
            logger.info(f"已写入: {text[:30]}...")

    @logger.catch
    def process_file(self, file_path):
        text_content = self._read_txt_file(file_path)
        split_docs = self._split_text(text_content)
        self._write_to_lancedb([doc.page_content for doc in split_docs], file_path)


if __name__ == "__main__":
    loader = TextLoader()
    loader.process_file(r"D:\Test_Data\10月11日 Win版本发现问题.md")
