"""
嵌入模型调用
"""

import requests
from langchain_openai import OpenAIEmbeddings
from pydantic import Field, SecretStr

from app.core.log_adapter import logger

__all__ = [
    "SiliconCloudEmbedding",
    "OpenAIEmbedding",
]


class SiliconCloudEmbedding:
    """使用硅基流动接口的模型嵌入"""

    key: str
    base_url: str = Field(default="https://api.siliconflow.cn/v1/embeddings")
    model_name: str = Field(default="BAAI/bge-large-zh-v1.5")

    def __init__(self, key: str, base_url: str, model_name: str) -> None:
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.model_name = model_name
        self.base_url = base_url

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str, encoding_format: str = "float") -> list[float]:
        """
        获取文本的嵌入向量
        :param text: 输入的文本
        :param encoding_format: 编码格式，默认为 "float"
        :return:
        """
        payload = {"model": self.model_name, "input": text, "encoding_format": encoding_format}
        try:
            response = requests.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("data", [])[0].get("embedding", [])
            if not embedding:
                logger.error("Error: Invalid response from the API. No 'embedding' found in the response.")
                return []
            return list(map(float, embedding))
        except requests.exceptions.RequestException as e:
            logger.error(f"Error: {str(e)}")
            return []


class OpenAIEmbedding:
    """使用Langchain的OpenAI嵌入模型"""

    key: str
    model_name: str = Field(default="text-embedding-ada-002")
    base_url: str = Field(default="https://api.openai.com/v1")

    def __init__(self, key: str, model_name: str, base_url: str) -> None:
        self.embeddings = OpenAIEmbeddings(model=model_name, api_key=SecretStr(key), base_url=base_url)
        logger.info("OpenAIEmbedding initialized", model=model_name, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents with Langchain OpenAI: {str(e)}")
            return [[] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """
        嵌入单个查询文本
        :param text: 查询文本
        :return: 嵌入向量
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query with Langchain OpenAI: {str(e)}")
            return []
