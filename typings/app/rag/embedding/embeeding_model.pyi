# typings/app/rag/embedding/__init__.pyi

from langchain_core.embeddings import Embeddings

from app.enums import EmbeddingDriverEnum

class MemoMateEmbeddings:
    def local_embedding(self, model_name: str, driver: EmbeddingDriverEnum, normalize: bool) -> Embeddings: ...
    def openai_embedding(self, api_key: str, base_url: str, model_name: str, normalize: bool) -> Embeddings: ...
