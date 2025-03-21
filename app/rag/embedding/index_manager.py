import shutil
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.rag.embedding.document_processor import DirectoryProcessor, DocumentProcessor
from app.rag.embedding.vector_search import HuggingFaceEmbeddings
from app.rag.embedding.vector_store import QdrantStore


class IndexManager(BaseModel):
    """索引管理器"""

    vector_store_path: str = Field(default=...)
    embedding_model_path: str = Field(default="BAAI/bge-large-zh-v1.5")
    collection_name: str = Field(default="documents")
    embeddings: HuggingFaceEmbeddings = None
    vector_store: QdrantStore = None

    def __init__(self, **kwargs):
        """初始化IndexManager"""
        super().__init__(**kwargs)

        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_path)

        # 初始化向量存储
        self.vector_store = QdrantStore(
            path=self.vector_store_path,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )

    def create_index(self, documents: list[Document]) -> dict[str, Any]:
        """创建索引"""
        start_time = time.time()

        # 添加文档到向量存储
        ids = self.vector_store.add_documents(documents)

        end_time = time.time()

        # 返回索引信息
        return {
            "status": "success",
            "document_count": len(documents),
            "ids": ids,
            "collection_name": self.collection_name,
            "time_taken": end_time - start_time,
        }

    def index_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """索引目录"""
        # 初始化目录处理器
        directory_processor = DirectoryProcessor()

        # 处理目录
        documents = directory_processor.process_directory(
            directory_path=directory_path,
            recursive=recursive,
            metadata=metadata,
        )

        # 创建索引
        return self.create_index(documents)

    def index_file(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """索引文件"""
        # 初始化文档处理器
        document_processor = DocumentProcessor()

        # 处理文件
        documents = document_processor.process_file(
            file_path=file_path,
            metadata=metadata,
        )

        # 创建索引
        return self.create_index(documents)

    def index_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """索引文本"""
        # 初始化文档处理器
        document_processor = DocumentProcessor()

        # 处理文本
        documents = document_processor.split_text(
            text=text,
            metadata=metadata,
        )

        # 创建索引
        return self.create_index(documents)

    def delete_index(self) -> dict[str, Any]:
        """删除索引"""
        # 删除集合
        self.vector_store.delete_collection()

        return {
            "status": "success",
            "message": f"已删除集合: {self.collection_name}",
        }

    def backup_index(self, backup_dir: str | Path) -> dict[str, Any]:
        """备份索引"""
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        # 备份向量存储
        source_path = Path(self.vector_store_path)
        target_path = backup_dir / f"{self.collection_name}_{int(time.time())}"

        # 复制目录
        shutil.copytree(source_path, target_path)

        return {
            "status": "success",
            "message": f"已备份集合: {self.collection_name}",
            "backup_path": str(target_path),
        }

    def restore_index(self, backup_path: str | Path) -> dict[str, Any]:
        """恢复索引"""
        backup_path = Path(backup_path)

        if not backup_path.exists():
            return {
                "status": "error",
                "message": f"备份路径不存在: {backup_path}",
            }

        # 删除现有向量存储
        target_path = Path(self.vector_store_path)
        if target_path.exists():
            shutil.rmtree(target_path)

        # 复制备份
        shutil.copytree(backup_path, target_path)

        return {
            "status": "success",
            "message": f"已恢复集合: {self.collection_name}",
            "restore_path": str(target_path),
        }

    def get_index_stats(self) -> dict[str, Any]:
        """获取索引统计信息"""
        try:
            # 获取集合信息
            collection_info = self.vector_store.client.get_collection(self.collection_name)

            # 获取点数量
            points_count = self.vector_store.client.count(collection_name=self.collection_name).count

            return {
                "status": "success",
                "collection_name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "points_count": points_count,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"获取索引统计信息失败: {str(e)}",
            }
