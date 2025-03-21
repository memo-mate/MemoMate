import hashlib
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.core.config import settings


class DocumentProcessor(BaseModel):
    """文档处理器"""

    chunk_size: int = Field(default=settings.CHUNK_SIZE)
    chunk_overlap: int = Field(default=settings.CHUNK_OVERLAP)

    def split_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """将文本分割成块"""
        if metadata is None:
            metadata = {}

        # 创建文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # 分割文本
        chunks = text_splitter.create_documents([text], [metadata])

        # 添加额外元数据
        for i, chunk in enumerate(chunks):
            # 添加块索引
            chunk.metadata["chunk_index"] = i
            # 添加总块数
            chunk.metadata["chunk_count"] = len(chunks)
            # 添加文本哈希
            chunk.metadata["text_hash"] = self._hash_text(chunk.page_content)

        return chunks

    def process_file(self, file_path: str | Path, metadata: dict[str, Any] | None = None) -> list[Document]:
        """处理文件"""
        if metadata is None:
            metadata = {}

        file_path = Path(file_path)

        # 添加文件元数据
        file_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "filetype": file_path.suffix.lower().lstrip("."),
            "created_at": os.path.getctime(file_path),
            "modified_at": os.path.getmtime(file_path),
            **metadata,
        }

        # 读取文件内容
        text = self._read_file(file_path)

        # 分割文本
        return self.split_text(text, file_metadata)

    def _read_file(self, file_path: Path) -> str:
        """读取文件内容"""
        # 根据文件类型选择不同的读取方法
        suffix = file_path.suffix.lower()

        if suffix in [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv"]:
            # 文本文件
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        elif suffix in [".pdf"]:
            # PDF文件
            try:
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                return "\n\n".join([doc.page_content for doc in docs])
            except ImportError:
                raise ImportError("PyPDF包未安装，请使用 'pip install pypdf' 安装")
        elif suffix in [".docx", ".doc"]:
            # Word文件
            try:
                from langchain_community.document_loaders import Docx2txtLoader

                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
                return "\n\n".join([doc.page_content for doc in docs])
            except ImportError:
                raise ImportError("docx2txt包未安装，请使用 'pip install docx2txt' 安装")
        else:
            raise ValueError(f"不支持的文件类型: {suffix}")

    def _hash_text(self, text: str) -> str:
        """计算文本哈希值"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空白
        text = re.sub(r"\s+", " ", text).strip()
        # 移除特殊字符
        text = re.sub(r"[^\w\s\u4e00-\u9fff.,?!;:()\[\]{}-]", "", text)
        return text


class DirectoryProcessor(BaseModel):
    """目录处理器"""

    document_processor: DocumentProcessor = Field(default_factory=DocumentProcessor)
    include_extensions: list[str] = Field(default=[".txt", ".md", ".pdf", ".docx", ".doc"])
    exclude_patterns: list[str] = Field(default=["node_modules", "__pycache__", ".git", ".venv"])

    def process_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """处理目录中的所有文件"""
        if metadata is None:
            metadata = {}

        directory_path = Path(directory_path)

        # 添加目录元数据
        dir_metadata = {
            "source_dir": str(directory_path),
            "dir_name": directory_path.name,
            **metadata,
        }

        # 获取文件列表
        files = self._get_files(directory_path, recursive)

        # 处理所有文件
        all_documents = []
        for file_path in files:
            try:
                # 处理文件
                documents = self.document_processor.process_file(file_path, dir_metadata.copy())
                all_documents.extend(documents)
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
                continue

        return all_documents

    def _get_files(self, directory_path: Path, recursive: bool) -> list[Path]:
        """获取目录中的所有文件"""
        files = []

        # 遍历目录
        if recursive:
            for root, dirs, filenames in os.walk(directory_path):
                # 排除不需要的目录
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]

                for filename in filenames:
                    file_path = Path(root) / filename
                    # 检查文件扩展名
                    if file_path.suffix.lower() in self.include_extensions:
                        files.append(file_path)
        else:
            # 只处理当前目录
            for file_path in directory_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.include_extensions:
                    files.append(file_path)

        return files
