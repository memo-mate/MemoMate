import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core import logger
from app.core.config import settings


class TxtParser:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def __call__(self, *args: Any, **kwds: Any) -> list[Document]:
        """处理文档并返回分块后的文档列表"""
        return self.parse()

    def parse(self) -> list[Document]:
        """
        解析文本文件并返回文档块列表

        Returns:
            list[Document]: 分割后的文档块列表
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"文件不存在: {self.file_path}")

            # 读取文件内容
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # 创建基础元数据
            metadata = {
                "source": str(self.file_path),
                "file_name": Path(self.file_path).name,
                "file_type": "txt",
                "created_at": os.path.getctime(self.file_path),
                "modified_at": os.path.getmtime(self.file_path),
            }

            # 如果文本为空，返回空列表
            if not text.strip():
                logger.warning("文件内容为空", file_path=self.file_path)
                return []

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
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_count"] = len(chunks)

            logger.info(
                "文本解析完成",
                file_path=self.file_path,
                chunk_count=len(chunks),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            return chunks

        except UnicodeDecodeError:
            # 尝试使用其他编码
            encodings = ["gbk", "gb2312", "big5", "latin1"]
            for encoding in encodings:
                try:
                    with open(self.file_path, "r", encoding=encoding) as f:
                        text = f.read()
                    logger.info(f"使用 {encoding} 编码成功读取文件", file_path=self.file_path)
                    return self.parse()  # 递归调用以处理成功读取的内容
                except UnicodeDecodeError:
                    continue
            
            # 如果所有编码都失败
            logger.error("无法解码文件内容", file_path=self.file_path)
            raise

        except Exception as e:
            logger.exception("解析文本文件失败", exc_info=e, file_path=self.file_path)
            raise
