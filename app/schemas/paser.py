from pydantic import computed_field
from sqlmodel import Field, SQLModel

from app.enums.task import DocumentFileTaskType, FileParsingTaskState


class FileParsingTaskUploadParams(SQLModel):
    """文件解析任务上传参数"""

    file_name: str | None = Field(default=None, max_length=255)
    title: str | None = Field(default=None, max_length=255)


class FileParsingTaskCreate(SQLModel):
    """文件解析任务创建"""

    title: str | None = Field(default=None, max_length=255)
    file_path: str = Field(max_length=255)
    file_name: str = Field(max_length=255)
    file_size: int = Field(default=0)
    file_md5: str = Field(max_length=255)
    status: FileParsingTaskState = Field(default=FileParsingTaskState.pedding)

    @computed_field
    def file_type(self) -> str:
        """获取文件类型，从文件名中提取后缀"""
        if not self.file_path:
            raise ValueError("file_path is required")
        return DocumentFileTaskType.get_file_type(self.file_path)
