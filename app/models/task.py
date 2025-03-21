import datetime

from sqlmodel import Field, SQLModel

from app.enums.task import FileParsingTaskState

"""
提前分表，每个任务类型一个表
"""


class FileParsingTask(SQLModel):
    """文件解析任务"""

    __tablename__ = "file_parsing_task"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    title: str | None = Field(default=None, max_length=255)
    status: FileParsingTaskState = Field(default=FileParsingTaskState.pedding)
    file_path: str = Field(max_length=255)
    file_name: str = Field(max_length=255)
    file_size: int = Field(default=0)
    file_type: str = Field(max_length=255)
    file_md5: str = Field(max_length=255)

    def __repr__(self) -> str:
        return f"FileParsingTask(id={self.id}, title={self.title}, status={self.status})"
