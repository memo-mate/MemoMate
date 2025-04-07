import datetime
import uuid

from sqlmodel import Field, Relationship, SQLModel

from app.enums.upload import FileUploadState

"""
大文件上传
"""


class FileMatedata(SQLModel, table=True):
    __tablename__ = "file_matedata"

    upload_id: str = Field(primary_key=True, default_factory=lambda: str(uuid.uuid4()), index=True)
    file_name: str = Field(max_length=255, description="原始文件名")
    total_size: int = Field(description="文件总大小（字节）")
    chunk_size: int = Field(description="分块大小（默认5MB")
    total_chunks: int = Field(description="总分块数（计算值：CEIL(total_size/chunk_size))")
    status: FileUploadState = Field(default=FileUploadState.pending, description="状态")
    uploader_id: str | None = Field(default=None, description="关联用户系统的上传者ID")
    storage_path: str | None = Field(default=None, max_length=500, description="最终存储路径（合并完成后更新）")
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now, description="任务创建时间（默认now()）"
    )
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now, description="最后更新时间（自动更新）")
    chunks: list["UploadChunk"] = Relationship(back_populates="upload_file")

    def __repr__(self) -> str:
        return f"Upload(upload_id={self.upload_id}, file_name={self.file_name}, status={self.status}, uploader_id={self.uploader_id})"

    def get_chunks(self, only_index: bool = False) -> list["UploadChunk"] | list[int]:
        """
        获取所有与此上传文件关联的分块
        """
        if only_index:
            return [chunk.chunk_index for chunk in self.chunks]
        return self.chunks


class UploadChunk(SQLModel, table=True):
    __tablename__ = "upload_chunks"

    id: int = Field(primary_key=True)
    upload_id: str = Field(foreign_key="file_matedata.upload_id", description="关联的上传任务ID")
    chunk_index: int = Field(description="分块序号（从0开始）")
    uploaded_size: int = Field(description="实际接收字节数（用于异常检测）")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now, description="分块上传时间")
    upload_file: FileMatedata = Relationship(back_populates="chunks")
