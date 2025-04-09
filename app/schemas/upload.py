from pydantic import BaseModel, computed_field
from sqlmodel import Field, SQLModel


class UploadCreateParams(SQLModel):
    """大文件上传数据库创建参数"""

    file_name: str = Field(..., max_length=255, description="原始文件名")
    total_size: int = Field(description="文件总大小(字节)")
    chunk_size: int = Field(
        default=5 * 1024 * 1024,
        description="分块大小(字节)",
    )

    @computed_field
    def total_chunks(self) -> int:
        """计算总分块数"""
        return (self.total_size + self.chunk_size - 1) // self.chunk_size


class UploadChunkParams(SQLModel):
    """分块上传参数"""

    upload_id: str = Field(..., description="关联的上传任务ID")
    chunk_index: int = Field(..., description="分块序号(从0开始)")
    chunk_data: bytes = Field(..., description="分块二进制数据")


class UploadStatusRequest(BaseModel):
    """上传状态查询请求参数"""

    file_name: str = Field(..., description="文件名")
    user_id: str = Field(..., description="用户ID")
    file_size: int = Field(..., description="文件大小")


class UploadStatusResponse(BaseModel):
    """上传状态查询响应数据"""

    status: str = Field(..., description="上传状态")
    upload_id: str | None = Field(None, description="上传任务ID")
    message: str | None = Field(None, description="上传状态信息")
    chunk_list: list[int] = Field(default_factory=list, description="已上传的分块索引列表")
    file_size: int = Field(default=0, description="文件大小")
    chunk_size: int = Field(default=0, description="分块大小")
    total_chunks: int = Field(default=0, description="总分块数量")


class UploadChunkRequest(BaseModel):
    """分块上传请求参数（表单参数）"""

    upload_id: str = Field(..., description="上传任务ID")
    chunk_index: int = Field(..., description="分块序号(从0开始)")


class UploadChunkResponse(BaseModel):
    """分块上传响应数据"""

    success: bool = Field(..., description="上传是否成功")
    message: str = Field(..., description="上传结果信息")
    upload_id: str = Field(..., description="上传任务ID")
    chunk_index: int = Field(..., description="当前上传的分块序号")


class MergeChunksRequest(BaseModel):
    """合并分块请求参数"""

    file_name: str = Field(..., description="文件名")
    upload_id: str = Field(..., description="上传任务ID")


class MergeChunksResponse(BaseModel):
    """合并分块响应数据"""

    success: bool = Field(..., description="合并是否成功")
    message: str = Field(..., description="合并结果信息")
    upload_id: str = Field(..., description="上传任务ID")
    status: str = Field(..., description="文件状态")
    file_path: str | None = Field(None, description="合并后的文件路径")


class DeleteUploadResponse(BaseModel):
    """取消上传响应数据"""

    success: bool = Field(..., description="取消上传是否成功")
    message: str = Field(..., description="操作结果信息")
    upload_id: str = Field(..., description="上传任务ID")
