from typing import Any

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """嵌入请求"""

    texts: list[str] = Field(..., description="待嵌入的文本列表", min_length=1)
    model: str | None = Field(None, description="模型名称")
    is_query: bool = Field(False, description="是否为查询嵌入（使用不同的嵌入策略）")


class EmbedResponse(BaseModel):
    """嵌入响应"""

    vectors: list[list[float]] = Field(..., description="嵌入向量列表")
    model: str = Field(..., description="使用的模型名称")


class VectorMetadata(BaseModel):
    """向量元数据"""

    key: str = Field(..., description="元数据键")
    value: Any = Field(..., description="元数据值")


class AddVectorRequest(BaseModel):
    """添加向量请求"""

    collection_name: str = Field(..., description="集合名称")
    text: str = Field(..., description="文本内容")
    metadata: list[VectorMetadata] | None = Field(None, description="元数据")
    id: str | None = Field(None, description="向量ID，不指定则自动生成")


class AddVectorResponse(BaseModel):
    """添加向量响应"""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="操作结果信息")
    vector_id: str | None = Field(None, description="向量ID")


class BatchAddVectorsRequest(BaseModel):
    """批量添加向量请求"""

    collection_name: str = Field(..., description="集合名称")
    texts: list[str] = Field(..., description="文本内容列表")
    metadatas: list[dict[str, Any]] | None = Field(None, description="元数据列表")
    ids: list[str] | None = Field(None, description="向量ID列表，不指定则自动生成")


class BatchAddVectorsResponse(BaseModel):
    """批量添加向量响应"""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="操作结果信息")
    vector_ids: list[str] = Field(..., description="向量ID列表")


class DeleteVectorRequest(BaseModel):
    """删除向量请求"""

    collection_name: str = Field(..., description="集合名称")
    vector_ids: list[str] = Field(..., description="向量ID列表")


class DeleteVectorResponse(BaseModel):
    """删除向量响应"""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="操作结果信息")


class SearchVectorRequest(BaseModel):
    """向量搜索请求"""

    collection_name: str = Field(..., description="集合名称")
    query: str = Field(..., description="搜索查询文本")
    top_k: int = Field(4, description="返回结果数量")
    search_type: str = Field("similarity", description="搜索类型：similarity, mmr")
    filter: dict[str, Any] | None = Field(None, description="过滤条件")


class DocumentModel(BaseModel):
    """文档模型"""

    page_content: str = Field(..., description="文档内容")
    metadata: dict[str, Any] = Field(..., description="文档元数据")
    score: float | None = Field(None, description="相似度得分")


class SearchVectorResponse(BaseModel):
    """向量搜索响应"""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="操作结果信息")
    results: list[DocumentModel] = Field(..., description="搜索结果")


class CollectionListResponse(BaseModel):
    """集合列表响应"""

    success: bool = Field(..., description="是否成功")
    collections: list[str] = Field(..., description="集合列表")


class CreateCollectionRequest(BaseModel):
    """创建集合请求"""

    collection_name: str = Field(..., description="集合名称")
    force: bool = Field(False, description="如果集合已存在，是否强制创建")


class CreateCollectionResponse(BaseModel):
    """创建集合响应"""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="操作结果信息")


class DeleteCollectionRequest(BaseModel):
    """删除集合请求"""

    collection_name: str = Field(..., description="集合名称")


class DeleteCollectionResponse(BaseModel):
    """删除集合响应"""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="操作结果信息")
