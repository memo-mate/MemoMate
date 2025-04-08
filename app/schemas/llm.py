from sqlmodel import Field, SQLModel


class ChatRequest(SQLModel):
    """聊天请求"""

    message: str = Field(default=..., description="聊天消息")
    history: list[tuple[str, str]] = Field(default_factory=list, description="聊天历史")


class ChatResponse(SQLModel):
    """聊天响应"""

    message: str = Field(default=..., description="聊天消息")
    history: list[tuple[str, str]] = Field(default_factory=list, description="聊天历史")
    session_id: str = Field(default=None, description="会话ID")


class RAGChatRequest(SQLModel):
    """RAG聊天请求"""

    message: str = Field(default=..., description="聊天消息")
    history: list[tuple[str, str]] = Field(default_factory=list, description="聊天历史")
    retrieve_top_k: int = Field(default=5, description="检索文档数量")
    use_history: bool = Field(default=True, description="是否使用历史记录")
    session_id: str | None = Field(default=None, description="会话ID")
