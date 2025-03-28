from sqlmodel import Field, SQLModel


class ChatRequest(SQLModel):
    """聊天请求"""

    message: str = Field(default=..., description="聊天消息")
    history: list[tuple[str, str]] = Field(default_factory=list, description="聊天历史")


class ChatResponse(SQLModel):
    """聊天响应"""

    message: str = Field(default=..., description="聊天消息")
    history: list[tuple[str, str]] = Field(default_factory=list, description="聊天历史")
