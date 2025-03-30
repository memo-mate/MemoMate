import datetime

from pydantic import field_validator
from sqlmodel import Field, SQLModel

from app.enums import HistoryMessageType


class HistoryMessageSession(SQLModel):
    session_id: str = Field(description="会话ID")
    session_name: str = Field(description="会话名称")
    created_at: datetime.datetime = Field(description="创建时间")

    @field_validator("session_name")
    def validate_session_name(cls, v):
        if len(v) > 15:
            return v[:15]
        return v


class HistoryMessage(SQLModel):
    id: int | None = Field(description="消息ID")
    message: str = Field(description="消息内容")
    message_type: HistoryMessageType = Field(description="消息类型")
    session_id: str = Field(description="会话ID")
    created_at: datetime.datetime = Field(description="创建时间")
