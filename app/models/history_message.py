import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import TEXT
from sqlmodel import Field, Relationship, SQLModel

from app.enums import HistoryMessageType

# from app.models.user import User
if TYPE_CHECKING:
    from app.models.user import User


class HistoryMessage(SQLModel, table=True):
    __tablename__ = "history_message"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, description="消息ID")
    message: str = Field(sa_column=Column(TEXT, nullable=False), description="消息内容")
    message_type: HistoryMessageType = Field(default=..., description="消息类型")
    session_id: str = Field(nullable=False, description="会话ID")
    user_id: uuid.UUID = Field(default=None, foreign_key="user.id", ondelete="CASCADE", description="用户ID")
    user: "User" = Relationship(back_populates="history_messages")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
