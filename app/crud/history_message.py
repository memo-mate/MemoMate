from sqlmodel import Session, delete, select

from app.enums import HistoryMessageType
from app.models.history_message import HistoryMessage


def add_history_message(*, session: Session, message: str, message_type: HistoryMessageType, session_id: str) -> None:
    """
    添加历史消息
    """
    history_message = HistoryMessage(message=message, message_type=message_type, session_id=session_id)
    session.add(history_message)
    session.commit()


def get_history_messages(*, session: Session, session_id: str) -> list[HistoryMessage]:
    """
    获取历史消息
    """
    statement = select(HistoryMessage).where(HistoryMessage.session_id == session_id)
    return session.exec(statement).all()


def delete_history_messages(*, session: Session, session_id: str) -> None:
    """
    删除历史消息
    """
    statement = delete(HistoryMessage).where(HistoryMessage.session_id == session_id)
    session.exec(statement)
    session.commit()


def get_history_message_session_list(*, session: Session) -> list[HistoryMessage]:
    """
    获取历史消息会话列表

    检索每个不同session id的最早创建的历史消息（PostgreSQL使用 DISTINCT ON 优化）
    """
    statement = (
        select(HistoryMessage)
        .distinct(HistoryMessage.session_id)  # 只选每个 session_id 的第一条记录
        .order_by(HistoryMessage.session_id, HistoryMessage.created_at)  # 先按 session_id 排序，再按创建时间排序
    )

    return session.exec(statement).all()
