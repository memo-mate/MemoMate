from typing import Any

from fastapi import APIRouter, Depends

from app import crud, schemas
from app.api.deps import CurrentUser, SessionDep, get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("/sessions", response_model=list[schemas.HistoryMessageSession], description="获取会话列表")
def get_history_message_session_list(current_user: CurrentUser, session: SessionDep) -> Any:
    ses_list = crud.get_history_message_session_list(session=session, user_id=current_user.id)
    items = [
        schemas.HistoryMessageSession(
            session_id=ses.session_id,
            session_name=ses.message,
            created_at=ses.created_at,
        )
        for ses in ses_list
    ]
    return items


@router.delete("/sessions/{session_id}", description="删除会话")
def delete_history_session(session_id: str) -> Any:
    crud.delete_history_messages(session_id=session_id)


@router.get("/messages/{session_id}", response_model=list[schemas.HistoryMessage], description="获取会话消息")
def get_history_messages(session_id: str, session: SessionDep) -> Any:
    messages = crud.get_history_messages(session=session, session_id=session_id)
    return messages
