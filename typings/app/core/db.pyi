# typings/app/core/db.pyi

from sqlalchemy import Engine
from sqlmodel import Session

engine: Engine

def init_db(session: Session) -> None: ...
