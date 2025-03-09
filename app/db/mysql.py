from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from app.config.constants import DATABASE_URL, DB_POOL_CONFIG
from app.config.log_config import get_logger

logger = get_logger(__name__)

# 创建引擎
engine = create_engine(
    DATABASE_URL,
    **DB_POOL_CONFIG
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明基类
Base = declarative_base()

@contextmanager
def get_db_session() -> Session:
    """
    数据库会话上下文管理器
    使用方法:
    with get_db_session() as session:
        session.query(...)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        session.close()

def get_db():
    """
    FastAPI依赖注入使用的数据库会话生成器
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
