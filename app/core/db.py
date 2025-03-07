import lancedb  # type: ignore
import pyarrow as pa  # 添加pyarrow导入  # noqa: F401
from lancedb.pydantic import LanceModel, Vector  # type: ignore
from app.core.config import DB_PATH, TABLE_NAME
from typing import Dict, Any, Optional, List, Union  # noqa: F401
from loguru import logger


# 定义向量表结构
class TextVectorModel(LanceModel):
    id: str
    vector: Vector(1024) = lancedb.vector(1024) 
    text: str
    source: str
    metadata: str = ""  # 将字典类型改回字符串类型


# 初始化数据库连接
def init_database():
    try:
        connection = lancedb.connect(DB_PATH)
        logger.info(f"成功连接到数据库: {DB_PATH}")

        if not connection.table_names():
            try:
                connection.create_table(
                    TABLE_NAME, schema=TextVectorModel, mode="overwrite"
                )
                logger.success(f"已创建新表: {TABLE_NAME}")
            except Exception as create_error:
                logger.error(f"创建表失败: {str(create_error)}")
                raise

        return connection
    except Exception as conn_error:
        logger.critical(f"数据库连接失败: {str(conn_error)}")
        raise
