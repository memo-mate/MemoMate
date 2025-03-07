from app.core.config import DB_PATH
import lancedb

db = lancedb.connect(DB_PATH)
print(db.table_names())
print(
    f"数据库包含 {len(db.table_names())} 个表，总记录数：{sum([db.open_table(name).count_rows() for name in db.table_names()])}"
)
