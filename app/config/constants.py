import os

# 数据库配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "113.45.191.109"),
    "port": os.getenv("DB_PORT", "53000"),
    "user": os.getenv("DB_USER", "memo"),
    "password": os.getenv("DB_PASSWORD", "heKHemO6_pM0Y81sOkPMpj5nAM_eHLX62fXC36M3M1w"),
    "database": os.getenv("DB_NAME", "app")
}

# Redis配置
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "password": os.getenv("REDIS_PASSWORD", None),
    "db": int(os.getenv("REDIS_DB", 0)),
    "decode_responses": True,  # 自动解码响应
    "socket_timeout": 5,  # 连接超时时间
    "retry_on_timeout": True,  # 超时时重试
    "max_connections": 10  # 连接池最大连接数
}

# 数据库连接池配置
DB_POOL_CONFIG = {
    "pool_size": 5,  # 连接池大小
    "max_overflow": 10,  # 超过pool_size后最多可以创建的连接数
    "pool_timeout": 30,  # 连接池获取连接的超时时间
    "pool_recycle": 1800,  # 连接在连接池中重复使用的时间间隔
    "pool_pre_ping": True  # 每次连接前ping一下数据库，确保连接可用
}

# 构建数据库URL
DATABASE_URL = f"mysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# JWT配置
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-keep-it-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
