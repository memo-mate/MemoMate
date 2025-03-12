from .config import settings
from .log_adapter import logger, setup_logging

__all__ = ["settings", "logger"]

# 初始化日志库
setup_logging()
