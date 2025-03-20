"""重排序器测试模块"""

import logging

from app.core.log_adapter import logger

# 设置日志级别为DEBUG
logger.setLevel(logging.DEBUG)
# 添加控制台处理器
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# ... existing code ...
