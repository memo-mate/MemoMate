import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

class LogConfig:
    def __init__(self):
        # 创建logs目录
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # 创建日志格式
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 初始化日志记录器
        self.app_logger = self._setup_logger('app', 'logs/app.log')
        self.access_logger = self._setup_logger('access', 'logs/access.log')
        self.error_logger = self._setup_logger('error', 'logs/error.log', level=logging.ERROR)

    def _setup_logger(self, name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # 文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.formatter)
        logger.addHandler(file_handler)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        logger.addHandler(console_handler)

        return logger

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """获取日志记录器"""
        if name:
            return logging.getLogger(f"app.{name}")
        return self.app_logger

# 创建全局日志配置实例
log_config = LogConfig()

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志记录器的便捷函数"""
    return log_config.get_logger(name) 