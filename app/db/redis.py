from redis import Redis, ConnectionPool
from typing import Optional, Any, Union
from app.config.constants import REDIS_CONFIG
from app.config.log_config import get_logger
import json
from contextlib import contextmanager

logger = get_logger(__name__)

class RedisClient:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            # 创建连接池
            cls._pool = ConnectionPool(**REDIS_CONFIG)
        return cls._instance

    def __init__(self):
        self.redis_client = Redis(connection_pool=self._pool)

    @contextmanager
    def get_connection(self):
        """获取Redis连接的上下文管理器"""
        try:
            yield self.redis_client
        except Exception as e:
            logger.error(f"Redis operation error: {str(e)}")
            raise

    def set_value(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """
        设置键值对
        :param key: 键
        :param value: 值
        :param expire: 过期时间（秒）
        :return: 是否成功
        """
        try:
            with self.get_connection() as redis:
                if not isinstance(value, (str, int, float, bool)):
                    value = json.dumps(value)
                return redis.set(key, value, ex=expire)
        except Exception as e:
            logger.error(f"Error setting key {key}: {str(e)}")
            return False

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        获取值
        :param key: 键
        :param default: 默认值
        :return: 值
        """
        try:
            with self.get_connection() as redis:
                value = redis.get(key)
                if value is None:
                    return default
                try:
                    return json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    return value
        except Exception as e:
            logger.error(f"Error getting key {key}: {str(e)}")
            return default

    def delete_key(self, key: str) -> bool:
        """
        删除键
        :param key: 键
        :return: 是否成功
        """
        try:
            with self.get_connection() as redis:
                return bool(redis.delete(key))
        except Exception as e:
            logger.error(f"Error deleting key {key}: {str(e)}")
            return False

    def set_with_lock(self, key: str, value: Any, expire: int = 300) -> bool:
        """
        使用分布式锁设置值
        :param key: 键
        :param value: 值
        :param expire: 过期时间（秒）
        :return: 是否成功
        """
        lock_key = f"lock:{key}"
        try:
            with self.get_connection() as redis:
                # 尝试获取锁
                if redis.set(lock_key, "1", ex=5, nx=True):
                    try:
                        return self.set_value(key, value, expire)
                    finally:
                        redis.delete(lock_key)
                return False
        except Exception as e:
            logger.error(f"Error setting with lock for key {key}: {str(e)}")
            return False

    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        增加计数
        :param key: 键
        :param amount: 增加量
        :return: 新值
        """
        try:
            with self.get_connection() as redis:
                return redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing key {key}: {str(e)}")
            return None

    def expire_at(self, key: str, timestamp: int) -> bool:
        """
        设置过期时间
        :param key: 键
        :param timestamp: UNIX时间戳
        :return: 是否成功
        """
        try:
            with self.get_connection() as redis:
                return redis.expireat(key, timestamp)
        except Exception as e:
            logger.error(f"Error setting expiry for key {key}: {str(e)}")
            return False

    def hset(self, name: str, key: str, value: Any) -> bool:
        """
        设置哈希表字段的值
        :param name: 哈希表名称
        :param key: 字段名
        :param value: 值
        :return: 是否成功
        """
        try:
            with self.get_connection() as redis:
                if not isinstance(value, (str, int, float, bool)):
                    value = json.dumps(value)
                return redis.hset(name, key, value)
        except Exception as e:
            logger.error(f"Error setting hash field {name}:{key}: {str(e)}")
            return False

    def hget(self, name: str, key: str, default: Any = None) -> Any:
        """
        获取哈希表字段的值
        :param name: 哈希表名称
        :param key: 字段名
        :param default: 默认值
        :return: 值
        """
        try:
            with self.get_connection() as redis:
                value = redis.hget(name, key)
                if value is None:
                    return default
                try:
                    return json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    return value
        except Exception as e:
            logger.error(f"Error getting hash field {name}:{key}: {str(e)}")
            return default

    def hgetall(self, name: str) -> dict:
        """
        获取哈希表中所有字段和值
        :param name: 哈希表名称
        :return: 字段和值的字典
        """
        try:
            with self.get_connection() as redis:
                result = redis.hgetall(name)
                return {
                    k: self._decode_value(v)
                    for k, v in result.items()
                }
        except Exception as e:
            logger.error(f"Error getting all hash fields for {name}: {str(e)}")
            return {}

    def hmset(self, name: str, mapping: dict) -> bool:
        """
        批量设置哈希表字段的值
        :param name: 哈希表名称
        :param mapping: 字段和值的字典
        :return: 是否成功
        """
        try:
            with self.get_connection() as redis:
                encoded_mapping = {
                    k: json.dumps(v) if not isinstance(v, (str, int, float, bool)) else v
                    for k, v in mapping.items()
                }
                return redis.hmset(name, encoded_mapping)
        except Exception as e:
            logger.error(f"Error setting multiple hash fields for {name}: {str(e)}")
            return False

    def hdel(self, name: str, *keys: str) -> int:
        """
        删除哈希表中的一个或多个字段
        :param name: 哈希表名称
        :param keys: 要删除的字段
        :return: 成功删除的字段数量
        """
        try:
            with self.get_connection() as redis:
                return redis.hdel(name, *keys)
        except Exception as e:
            logger.error(f"Error deleting hash fields from {name}: {str(e)}")
            return 0

    def _decode_value(self, value: Any) -> Any:
        """解码值，尝试 JSON 解析"""
        if value is None:
            return None
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value

# 创建全局Redis客户端实例
redis_client = RedisClient()
