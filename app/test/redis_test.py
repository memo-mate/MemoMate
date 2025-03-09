from app.db.redis import redis_client
import time
from datetime import datetime
import json

def test_basic_operations():
    """测试基本的键值操作"""
    print("\n测试基本的键值操作:")
    
    # 设置值
    success = redis_client.set_value("test_key", "test_value", expire=60)
    print(f"设置键值对: {'成功' if success else '失败'}")
    
    # 获取值
    value = redis_client.get_value("test_key")
    print(f"获取值: {value}")
    
    # 删除键
    deleted = redis_client.delete_key("test_key")
    print(f"删除键: {'成功' if deleted else '失败'}")

def test_complex_data():
    """测试复杂数据类型的存储"""
    print("\n测试复杂数据类型:")
    
    test_data = {
        "name": "测试用户",
        "age": 25,
        "scores": [85, 92, 78],
        "info": {
            "city": "北京",
            "occupation": "程序员"
        }
    }
    
    # 存储复杂数据
    success = redis_client.set_value("user:1", test_data)
    print(f"存储复杂数据: {'成功' if success else '失败'}")
    
    # 读取复杂数据
    retrieved_data = redis_client.get_value("user:1")
    print(f"读取的数据: {json.dumps(retrieved_data, ensure_ascii=False, indent=2)}")

def test_hash_operations():
    """测试哈希表操作"""
    print("\n测试哈希表操作:")
    
    # 设置单个哈希字段
    redis_client.hset("user:profile:1", "name", "张三")
    redis_client.hset("user:profile:1", "age", 25)
    redis_client.hset("user:profile:1", "data", {"city": "上海", "job": "工程师"})
    
    # 获取单个字段
    name = redis_client.hget("user:profile:1", "name")
    print(f"获取名字: {name}")
    
    # 获取复杂字段
    data = redis_client.hget("user:profile:1", "data")
    print(f"获取复杂数据: {data}")
    
    # 批量设置字段
    user_data = {
        "email": "zhangsan@example.com",
        "phone": "13800138000",
        "address": {"city": "上海", "street": "南京路"}
    }
    redis_client.hmset("user:profile:1", user_data)
    
    # 获取所有字段
    all_data = redis_client.hgetall("user:profile:1")
    print(f"所有数据: {json.dumps(all_data, ensure_ascii=False, indent=2)}")
    
    # 删除字段
    deleted = redis_client.hdel("user:profile:1", "phone", "address")
    print(f"删除的字段数: {deleted}")

def test_distributed_lock():
    """测试分布式锁"""
    print("\n测试分布式锁:")
    
    # 使用锁设置值
    success1 = redis_client.set_with_lock("locked_key", "value1")
    print(f"第一次获取锁并设置值: {'成功' if success1 else '失败'}")
    
    # 立即尝试再次获取锁
    success2 = redis_client.set_with_lock("locked_key", "value2")
    print(f"第二次获取锁: {'成功' if success2 else '失败'}")
    
    # 等待锁释放后再试
    time.sleep(5)
    success3 = redis_client.set_with_lock("locked_key", "value3")
    print(f"等待后获取锁: {'成功' if success3 else '失败'}")

def test_counter():
    """测试计数器"""
    print("\n测试计数器:")
    
    # 增加计数
    count1 = redis_client.increment("visitor_count")
    print(f"第一次增加后的计数: {count1}")
    
    # 增加指定数量
    count2 = redis_client.increment("visitor_count", 5)
    print(f"增加5后的计数: {count2}")

def main():
    """运行所有测试"""
    print("开始Redis功能测试...")
    
    test_basic_operations()
    print("\n" + "="*50)
    
    test_complex_data()
    print("\n" + "="*50)
    
    test_hash_operations()
    print("\n" + "="*50)
    
    test_distributed_lock()
    print("\n" + "="*50)
    
    test_counter()
    print("\n" + "="*50)
    
    print("\nRedis测试完成！")

if __name__ == "__main__":
    main()
