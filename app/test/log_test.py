from app.config.log_config import get_logger
import time
from datetime import datetime

# 获取不同模块的logger
default_logger = get_logger()  # 使用默认logger
test_logger = get_logger(__name__)  # 使用当前模块名称的logger
custom_logger = get_logger("custom")  # 使用自定义名称的logger

def test_different_log_levels():
    """测试不同日志级别"""
    test_logger.debug("这是一条调试日志")  # 默认不会显示，因为日志级别是INFO
    test_logger.info("这是一条信息日志")
    test_logger.warning("这是一条警告日志")
    test_logger.error("这是一条错误日志")
    test_logger.critical("这是一条严重错误日志")

def test_log_with_variables():
    """测试带变量的日志"""
    user_id = 123
    action = "login"
    test_logger.info(f"用户 {user_id} 执行了 {action} 操作")

def test_log_with_exception():
    """测试异常日志"""
    try:
        result = 1 / 0
    except Exception as e:
        test_logger.error(f"发生异常: {str(e)}", exc_info=True)

def test_log_with_context():
    """测试带上下文信息的日志"""
    request_id = "req-001"
    user_agent = "Mozilla/5.0"
    ip = "192.168.1.1"
    
    test_logger.info(
        "收到API请求",
        extra={
            "request_id": request_id,
            "user_agent": user_agent,
            "ip": ip
        }
    )

def test_performance_logging():
    """测试性能日志"""
    start_time = time.time()
    
    # 模拟一些操作
    time.sleep(1)
    
    end_time = time.time()
    duration = end_time - start_time
    test_logger.info(f"操作耗时: {duration:.2f}秒")

def test_structured_logging():
    """测试结构化日志"""
    log_data = {
        "event": "user_action",
        "user_id": 123,
        "action": "purchase",
        "item_id": "item-456",
        "amount": 99.99,
        "timestamp": datetime.now().isoformat()
    }
    test_logger.info(f"用户行为: {log_data}")

def main():
    """运行所有测试"""
    print("开始测试日志功能...")
    
    test_different_log_levels()
    print("\n" + "="*50 + "\n")
    
    test_log_with_variables()
    print("\n" + "="*50 + "\n")
    
    test_log_with_exception()
    print("\n" + "="*50 + "\n")
    
    test_log_with_context()
    print("\n" + "="*50 + "\n")
    
    test_performance_logging()
    print("\n" + "="*50 + "\n")
    
    test_structured_logging()
    print("\n" + "="*50 + "\n")
    
    print("日志测试完成！")

if __name__ == "__main__":
    main()
