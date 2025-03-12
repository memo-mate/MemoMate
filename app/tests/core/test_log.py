from app.core import logger


def test_log_adapter():
    # 测试日志适配器

    # 普通日志
    logger.debug("test")
    logger.info("test")
    logger.warning("test")
    logger.error("test")
    logger.critical("test")

    # 异常日志
    # 高亮打印错误堆栈
    # 打印运行时变量kv
    try:
        raise Exception("test")
    except Exception as e:
        logger.exception("test", exc_info=e)

    # 打印运行时变量kv
    logger.info("test", a=1, b=2)

    # 打印运行时变量字典
    test_dict = {"a": 1, "b": 2}
    logger.info("test", val=test_dict)

    # 打印运行时变量列表
    test_list = [1, 2, 3]
    logger.info("test", val=test_list)
