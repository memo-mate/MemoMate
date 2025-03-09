from fastapi import FastAPI
from app.routers import auth
from app.database import engine
from app.models.login_model import Base
from app.config.log_config import get_logger
import uvicorn
from app.middleware.rate_limit import RateLimitMiddleware

logger = get_logger(__name__)

def create_app() -> FastAPI:
    # 创建数据库表
    Base.metadata.create_all(bind=engine)
    
    # 创建FastAPI应用
    app = FastAPI()
    
    # 添加速率限制中间件
    app.add_middleware(RateLimitMiddleware, max_requests=100, window_size=60)
    
    # 包含路由
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    
    @app.get("/")
    def read_root():
        logger.info("访问了根路径")
        return {"message": "Welcome to the login app"}

    @app.on_event("startup")
    async def startup_event():
        logger.info("应用启动")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("应用关闭")
        
    return app

def main():
    logger.info("正在启动应用...")
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # 在主函数中直接运行时，不需要reload
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
