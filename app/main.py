from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.core.config import settings
from app.core.log_adapter import logger
from app.core.responses import CustomORJSONResponse
from app.core.sessions import SessionFactory
from app.utils.aio_producer import AIOProducer
from app.utils.art_name import run_figlet_lolcat


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 初始化 Kafka Producer
    SessionFactory.aio_producer = AIOProducer({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})

    # 记录嵌入服务配置
    logger.info(
        "Embedding service configured",
        api_base=settings.EMBEDDING_API_BASE,
        model=settings.EMBEDDING_MODEL,
    )

    yield

    # 清理资源
    SessionFactory.get_aio_producer().close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    default_response_class=CustomORJSONResponse,
    lifespan=lifespan,
)

# Set all CORS enabled origins
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app.include_router(api_router, prefix=settings.API_V1_STR)
run_figlet_lolcat()
