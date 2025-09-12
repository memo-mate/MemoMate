import secrets
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AnyUrl,
    BeforeValidator,
    PostgresDsn,
    RedisDsn,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).parent.parent


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # 验证默认值是否正确
        validate_default=False,
        # 优先级：后面文件的配置会覆盖前面文件的配置
        env_file=[".env"],
        env_ignore_empty=True,
        env_file_encoding="utf-8",
        # 忽略未定义的配置
        extra="ignore",
    )
    API_V1_STR: str = ""
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    FRONTEND_HOST: str = "http://localhost:5666"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    BACKEND_CORS_ORIGINS: Annotated[list[AnyUrl] | str, BeforeValidator(parse_cors)] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [self.FRONTEND_HOST]

    PROJECT_NAME: str
    POSTGRES_SERVER: str
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql+psycopg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_SERVER,
            port=self.POSTGRES_PORT,
            path=self.POSTGRES_DB,
        )

    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def REDIS_URL(self) -> RedisDsn:
        return RedisDsn.build(
            scheme="redis",
            host=self.REDIS_HOST,
            port=self.REDIS_PORT,
            password=self.REDIS_PASSWORD,
        )

    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    SMTP_PORT: int = 587
    SMTP_HOST: str | None = None
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    # TODO: update type to EmailStr when sqlmodel supports it
    EMAILS_FROM_EMAIL: str | None = None
    EMAILS_FROM_NAME: str | None = None

    @model_validator(mode="after")
    def _set_default_emails_from(self) -> Self:
        if not self.EMAILS_FROM_NAME:
            self.EMAILS_FROM_NAME = self.PROJECT_NAME
        return self

    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48

    @computed_field  # type: ignore[prop-decorator]
    @property
    def emails_enabled(self) -> bool:
        return bool(self.SMTP_HOST and self.EMAILS_FROM_EMAIL)

    # TODO: update type to EmailStr when sqlmodel supports it
    EMAIL_TEST_USER: str = "test@example.com"
    # TODO: update type to EmailStr when sqlmodel supports it
    FIRST_SUPERUSER: str = "memo"
    FIRST_SUPERUSER_PASSWORD: str = "memo"

    def _check_default_secret(self, var_name: str, value: str | None) -> None:
        if value == "changethis":
            message = (
                f'The value of {var_name} is "changethis", for security, please change it, at least for deployments.'
            )
            if self.ENVIRONMENT == "local":
                warnings.warn(message, stacklevel=1)
            else:
                raise ValueError(message)

    @model_validator(mode="after")
    def _enforce_non_default_secrets(self) -> Self:
        self._check_default_secret("SECRET_KEY", self.SECRET_KEY)
        self._check_default_secret("POSTGRES_PASSWORD", self.POSTGRES_PASSWORD)
        self._check_default_secret("FIRST_SUPERUSER_PASSWORD", self.FIRST_SUPERUSER_PASSWORD)

        return self

    # 分块配置
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 100

    # OpenAI API Key
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    CHAT_MODEL: str = "gpt-4o-mini"

    # Tavily API Key
    TAVILY_API_KEY: str = ""

    # 嵌入模块配置
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    EMBEDDING_URL: str = ""
    EMBEDDING_DIMENSION: int = 1024

    # Qrant 配置
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "memomate_dev"
    QDRANT_PATH: str = "database/qdrant_db"
    # 重排序模块配置
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # 数据目录
    DATA_DIR: str = "data"

    # 上传文件目录
    UPLOAD_DIR: str = "uploads"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def UPLOAD_DIR_PATH(self) -> Path:
        return Path(self.DATA_DIR) / self.UPLOAD_DIR

    # Kafka 配置
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"

    OPENRUOTER_API_KEY: str = ""
    OPENROUTER_API_BASE: str = "https://openrouter.ai/api/v1"

    SILICONFLOW_API_KEY: str = ""
    SILICONFLOW_API_BASE: str = "https://api.siliconflow.cn/"

    MCP_CONFIG: dict = {
        "mcpServers": {
            "context7": {"url": "https://mcp.context7.com/mcp"},
            "deepwiki": {"url": "https://mcp.deepwiki.com/mcp"},
        }
    }


settings: Settings = Settings()  # type: ignore
