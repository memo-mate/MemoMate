import getpass
import os

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # 验证默认值是否正确
        validate_default=False,
        # 优先级：后面文件的配置会覆盖前面文件的配置
        env_file=[".env"],
        env_file_encoding="utf-8",
        # 忽略未定义的配置
        extra="ignore",
    )

    # model 参数
    chunk_size: int = 3200
    chunk_overlap: int = 30

    # OpenAI API Key
    openai_api_key: str

    # Tavily API Key
    tavily_api_key: str

    # 验证 tavily_api_key 是否正确
    @field_validator("tavily_api_key")
    @classmethod
    def validate_tavily_api_key(cls, v):
        if not v:
            v = getpass.getpass("Tavily API key:\n")
            # raise ValueError("tavily_api_key is required")
        os.environ["TAVILY_API_KEY"] = v
        return v


settings = Settings()
