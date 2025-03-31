import httpx

from app.core.config import settings


def get_model_list() -> list[str]:
    """获取模型列表"""
    base_url = settings.OPENAI_API_BASE
    params = {"type": "text"}
    if settings.OPENAI_API_KEY:
        headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
    else:
        headers = None
    response = httpx.get(f"{base_url}/models", params=params, headers=headers)
    items = response.json()
    return [item["id"] for item in items]
