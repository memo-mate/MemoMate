from fastapi import APIRouter

from app.api.demo import sse, websocket
from app.api.routes import auth, emb, history, llm, login, parser, rag_llm, upload, user, utils

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(user.router, prefix="/user", tags=["user"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
api_router.include_router(sse.router, prefix="/demo", tags=["demo"])
api_router.include_router(websocket.router, prefix="/demo", tags=["demo"])
api_router.include_router(parser.router, prefix="/parser", tags=["parser"])
api_router.include_router(llm.router, prefix="/llm", tags=["llm"])
api_router.include_router(history.router, prefix="/history", tags=["history"])
api_router.include_router(rag_llm.router, prefix="/rag_llm", tags=["rag_llm"])  # 新增RAG路由
api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(emb.router, prefix="/embedding", tags=["embedding"])  # 新增向量库路由
