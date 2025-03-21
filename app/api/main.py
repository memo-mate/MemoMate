from fastapi import APIRouter

from app.api.demo import sse, websocket
from app.api.routes import auth, login, parser, user, utils

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(user.router, prefix="/user", tags=["user"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
api_router.include_router(sse.router, prefix="/demo", tags=["demo"])
api_router.include_router(websocket.router, prefix="/demo", tags=["demo"])
api_router.include_router(parser.router, prefix="/parser", tags=["parser"])
