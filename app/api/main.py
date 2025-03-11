from fastapi import APIRouter

from app.api.routes import auth, login, user, utils

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(user.router, prefix="/user", tags=["user"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
