from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import timedelta
from app.schemas.login import UserCreate, UserResponse, Token
from app.services.login import LoginService
from app.config.constants import ACCESS_TOKEN_EXPIRE_MINUTES
from app.models.login_model import User
from app.config.log_config import get_logger

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
logger = get_logger(__name__)

@router.post("/register", response_model=UserResponse)
def register(user: UserCreate):
    logger.info(f"新用户注册请求: {user.email}")
    try:
        new_user = LoginService.create_user(user)
        logger.info(f"用户注册成功: {user.email}")
        return new_user
    except HTTPException as e:
        logger.warning(f"用户注册失败: {user.email}, 原因: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"用户注册发生错误: {user.email}, 错误: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    logger.info(f"用户登录尝试: {form_data.username}")
    user = LoginService.authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"登录失败: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = LoginService.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    logger.info(f"用户登录成功: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
def read_users_me(token: str = Depends(oauth2_scheme)):
    current_user = LoginService.get_current_user(token)
    logger.info(f"用户信息获取: {current_user.email}")
    return current_user
