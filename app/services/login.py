from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from app.models.login_model import User
from app.schemas.login import UserCreate, UserLogin
from app.config.constants import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from app.db.mysql import get_db_session
from fastapi import HTTPException, status
from app.utils.password_validator import validate_password

# 配置密码哈希和JWT
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class LoginService:
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def get_user_by_email(email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        with get_db_session() as session:
            return session.query(User).filter(User.email == email).first()

    @staticmethod
    def authenticate_user(email: str, password: str) -> Union[User, bool]:
        """验证用户"""
        user = LoginService.get_user_by_email(email)
        if not user:
            return False
        if not LoginService.verify_password(password, user.hashed_password):
            return False
        return user

    @staticmethod
    def create_user(user_data: UserCreate) -> User:
        """创建新用户"""
        # 验证密码强度
        validate_password(user_data.password)
        
        with get_db_session() as session:
            # 检查邮箱是否已存在
            if session.query(User).filter(User.email == user_data.email).first():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # 创建新用户
            hashed_password = LoginService.get_password_hash(user_data.password)
            db_user = User(
                username=user_data.username,
                email=user_data.email,
                hashed_password=hashed_password
            )
            session.add(db_user)
            session.commit()
            session.refresh(db_user)
            return db_user

    @staticmethod
    def get_current_user(token: str) -> User:
        """获取当前用户"""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email: str = payload.get("sub")
            if email is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception
            
        user = LoginService.get_user_by_email(email)
        if user is None:
            raise credentials_exception
        return user
