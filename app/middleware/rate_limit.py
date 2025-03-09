from fastapi import HTTPException, Request
from fastapi.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 100, window_size: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        # 清理过期的请求记录
        self.requests[client_ip] = [req_time for req_time in self.requests[client_ip] 
                                  if now - req_time < self.window_size]
        
        # 检查请求频率
        if len(self.requests[client_ip]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Too many requests")
        
        self.requests[client_ip].append(now)
        return await call_next(request) 