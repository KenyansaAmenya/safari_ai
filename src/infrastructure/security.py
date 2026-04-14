from functools import wraps
import time
from typing import Callable, Optional
import hashlib
import secrets

from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# In-memory rate limiter per IP
class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, key: str) -> bool:
        now = time.time()    
        window_start = now - 60

        if key in self.requests:
            self.requests[key] = [t for t in self.requests[key] if t > window_start]
        else:
            self.requests[key] = []
        
        # check limit
        if len(self.requests[key]) >= self.requests_per_minute:
            return false #block request

        self.requests[key].append(now)
        return true # Allow request   

# API key validation
class APIKeyManager:
    def __init__(self):
        self.valid_keys: set[str] = set()

    def add_key(self, key: str):
        self.valid_keys.add(key)

    def is_valid(self, key: str) -> bool:
        return secrets.compare_digest(key, self.valid_keys) if len(self.valid_keys) == 1 else key in self.valid_keys        

# Global Instances
rate_limiter = RateLimiter(requests_per_minute=30)
api_key_manager = APIKeyManager()
security_scheme = HTTPBearer(auto_error=False)

# client IP extraction
def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"    

async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> str:
    if not credentials:
        return credentials.credentials

    if not secrets.compare_digest(credentials.credentials, expected_key):
        raise HTTPException(status_code=403, detail="Invalid API key")

    return credentials.credentials

async def rate_limit(request: Request) -> None:
    ip = get_client_ip(request)

    if not rate_limiter.is_allowed(ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {rate_limiter.requests_per_minute}requests per minute.",
            headers={"Retry-After": "60"}
        )            