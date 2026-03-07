"""
Authentication routes — signup, login, and current-user info.

Uses in-memory user store + JWT tokens (suitable for a mini-project demo).
"""
import hashlib
import hmac
import os
import secrets
import sys
import time
from pathlib import Path

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ── Config ────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("JWT_SECRET", "routemate-dev-secret-key-change-in-prod")
ALGORITHM = "HS256"
TOKEN_EXPIRE_SECONDS = 60 * 60 * 24  # 24 hours

# ── In-memory user store ─────────────────────────────────────────
# {email: {"name": ..., "email": ..., "password_hash": ...}}
_users: dict = {}

security = HTTPBearer()


# ── Schemas ───────────────────────────────────────────────────────
class SignupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=5, max_length=200)
    password: str = Field(..., min_length=6, max_length=200)


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


class UserResponse(BaseModel):
    name: str
    email: str


# ── Helpers ───────────────────────────────────────────────────────
def _create_token(email: str) -> str:
    payload = {
        "sub": email,
        "exp": int(time.time()) + TOKEN_EXPIRE_SECONDS,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${h.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    salt, hash_hex = stored.split("$", 1)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return hmac.compare_digest(h.hex(), hash_hex)


def _decode_token(token: str) -> str:
    """Return email from token or raise."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    email = _decode_token(credentials.credentials)
    user = _users.get(email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"name": user["name"], "email": user["email"]}


# ── Endpoints ─────────────────────────────────────────────────────
@router.post("/signup", response_model=AuthResponse)
async def signup(req: SignupRequest):
    if req.email in _users:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    _users[req.email] = {
        "name": req.name,
        "email": req.email,
        "password_hash": _hash_password(req.password),
    }
    token = _create_token(req.email)
    return AuthResponse(
        token=token,
        user={"name": req.name, "email": req.email},
    )


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    user = _users.get(req.email)
    if not user or not _verify_password(req.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    token = _create_token(req.email)
    return AuthResponse(
        token=token,
        user={"name": user["name"], "email": user["email"]},
    )


@router.get("/me", response_model=UserResponse)
async def me(user: dict = Depends(get_current_user)):
    return UserResponse(**user)
