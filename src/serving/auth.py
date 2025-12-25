"""
Authentication and authorization for API.

Features:
- JWT authentication
- API key validation
- OAuth2 support
- Role-based access control
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Role(str, Enum):
    """User roles."""
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"
    READONLY = "readonly"


@dataclass
class User:
    """Authenticated user."""
    
    user_id: str
    email: Optional[str] = None
    roles: Set[Role] = field(default_factory=lambda: {Role.USER})
    metadata: Dict[str, Any] = field(default_factory=dict)
    api_key: Optional[str] = None
    
    def has_role(self, role: Role) -> bool:
        """Check if user has role."""
        return role in self.roles
    
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return Role.ADMIN in self.roles


@dataclass
class AuthResult:
    """Authentication result."""
    
    success: bool
    user: Optional[User] = None
    error: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if authentication is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class AuthProvider(ABC):
    """Abstract authentication provider."""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate with credentials.
        
        Args:
            credentials: Authentication credentials.
            
        Returns:
            AuthResult.
        """
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> AuthResult:
        """Validate authentication token.
        
        Args:
            token: Token to validate.
            
        Returns:
            AuthResult.
        """
        pass


class JWTProvider(AuthProvider):
    """JWT-based authentication."""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7
    ):
        """Initialize JWT provider.
        
        Args:
            secret_key: JWT secret.
            algorithm: JWT algorithm.
            access_token_expire_minutes: Access token lifetime.
            refresh_token_expire_days: Refresh token lifetime.
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
        self.algorithm = algorithm
        self.access_token_expire = timedelta(minutes=access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=refresh_token_expire_days)
        
        self._users: Dict[str, Dict[str, Any]] = {}  # Simple in-memory store
        
    async def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate with username/password."""
        email = credentials.get("email", "")
        password = credentials.get("password", "")
        
        # Simple demo authentication
        if email and password:
            user_data = self._users.get(email)
            
            if user_data and self._verify_password(password, user_data.get("password_hash", "")):
                user = User(
                    user_id=user_data.get("user_id", email),
                    email=email,
                    roles=user_data.get("roles", {Role.USER}),
                    metadata=user_data.get("metadata", {})
                )
                
                return AuthResult(
                    success=True,
                    user=user,
                    expires_at=datetime.utcnow() + self.access_token_expire
                )
                
        return AuthResult(
            success=False,
            error="Invalid credentials"
        )
    
    async def validate_token(self, token: str) -> AuthResult:
        """Validate JWT token."""
        try:
            import jwt
            
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            user = User(
                user_id=payload.get("sub", ""),
                email=payload.get("email"),
                roles={Role(r) for r in payload.get("roles", ["user"])},
                metadata=payload.get("metadata", {})
            )
            
            expires_at = datetime.fromtimestamp(payload.get("exp", 0))
            
            return AuthResult(
                success=True,
                user=user,
                expires_at=expires_at
            )
            
        except ImportError:
            logger.warning("PyJWT not installed")
            return AuthResult(success=False, error="JWT not available")
            
        except Exception as e:
            return AuthResult(
                success=False,
                error=str(e)
            )
            
    async def create_token(
        self,
        user: User,
        token_type: str = "access"
    ) -> str:
        """Create JWT token for user.
        
        Args:
            user: User to create token for.
            token_type: 'access' or 'refresh'.
            
        Returns:
            JWT token string.
        """
        try:
            import jwt
            
            expire = datetime.utcnow() + (
                self.access_token_expire if token_type == "access"
                else self.refresh_token_expire
            )
            
            payload = {
                "sub": user.user_id,
                "email": user.email,
                "roles": [r.value for r in user.roles],
                "metadata": user.metadata,
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": token_type
            }
            
            return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
        except ImportError:
            logger.error("PyJWT not installed")
            raise RuntimeError("JWT not available")
            
    def register_user(
        self,
        email: str,
        password: str,
        roles: Optional[Set[Role]] = None
    ) -> User:
        """Register new user (demo implementation).
        
        Args:
            email: User email.
            password: User password.
            roles: User roles.
            
        Returns:
            Created user.
        """
        user_id = hashlib.sha256(email.encode()).hexdigest()[:16]
        password_hash = self._hash_password(password)
        
        user_data = {
            "user_id": user_id,
            "email": email,
            "password_hash": password_hash,
            "roles": roles or {Role.USER},
            "metadata": {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        self._users[email] = user_data
        
        return User(
            user_id=user_id,
            email=email,
            roles=roles or {Role.USER}
        )
        
    def _hash_password(self, password: str) -> str:
        """Hash password with salt."""
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        )
        return f"{salt}${hashed.hex()}"
        
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        if '$' not in password_hash:
            return False
            
        salt, stored_hash = password_hash.split('$', 1)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        )
        return hmac.compare_digest(hashed.hex(), stored_hash)


class APIKeyProvider(AuthProvider):
    """API key authentication."""
    
    def __init__(self):
        """Initialize API key provider."""
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        
    async def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate with API key."""
        api_key = credentials.get("api_key", "")
        return await self.validate_token(api_key)
        
    async def validate_token(self, token: str) -> AuthResult:
        """Validate API key."""
        key_hash = self._hash_key(token)
        
        if key_hash in self._api_keys:
            key_data = self._api_keys[key_hash]
            
            # Check expiration
            expires_at = key_data.get("expires_at")
            if expires_at and datetime.fromisoformat(expires_at) < datetime.utcnow():
                return AuthResult(
                    success=False,
                    error="API key expired"
                )
                
            user = User(
                user_id=key_data.get("user_id", ""),
                roles={Role(r) for r in key_data.get("roles", ["user"])},
                metadata=key_data.get("metadata", {}),
                api_key=token[:8] + "..."  # Masked
            )
            
            return AuthResult(
                success=True,
                user=user,
                expires_at=datetime.fromisoformat(expires_at) if expires_at else None
            )
            
        return AuthResult(
            success=False,
            error="Invalid API key"
        )
        
    def create_api_key(
        self,
        user_id: str,
        roles: Optional[Set[Role]] = None,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new API key.
        
        Args:
            user_id: Owner user ID.
            roles: Roles for this key.
            expires_in_days: Key lifetime.
            metadata: Additional metadata.
            
        Returns:
            API key string.
        """
        api_key = f"ak_{secrets.token_hex(24)}"
        key_hash = self._hash_key(api_key)
        
        expires_at = None
        if expires_in_days:
            expires_at = (datetime.utcnow() + timedelta(days=expires_in_days)).isoformat()
            
        self._api_keys[key_hash] = {
            "user_id": user_id,
            "roles": [r.value for r in (roles or {Role.USER})],
            "expires_at": expires_at,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        return api_key
        
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key.
        
        Args:
            api_key: Key to revoke.
            
        Returns:
            True if revoked.
        """
        key_hash = self._hash_key(api_key)
        
        if key_hash in self._api_keys:
            del self._api_keys[key_hash]
            return True
        return False
        
    def _hash_key(self, api_key: str) -> str:
        """Hash API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()


class AuthManager:
    """Unified authentication manager."""
    
    def __init__(
        self,
        jwt_provider: Optional[JWTProvider] = None,
        api_key_provider: Optional[APIKeyProvider] = None
    ):
        """Initialize manager.
        
        Args:
            jwt_provider: JWT provider.
            api_key_provider: API key provider.
        """
        self.jwt_provider = jwt_provider or JWTProvider()
        self.api_key_provider = api_key_provider or APIKeyProvider()
        
    async def authenticate(
        self,
        token: Optional[str] = None,
        api_key: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None
    ) -> AuthResult:
        """Authenticate using any method.
        
        Args:
            token: JWT token.
            api_key: API key.
            credentials: Username/password credentials.
            
        Returns:
            AuthResult.
        """
        # Try JWT token
        if token:
            result = await self.jwt_provider.validate_token(token)
            if result.success:
                return result
                
        # Try API key
        if api_key:
            result = await self.api_key_provider.validate_token(api_key)
            if result.success:
                return result
                
        # Try credentials
        if credentials:
            return await self.jwt_provider.authenticate(credentials)
            
        return AuthResult(
            success=False,
            error="No authentication provided"
        )
        
    async def get_current_user(
        self,
        authorization: Optional[str] = None,
        x_api_key: Optional[str] = None
    ) -> Optional[User]:
        """Extract current user from headers.
        
        Args:
            authorization: Authorization header.
            x_api_key: X-API-Key header.
            
        Returns:
            User if authenticated, None otherwise.
        """
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            
        result = await self.authenticate(token=token, api_key=x_api_key)
        
        return result.user if result.success else None


def require_auth(roles: Optional[List[Role]] = None):
    """Decorator to require authentication.
    
    Args:
        roles: Required roles (any of them).
        
    Usage:
        @require_auth(roles=[Role.ADMIN])
        async def admin_endpoint(user: User):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would be implemented with FastAPI dependencies
            # Here's the logic that would be used:
            user = kwargs.get("current_user")
            
            if user is None:
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Not authenticated")
                
            if roles:
                if not any(user.has_role(r) for r in roles):
                    from fastapi import HTTPException
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                    
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator
