import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlite3 import Connection
import os
import secrets

from app.models.auth import (
    User, UserCreate, UserUpdate, Token, TokenData,
    Session, UserRole, PasswordResetConfirm
)

logger = logging.getLogger(__name__)

class AuthService:
    """Service for handling authentication and user management."""
    
    def __init__(self, storage: 'DocumentStorage'):
        """Initialize the auth service."""
        self.storage = storage
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
        self.secret_key = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def _init_db(self, conn: Connection) -> None:
        """Initialize authentication tables."""
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                last_login TIMESTAMP
            )
        """)
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                last_activity TIMESTAMP NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Password reset tokens table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used BOOLEAN NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        conn.commit()
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if email exists
                cursor.execute(
                    "SELECT id FROM users WHERE email = ?",
                    (user_data.email,)
                )
                if cursor.fetchone():
                    raise HTTPException(
                        status_code=400,
                        detail="Email already registered"
                    )
                
                # Create user
                user_id = str(uuid.uuid4())
                now = datetime.utcnow()
                
                cursor.execute("""
                    INSERT INTO users (
                        id, email, full_name, password_hash, role,
                        is_active, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    user_data.email,
                    user_data.full_name,
                    self.pwd_context.hash(user_data.password),
                    user_data.role.value,
                    user_data.is_active,
                    now,
                    now
                ))
                
                conn.commit()
                
                return await self.get_user(user_id)
                
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM users WHERE id = ?
                """, (user_id,))
                row = cursor.fetchone()
                
                if row:
                    return User(
                        id=row[0],
                        email=row[1],
                        full_name=row[2],
                        role=UserRole(row[4]),
                        is_active=bool(row[5]),
                        created_at=datetime.fromisoformat(row[6]),
                        updated_at=datetime.fromisoformat(row[7]),
                        last_login=datetime.fromisoformat(row[8]) if row[8] else None
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            raise
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM users WHERE email = ?
                """, (email,))
                row = cursor.fetchone()
                
                if row:
                    return User(
                        id=row[0],
                        email=row[1],
                        full_name=row[2],
                        role=UserRole(row[4]),
                        is_active=bool(row[5]),
                        created_at=datetime.fromisoformat(row[6]),
                        updated_at=datetime.fromisoformat(row[7]),
                        last_login=datetime.fromisoformat(row[8]) if row[8] else None
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            raise
    
    async def update_user(
        self,
        user_id: str,
        user_data: UserUpdate
    ) -> User:
        """Update a user."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get current user data
                current_user = await self.get_user(user_id)
                if not current_user:
                    raise HTTPException(
                        status_code=404,
                        detail="User not found"
                    )
                
                # Build update query
                updates = []
                params = []
                
                if user_data.email is not None:
                    updates.append("email = ?")
                    params.append(user_data.email)
                
                if user_data.full_name is not None:
                    updates.append("full_name = ?")
                    params.append(user_data.full_name)
                
                if user_data.role is not None:
                    updates.append("role = ?")
                    params.append(user_data.role.value)
                
                if user_data.is_active is not None:
                    updates.append("is_active = ?")
                    params.append(user_data.is_active)
                
                if user_data.password is not None:
                    updates.append("password_hash = ?")
                    params.append(self.pwd_context.hash(user_data.password))
                
                if updates:
                    updates.append("updated_at = ?")
                    params.append(datetime.utcnow().isoformat())
                    params.append(user_id)
                    
                    cursor.execute(f"""
                        UPDATE users
                        SET {", ".join(updates)}
                        WHERE id = ?
                    """, params)
                    
                    conn.commit()
                
                return await self.get_user(user_id)
                
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            raise
    
    async def authenticate_user(
        self,
        email: str,
        password: str
    ) -> Tuple[Optional[User], bool]:
        """Authenticate a user."""
        try:
            user = await self.get_user_by_email(email)
            if not user:
                return None, False
            
            if not user.is_active:
                return None, False
            
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT password_hash FROM users WHERE id = ?
                """, (user.id,))
                row = cursor.fetchone()
                
                if not row or not self.pwd_context.verify(
                    password,
                    row[0]
                ):
                    return None, False
                
                # Update last login
                cursor.execute("""
                    UPDATE users
                    SET last_login = ?
                    WHERE id = ?
                """, (datetime.utcnow().isoformat(), user.id))
                conn.commit()
                
                return user, True
                
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            raise
    
    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> Token:
        """Create a JWT access token."""
        try:
            if expires_delta is None:
                expires_delta = timedelta(
                    minutes=self.access_token_expire_minutes
                )
            
            expire = datetime.utcnow() + expires_delta
            
            to_encode = {
                "sub": user.id,
                "email": user.email,
                "role": user.role.value,
                "exp": expire
            }
            
            token = jwt.encode(
                to_encode,
                self.secret_key,
                algorithm=self.algorithm
            )
            
            return Token(
                access_token=token,
                expires_at=expire
            )
            
        except Exception as e:
            logger.error(f"Error creating access token: {str(e)}")
            raise
    
    async def create_session(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Session:
        """Create a user session."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                
                session_id = str(uuid.uuid4())
                token = secrets.token_urlsafe(32)
                now = datetime.utcnow()
                expires_at = now + timedelta(days=self.refresh_token_expire_days)
                
                cursor.execute("""
                    INSERT INTO sessions (
                        id, user_id, token, created_at,
                        expires_at, last_activity,
                        ip_address, user_agent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    user.id,
                    token,
                    now.isoformat(),
                    expires_at.isoformat(),
                    now.isoformat(),
                    ip_address,
                    user_agent
                ))
                
                conn.commit()
                
                return Session(
                    id=session_id,
                    user_id=user.id,
                    token=token,
                    created_at=now,
                    expires_at=expires_at,
                    last_activity=now,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise
    
    async def get_current_user(
        self,
        token: str = None
    ) -> User:
        """Get the current user from the JWT token."""
        try:
            token = token or Depends(self.oauth2_scheme)
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication token"
                )
            
            user = await self.get_user(user_id)
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="User not found"
                )
            
            if not user.is_active:
                raise HTTPException(
                    status_code=401,
                    detail="User is inactive"
                )
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )
    
    async def get_current_active_user(
        self,
        current_user: User = Depends(get_current_user)
    ) -> User:
        """Get the current active user."""
        if not current_user.is_active:
            raise HTTPException(
                status_code=401,
                detail="User is inactive"
            )
        return current_user
    
    async def require_role(
        self,
        required_role: UserRole,
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        """Require a specific role for access."""
        if current_user.role != required_role and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions"
            )
        return current_user
    
    async def create_password_reset_token(self, email: str) -> str:
        """Create a password reset token."""
        try:
            user = await self.get_user_by_email(email)
            if not user:
                # Don't reveal that the user doesn't exist
                return None
            
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                
                # Invalidate any existing tokens
                cursor.execute("""
                    UPDATE password_resets
                    SET used = 1
                    WHERE user_id = ? AND used = 0
                """, (user.id,))
                
                # Create new token
                token = secrets.token_urlsafe(32)
                now = datetime.utcnow()
                expires_at = now + timedelta(hours=1)
                
                cursor.execute("""
                    INSERT INTO password_resets (
                        token, user_id, created_at, expires_at
                    ) VALUES (?, ?, ?, ?)
                """, (
                    token,
                    user.id,
                    now.isoformat(),
                    expires_at.isoformat()
                ))
                
                conn.commit()
                return token
                
        except Exception as e:
            logger.error(f"Error creating password reset token: {str(e)}")
            raise
    
    async def reset_password(
        self,
        reset_data: PasswordResetConfirm
    ) -> bool:
        """Reset a user's password using a token."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get token
                cursor.execute("""
                    SELECT user_id, expires_at, used
                    FROM password_resets
                    WHERE token = ?
                """, (reset_data.token,))
                row = cursor.fetchone()
                
                if not row:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid reset token"
                    )
                
                user_id, expires_at, used = row
                expires_at = datetime.fromisoformat(expires_at)
                
                if used or datetime.utcnow() > expires_at:
                    raise HTTPException(
                        status_code=400,
                        detail="Reset token has expired or been used"
                    )
                
                # Update password
                cursor.execute("""
                    UPDATE users
                    SET password_hash = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    self.pwd_context.hash(reset_data.new_password),
                    datetime.utcnow().isoformat(),
                    user_id
                ))
                
                # Mark token as used
                cursor.execute("""
                    UPDATE password_resets
                    SET used = 1
                    WHERE token = ?
                """, (reset_data.token,))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error resetting password: {str(e)}")
            raise 