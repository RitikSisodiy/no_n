from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from typing import List

from app.core.auth import AuthService
from app.models.auth import (
    User, UserCreate, UserUpdate, Token,
    PasswordReset, PasswordResetConfirm
)
from app.core.storage import DocumentStorage

router = APIRouter(prefix="/auth", tags=["auth"])

# Initialize services
document_storage = DocumentStorage()
auth_service = AuthService(document_storage)

@router.post("/register", response_model=User)
async def register_user(user_data: UserCreate):
    """Register a new user."""
    return await auth_service.create_user(user_data)

@router.post("/login", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """Login and get access token."""
    user, authenticated = await auth_service.authenticate_user(
        form_data.username,
        form_data.password
    )
    
    if not authenticated:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    
    # Create session
    await auth_service.create_session(
        user,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent")
    )
    
    # Create access token
    return auth_service.create_access_token(user)

@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(auth_service.get_current_active_user)
):
    """Refresh the access token."""
    return auth_service.create_access_token(current_user)

@router.post("/logout")
async def logout(
    current_user: User = Depends(auth_service.get_current_active_user)
):
    """Logout and invalidate the current session."""
    # TODO: Implement session invalidation
    return {"status": "success", "message": "Logged out successfully"}

@router.get("/me", response_model=User)
async def get_current_user(
    current_user: User = Depends(auth_service.get_current_active_user)
):
    """Get current user information."""
    return current_user

@router.put("/me", response_model=User)
async def update_current_user(
    user_data: UserUpdate,
    current_user: User = Depends(auth_service.get_current_active_user)
):
    """Update current user information."""
    return await auth_service.update_user(current_user.id, user_data)

@router.post("/password-reset")
async def request_password_reset(reset_request: PasswordReset):
    """Request a password reset token."""
    token = await auth_service.create_password_reset_token(reset_request.email)
    if token:
        # TODO: Send reset email
        pass
    return {
        "status": "success",
        "message": "If the email exists, a password reset link has been sent"
    }

@router.post("/password-reset/confirm")
async def confirm_password_reset(reset_data: PasswordResetConfirm):
    """Reset password using a token."""
    success = await auth_service.reset_password(reset_data)
    return {
        "status": "success" if success else "error",
        "message": "Password has been reset" if success else "Invalid token"
    }

# Admin endpoints
@router.get("/users", response_model=List[User])
async def list_users(
    current_user: User = Depends(
        lambda: auth_service.require_role(UserRole.ADMIN)
    )
):
    """List all users (admin only)."""
    # TODO: Implement user listing
    return []

@router.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    current_user: User = Depends(
        lambda: auth_service.require_role(UserRole.ADMIN)
    )
):
    """Get user by ID (admin only)."""
    user = await auth_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/users/{user_id}", response_model=User)
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    current_user: User = Depends(
        lambda: auth_service.require_role(UserRole.ADMIN)
    )
):
    """Update user (admin only)."""
    return await auth_service.update_user(user_id, user_data) 