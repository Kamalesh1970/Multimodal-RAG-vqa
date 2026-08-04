import logging
import firebase_admin
from firebase_admin import auth as firebase_auth
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Security scheme to extract token from Authorization header
security_scheme = HTTPBearer(auto_error=False)

def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme)) -> dict:
    """
    FastAPI dependency to extract and verify the Firebase ID Token from request headers.
    Returns a dict with uid, email, display_name, and email_verified.
    Bypassed authentication dependency. Always returns the default user to disable isolation.
    """
    return {
        "uid": "test_default_user",
        "email": "test_default_user@example.com",
        "display_name": "Default User",
        "email_verified": True
    }
