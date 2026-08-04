import logging
import firebase_admin
from firebase_admin import credentials, firestore
from backend.config import settings

logger = logging.getLogger(__name__)

_firebase_app = None
_firestore_client = None

def initialize_firebase() -> bool:
    """
    Initializes the Firebase Admin SDK singleton-safely.
    Exposers credentials resolution and prevents double initialization.
    """
    global _firebase_app, _firestore_client
    
    if _firebase_app is not None:
        return True
        
    if not settings.FIREBASE_ENABLED:
        logger.info("Firebase integration is disabled.")
        return False
        
    try:
        # Check if already initialized by another module
        if firebase_admin._apps:
            _firebase_app = firebase_admin.get_app()
            _firestore_client = firestore.client()
            logger.info("Firebase Admin SDK already initialized.")
            return True
            
        cred_path = settings.resolved_firebase_credentials
        if not cred_path:
            logger.error("Firebase credentials path not configured or could not be resolved.")
            return False
            
        # Initialize Firebase Admin SDK
        logger.info(f"Initializing Firebase Admin SDK using credentials from: {cred_path}")
        cred = credentials.Certificate(cred_path)
        _firebase_app = firebase_admin.initialize_app(cred)
        _firestore_client = firestore.client()
        logger.info("Firebase Admin SDK and Firestore client initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}", exc_info=True)
        _firebase_app = None
        _firestore_client = None
        return False

def get_firestore_client():
    """
    Returns the initialized Firestore client.
    Initializes Firebase if not already done.
    """
    global _firestore_client
    if _firestore_client is None:
        initialize_firebase()
    return _firestore_client

def is_firebase_available() -> bool:
    """
    Checks if Firebase integration is enabled and successfully connected/available.
    """
    if not settings.FIREBASE_ENABLED:
        return False
    try:
        client = get_firestore_client()
        return client is not None
    except Exception:
        return False
