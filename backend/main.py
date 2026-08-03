# pyrefly: ignore [missing-import]
import logging
from contextlib import asynccontextmanager
# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from backend.config import settings
from backend.database import init_db

# Configure standard Python logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backend")

@asynccontextmanager
def lifespan(app: FastAPI):
    # Startup Initialization
    logger.info("Initializing Multimodal RAG VQA backend application...")
    
    # 1. Initialize and verify required directories
    logger.info("Verifying and creating storage directories...")
    try:
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        settings.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Storage directories verified.")
    except Exception as e:
        logger.critical(f"Directory initialization failed: {e}", exc_info=True)
        raise e

    # 2. Initialize SQLite Database
    try:
        init_db()
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}", exc_info=True)
        raise e
        
    logger.info("Application startup completed successfully.")
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

app = FastAPI(
    title="Multimodal RAG VQA",
    description="Backend API for Multimodal RAG for Visual Question Answering (VQA) - Phase 1 Foundation",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    """
    Root endpoint containing basic project information.
    """
    return {
        "project": "Multimodal RAG for Visual Question Answering (VQA)",
        "phase": 1,
        "status": "running"
    }

@app.get("/health", status_code=200)
def health_check():
    """
    Health check endpoint returning status, service name, and active phase.
    """
    return {
        "status": "healthy",
        "service": "Multimodal RAG VQA",
        "phase": 1
    }
