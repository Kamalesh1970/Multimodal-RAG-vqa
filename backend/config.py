import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file from project root
dotenv_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path)

class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Multimodal RAG VQA")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    
    # Resolve paths relative to project root
    DATABASE_PATH: Path = PROJECT_ROOT / Path(os.getenv("DATABASE_PATH", "data/app.db"))
    
    UPLOADS_DIR: Path = PROJECT_ROOT / "data" / "uploads"
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    INDEXES_DIR: Path = PROJECT_ROOT / "data" / "indexes"
    
    # Phase 3 Embeddings configurations
    TEXT_EMBEDDING_MODEL: str = os.getenv("TEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    IMAGE_EMBEDDING_MODEL: str = os.getenv("IMAGE_EMBEDDING_MODEL", "ViT-B-32")
    IMAGE_EMBEDDING_PRETRAINED: str = os.getenv("IMAGE_EMBEDDING_PRETRAINED", "laion2b_s34b_b79k")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "auto")
    VECTOR_INDEX_DIR: Path = PROJECT_ROOT / Path(os.getenv("VECTOR_INDEX_DIR", "data/indexes"))
    # Phase 4 Retrieval configurations
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    TEXT_RETRIEVAL_WEIGHT: float = float(os.getenv("TEXT_RETRIEVAL_WEIGHT", "0.65"))
    IMAGE_RETRIEVAL_WEIGHT: float = float(os.getenv("IMAGE_RETRIEVAL_WEIGHT", "0.35"))
    RETRIEVAL_MIN_SCORE: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.0"))
    
    # Gemini API Key placeholder for future phases
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    
    # Ingestion settings
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
    
    def __init__(self):
        # Validate weights
        if not (0.0 <= self.TEXT_RETRIEVAL_WEIGHT <= 1.0):
            raise ValueError(f"TEXT_RETRIEVAL_WEIGHT must be between 0.0 and 1.0, got {self.TEXT_RETRIEVAL_WEIGHT}")
        if not (0.0 <= self.IMAGE_RETRIEVAL_WEIGHT <= 1.0):
            raise ValueError(f"IMAGE_RETRIEVAL_WEIGHT must be between 0.0 and 1.0, got {self.IMAGE_RETRIEVAL_WEIGHT}")
            
    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

settings = Settings()
