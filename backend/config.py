import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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
    
    # Phase 6 Optimization configurations
    FUSION_METHOD: str = os.getenv("FUSION_METHOD", "normalized") # 'weighted', 'normalized', 'rrf'
    QUERY_AWARE_FUSION: bool = os.getenv("QUERY_AWARE_FUSION", "false").lower() == "true"
    RRF_CONSTANT: int = int(os.getenv("RRF_CONSTANT", "60"))
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    
    
    # Phase 5 VLM Backend configuration
    VLM_PROVIDER: str = os.getenv("VLM_PROVIDER", "gemini") # 'gemini' or 'openai'
    
    # Gemini API configurations
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GEMINI_TIMEOUT: float = float(os.getenv("GEMINI_TIMEOUT", "30.0"))
    GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
    
    # OpenAI / OpenRouter API configurations
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TIMEOUT: float = float(os.getenv("OPENAI_TIMEOUT", "30.0"))
    OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
    
    # Phase 5 VLM cost & token budget configurations
    GENERATION_TOP_K: int = int(os.getenv("GENERATION_TOP_K", "2"))
    VLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("VLM_MAX_OUTPUT_TOKENS", "512"))
    MAX_OCR_CONTEXT_CHARS: int = int(os.getenv("MAX_OCR_CONTEXT_CHARS", "4000"))
    MAX_VLM_IMAGES: int = int(os.getenv("MAX_VLM_IMAGES", "2"))
    MAX_VLM_CALLS_PER_REQUEST: int = int(os.getenv("MAX_VLM_CALLS_PER_REQUEST", "1"))
    MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "1024"))
    ENABLE_LIVE_VLM_TESTS: bool = os.getenv("ENABLE_LIVE_VLM_TESTS", "false").lower() == "true"
    
    # Phase 8 Performance optimization settings
    VLM_MAX_PAGES: int = int(os.getenv("VLM_MAX_PAGES", "2"))
    VLM_SCORE_GAP_THRESHOLD: float = float(os.getenv("VLM_SCORE_GAP_THRESHOLD", "0.25"))
    VLM_TEXT_DETAIL: str = os.getenv("VLM_TEXT_DETAIL", "low")
    VLM_VISUAL_DETAIL: str = os.getenv("VLM_VISUAL_DETAIL", "high")
    VLM_TEXT_MAX_IMAGES: int = int(os.getenv("VLM_TEXT_MAX_IMAGES", "1"))
    VLM_VISUAL_MAX_IMAGES: int = int(os.getenv("VLM_VISUAL_MAX_IMAGES", "1"))
    VLM_CROP_EVIDENCE: bool = os.getenv("VLM_CROP_EVIDENCE", "false").lower() == "true"
    PDF_RENDER_DPI: int = int(os.getenv("PDF_RENDER_DPI", "150"))
    
    # Ingestion settings
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
    
    def __init__(self):
        # Validate weights
        if not (0.0 <= self.TEXT_RETRIEVAL_WEIGHT <= 1.0):
            raise ValueError(f"TEXT_RETRIEVAL_WEIGHT must be between 0.0 and 1.0, got {self.TEXT_RETRIEVAL_WEIGHT}")
        if not (0.0 <= self.IMAGE_RETRIEVAL_WEIGHT <= 1.0):
            raise ValueError(f"IMAGE_RETRIEVAL_WEIGHT must be between 0.0 and 1.0, got {self.IMAGE_RETRIEVAL_WEIGHT}")
            
        # Validate provider
        prov = self.VLM_PROVIDER.lower().strip()
        if prov not in ("gemini", "openai", "openrouter", "local"):
            raise ValueError(f"VLM_PROVIDER must be one of 'gemini', 'openai', 'openrouter', 'local', got {self.VLM_PROVIDER}")
        self.VLM_PROVIDER = prov
        
        # Log key absence warnings
        if self.VLM_PROVIDER == "gemini" and not self.GEMINI_API_KEY:
            logger.warning("VLM_PROVIDER is set to 'gemini' but GEMINI_API_KEY is not configured.")
        elif self.VLM_PROVIDER in ("openai", "openrouter") and not self.OPENAI_API_KEY:
            logger.warning(f"VLM_PROVIDER is set to '{self.VLM_PROVIDER}' but OPENAI_API_KEY is not configured.")
            
    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

settings = Settings()
