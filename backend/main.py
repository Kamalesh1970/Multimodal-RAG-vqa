# pyrefly: ignore [missing-import]
import logging
import json
from contextlib import asynccontextmanager
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from backend.config import settings
from backend.database import init_db, get_db_connection
from backend.ingestion.processor import ingest_document, IngestionError

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
async def lifespan(app: FastAPI):
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
        
    # 3. Initialize FAISS Vector Store
    try:
        from backend.embeddings.text_embedder import TextEmbedder
        from backend.embeddings.image_embedder import ImageEmbedder
        from backend.vector_store import VectorStore
        
        text_dim = TextEmbedder.get_dimension()
        image_dim = ImageEmbedder.get_dimension()
        
        VectorStore.initialize(text_dim=text_dim, image_dim=image_dim)
        logger.info("Vector store initialized successfully.")
    except Exception as e:
        logger.critical(f"Vector store initialization failed: {e}", exc_info=True)
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
        "phase": 2,
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
        "phase": 2
    }

@app.post("/documents/upload", status_code=200)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document (PDF, PNG, JPG, JPEG), processes orientation,
    renders pages (PDF), executes OCR, and persists metadata in DB.
    """
    try:
        file_bytes = await file.read()
        result = ingest_document(file_bytes, file.filename)
        return result
    except IngestionError as ie:
        raise HTTPException(status_code=ie.status_code, detail=ie.detail)
    except Exception as e:
        logger.error(f"Unexpected error during document upload {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected internal ingestion failure.")

@app.get("/documents/{doc_id}")
def get_document_metadata(doc_id: str):
    """
    Retrieves metadata for a specific document.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT doc_id, filename, file_type, page_count, status, created_at FROM documents WHERE doc_id = ?",
                (doc_id,)
            )
            row = cursor.fetchone()
    except Exception as e:
        logger.error(f"Database query error when retrieving document metadata {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database query error.")
        
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    return {
        "doc_id": row["doc_id"],
        "filename": row["filename"],
        "file_type": row["file_type"],
        "page_count": row["page_count"],
        "status": row["status"],
        "created_at": row["created_at"]
    }

@app.get("/documents/{doc_id}/ocr")
def get_document_ocr(doc_id: str):
    """
    Retrieves stored OCR results page-by-page for a specific document.
    """
    # 1. Verify document exists
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM documents WHERE doc_id = ?", (doc_id,))
            doc_row = cursor.fetchone()
    except Exception as e:
        logger.error(f"Database query error when checking document {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database query error.")
        
    if not doc_row:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    # 2. Fetch pages
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT page_number, width, height, ocr_text, ocr_blocks_json FROM pages WHERE doc_id = ? ORDER BY page_number ASC",
                (doc_id,)
            )
            rows = cursor.fetchall()
    except Exception as e:
        logger.error(f"Database query error when fetching pages for {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database query error.")
        
    pages = []
    for row in rows:
        try:
            blocks = json.loads(row["ocr_blocks_json"])
        except Exception:
            blocks = []
        pages.append({
            "page_number": row["page_number"],
            "width": row["width"],
            "height": row["height"],
            "ocr": {
                "full_text": row["ocr_text"],
                "blocks": blocks
            }
        })
        
    return {
        "doc_id": doc_id,
        "pages": pages
    }

@app.get("/documents/{doc_id}/embeddings")
def get_document_embeddings(doc_id: str):
    """
    Retrieves embedding metadata for a specific document's pages.
    Does not return actual vector values.
    """
    # Verify document exists
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM documents WHERE doc_id = ?", (doc_id,))
            doc_row = cursor.fetchone()
    except Exception as e:
        logger.error(f"Database query error when checking document {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database query error.")
        
    if not doc_row:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    # Fetch page indexing metadata
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, page_number, text_embedding_indexed, image_embedding_indexed, 
                       text_embedding_model, image_embedding_model 
                FROM pages WHERE doc_id = ? ORDER BY page_number ASC
                """,
                (doc_id,)
            )
            rows = cursor.fetchall()
    except Exception as e:
        logger.error(f"Database query error when fetching embeddings metadata for {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database query error.")
        
    try:
        from backend.embeddings.text_embedder import TextEmbedder
        text_dim = TextEmbedder.get_dimension()
    except Exception:
        text_dim = 384
        
    try:
        from backend.embeddings.image_embedder import ImageEmbedder
        image_dim = ImageEmbedder.get_dimension()
    except Exception:
        image_dim = 512
        
    pages = []
    for row in rows:
        pages.append({
            "page_number": row["page_number"],
            "page_id": row["id"],
            "text_embedding": {
                "indexed": bool(row["text_embedding_indexed"]),
                "dimension": text_dim,
                "model": row["text_embedding_model"]
            },
            "image_embedding": {
                "indexed": bool(row["image_embedding_indexed"]),
                "dimension": image_dim,
                "model": row["image_embedding_model"]
            }
        })
        
    return {
        "doc_id": doc_id,
        "pages": pages
    }

@app.get("/system/vector-status")
def get_system_vector_status():
    """
    Returns global status information about the text and image vector indexes.
    """
    try:
        from backend.vector_store import VectorStore
        status = VectorStore.get_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get vector store status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve vector store status.")

class RetrieveRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int | None = None

@app.post("/retrieve")
def retrieve_grounded_evidence(request: RetrieveRequest):
    """
    Multimodal retrieval endpoint. Returns an ordered list of page matches,
    their scores, matched modalities, and lexical evidence blocks.
    Does not run generative LLM responses.
    """
    from backend.retrieval import retrieve_evidence, DocumentNotFoundError, IncompleteDocumentError
    try:
        results = retrieve_evidence(
            doc_id=request.doc_id,
            question=request.question,
            top_k=request.top_k
        )
        return {
            "doc_id": request.doc_id,
            "question": request.question,
            "results": results
        }
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except IncompleteDocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during retrieval execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during retrieval.")

class AskRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int | None = None

@app.post("/ask")
def ask_grounded_question(request: AskRequest):
    """
    Multimodal Question Answering endpoint. Retrieves matched page context and layout images
    and calls Google Gemini API to generate a grounded answer.
    """
    from backend.generation.answer_generator import generate_grounded_answer
    from backend.retrieval import DocumentNotFoundError, IncompleteDocumentError
    try:
        response = generate_grounded_answer(
            doc_id=request.doc_id,
            question=request.question,
            top_k=request.top_k
        )
        return {
            "doc_id": request.doc_id,
            "question": request.question,
            **response
        }
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except IncompleteDocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        if "GEMINI_API_KEY" in str(e) or "OPENAI_API_KEY" in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch specific Gemini API errors and map them to appropriate gateway/service error codes
        try:
            from google.genai.errors import APIError
            if isinstance(e, APIError):
                logger.error(f"Gemini API error occurred: {e.message} (code: {e.code})")
                status_code = 503 if e.code == 429 else 502
                raise HTTPException(status_code=status_code, detail=f"Gemini API error: {e.message}")
        except ImportError:
            pass
            
        # Catch specific OpenAI API errors and map them to appropriate gateway/service error codes
        try:
            from openai import OpenAIError
            if isinstance(e, OpenAIError):
                status_code = getattr(e, "status_code", None)
                message = getattr(e, "message", str(e))
                logger.error(f"OpenAI API error occurred: {message} (code: {status_code})")
                mapped_status = 503 if status_code == 429 else 502
                raise HTTPException(status_code=mapped_status, detail=f"OpenAI API error: {message}")
        except ImportError:
            pass
            
        logger.error(f"Error during answer generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during answer generation.")

