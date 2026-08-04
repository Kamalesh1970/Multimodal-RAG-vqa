import time
start_import_time = time.time()
# pyrefly: ignore [missing-import]
import logging
# Force config reload
import json
from contextlib import asynccontextmanager
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from backend.config import settings
from backend.storage import repository
from backend.ingestion.processor import ingest_document, IngestionError
from backend.auth import get_current_user

# Configure standard Python logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backend")

async def load_models_background():
    """
    Background worker that pre-loads text/image embedders after startup.
    Uses run_in_threadpool to keep the event loop unblocked.
    """
    logger.info("Starting background loading of AI models...")
    try:
        from backend.embeddings.text_embedder import TextEmbedder
        from fastapi.concurrency import run_in_threadpool
        await run_in_threadpool(TextEmbedder.get_model)
        logger.info("Background text embedder loaded successfully.")
    except Exception as e:
        logger.error(f"Error during background text embedder loading: {e}", exc_info=True)
        
    try:
        from backend.embeddings.image_embedder import ImageEmbedder
        from fastapi.concurrency import run_in_threadpool
        await run_in_threadpool(ImageEmbedder.get_model_and_transforms)
        logger.info("Background image embedder loaded successfully.")
    except Exception as e:
        logger.error(f"Error during background image embedder loading: {e}", exc_info=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    start_time = time.time()
    logger.info("Initializing Multimodal RAG VQA backend application...")
    
    # 1. Initialize and verify required directories
    t_start = time.time()
    try:
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        settings.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        folders_time = time.time() - t_start
        logger.info(f"Storage directories verified: {folders_time:.4f}s")
    except Exception as e:
        logger.critical(f"Directory initialization failed: {e}", exc_info=True)
        raise e

    # 2. Initialize Storage Backend (SQLite or Firestore)
    t_start = time.time()
    try:
        repository.init_storage()
        db_time = time.time() - t_start
        db_provider = repository.get_db_provider()
        logger.info(f"Database ({db_provider}) initialized: {db_time:.4f}s")
    except Exception as e:
        logger.critical(f"Storage initialization failed: {e}", exc_info=True)
        raise e

    # 3. Initialize FAISS Vector Store Metadata only
    t_start = time.time()
    try:
        from backend.vector_store import VectorStore
        VectorStore.initialize()
        faiss_time = time.time() - t_start
        logger.info(f"FAISS metadata initialized: {faiss_time:.4f}s")
    except Exception as e:
        logger.critical(f"Vector store initialization failed: {e}", exc_info=True)
        raise e
        
    total_time = time.time() - start_time
    logger.info(f"Application ready: {total_time:.4f}s")
    
    # Trigger background model pre-loading
    asyncio.create_task(load_models_background())
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

app = FastAPI(
    title="Multimodal RAG VQA",
    description="Backend API for Multimodal RAG for Visual Question Answering (VQA) - Phase 1 Foundation",
    version="1.0.0",
    lifespan=lifespan
)

# Serve static frontend assets
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

def assert_document_access(doc_id: str, uid: str) -> dict:
    doc = repository.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found.")
    owner_id = doc.get("owner_id")
    if uid != "test_default_user":
        if owner_id != uid:
            raise HTTPException(status_code=403, detail="Access denied to this document resource.")
    return doc

# Secure serving of processed page images
@app.get("/processed/{doc_id}/{filename}")
def serve_processed_file(doc_id: str, filename: str, current_user: dict = Depends(get_current_user)):
    assert_document_access(doc_id, current_user["uid"])
    
    import re
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", filename):
        raise HTTPException(status_code=400, detail="Invalid file name.")
        
    file_path = settings.PROCESSED_DIR / doc_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
        
    return FileResponse(file_path)

@app.get("/")
def read_root():
    """
    Root endpoint serving the static HTML frontend.
    """
    return FileResponse("frontend/index.html")

@app.get("/auth/config")
def get_firebase_config():
    """
    Exposes Firebase Web SDK config parameters (non-sensitive API key, project ID, domain, etc.)
    for frontend client initialization.
    """
    import os
    return {
        "firebase_enabled": settings.FIREBASE_ENABLED,
        "apiKey": settings.FIREBASE_API_KEY or os.getenv("FIREBASE_API_KEY", ""),
        "authDomain": settings.FIREBASE_AUTH_DOMAIN or os.getenv("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": settings.FIREBASE_PROJECT_ID or os.getenv("FIREBASE_PROJECT_ID", ""),
        "appId": settings.FIREBASE_APP_ID or os.getenv("FIREBASE_APP_ID", "")
    }

@app.get("/health", status_code=200)
def health_check():
    """
    Health check endpoint returning status, service name, phase, and database details.
    """
    db_provider = repository.get_db_provider()
    db_connected = repository.is_db_connected()
    return {
        "status": "healthy",
        "service": "Multimodal RAG VQA",
        "phase": 2,
        "database": {
            "provider": db_provider,
            "connected": db_connected
        }
    }

@app.get("/system/status", status_code=200)
def system_status():
    """
    Returns runtime configuration status, simulation details, and database details.
    """
    is_simulated = (settings.VLM_PROVIDER == "local")
    if settings.VLM_PROVIDER == "gemini" and not settings.GEMINI_API_KEY:
        is_simulated = True
    elif settings.VLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
        is_simulated = True
        
    db_provider = repository.get_db_provider()
    db_connected = repository.is_db_connected()
    return {
        "status": "healthy",
        "project": "Multimodal RAG for Visual Question Answering (VQA)",
        "phase": 7,
        "vlm_provider": settings.VLM_PROVIDER,
        "generation_mode": "simulated" if is_simulated else "live",
        "database": {
            "provider": db_provider,
            "connected": db_connected
        }
    }


@app.get("/documents", status_code=200)
def list_documents(current_user: dict = Depends(get_current_user)):
    """
    Lists metadata of all documents belonging to the authenticated user.
    """
    return repository.get_documents(current_user["uid"])

@app.post("/upload", status_code=200)
@app.post("/documents/upload", status_code=200)
async def upload_document(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """
    Uploads a document (PDF, PNG, JPG, JPEG), processes orientation,
    renders pages (PDF), executes OCR, and persists metadata in DB.
    """
    try:
        file_bytes = await file.read()
        from fastapi.concurrency import run_in_threadpool
        result = await run_in_threadpool(ingest_document, file_bytes, file.filename, current_user["uid"])
        return result
    except IngestionError as ie:
        raise HTTPException(status_code=ie.status_code, detail=ie.detail)
    except Exception as e:
        logger.error(f"Unexpected error during document upload {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected internal ingestion failure.")

@app.get("/documents/{doc_id}")
def get_document_metadata(doc_id: str, current_user: dict = Depends(get_current_user)):
    """
    Retrieves metadata for a specific document.
    """
    row = assert_document_access(doc_id, current_user["uid"])
    return {
        "doc_id": row["doc_id"],
        "filename": row["filename"],
        "file_type": row["file_type"],
        "page_count": row["page_count"],
        "status": row["status"],
        "created_at": row["created_at"]
    }

@app.delete("/documents/{doc_id}", status_code=200)
def delete_user_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    """
    Deletes a document owned by the authenticated user along with its pages,
    files, and FAISS index mappings.
    """
    # 1. Authorize ownership
    assert_document_access(doc_id, current_user["uid"])
    
    # 2. Clear from FAISS index maps
    try:
        pages = repository.get_pages(doc_id)
        from backend.vector_store import VectorStore
        for p in pages:
            pid = p.get("page_id") or p.get("id")
            if pid is not None:
                VectorStore.remove_text_vector(pid)
                VectorStore.remove_image_vector(pid)
        VectorStore.save_indices()
    except Exception as err:
        logger.error(f"Failed to clear FAISS indices for document {doc_id}: {err}", exc_info=True)
        
    # 3. Clean up physical files
    try:
        doc = repository.get_document(doc_id)
        if doc and doc.get("stored_path"):
            stored_p = Path(doc["stored_path"])
            if stored_p.exists():
                stored_p.unlink()
        
        proc_dir = settings.PROCESSED_DIR / doc_id
        if proc_dir.exists():
            import shutil
            shutil.rmtree(proc_dir)
    except Exception as err:
        logger.error(f"Failed to clean up files for document {doc_id}: {err}", exc_info=True)
        
    # 4. Delete database metadata record
    try:
        repository.delete_document(doc_id)
    except Exception as err:
        logger.error(f"Failed to delete document metadata for {doc_id}: {err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete document metadata.")
        
    return {"status": "success", "message": f"Document {doc_id} successfully deleted."}

@app.get("/documents/{doc_id}/ocr")
def get_document_ocr(doc_id: str, current_user: dict = Depends(get_current_user)):
    """
    Retrieves stored OCR results page-by-page for a specific document.
    """
    # 1. Authorize document ownership
    assert_document_access(doc_id, current_user["uid"])
    
    # 2. Fetch pages
    try:
        rows = repository.get_pages(doc_id)
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
def get_document_embeddings(doc_id: str, current_user: dict = Depends(get_current_user)):
    """
    Retrieves embedding metadata for a specific document's pages.
    Does not return actual vector values.
    """
    # Authorize document ownership
    assert_document_access(doc_id, current_user["uid"])
    
    # Fetch page indexing metadata
    try:
        rows = repository.get_pages(doc_id)
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
            "page_id": row["page_id"],
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
def retrieve_grounded_evidence(request: RetrieveRequest, current_user: dict = Depends(get_current_user)):
    # Validate request parameters before database access checks
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Search question cannot be empty or whitespace-only.")
    if len(request.question) > 500:
        raise HTTPException(status_code=400, detail=f"Search question length exceeds maximum limit of 500 characters. Got {len(request.question)}.")
    if request.top_k is not None and request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer.")
        
    assert_document_access(request.doc_id, current_user["uid"])
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
    session_id: str | None = None

@app.post("/ask")
def ask_grounded_question(request: AskRequest, current_user: dict = Depends(get_current_user)):
    # Validate request parameters before database access checks
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Search question cannot be empty or whitespace-only.")
    if len(request.question) > 500:
        raise HTTPException(status_code=400, detail=f"Search question length exceeds maximum limit of 500 characters. Got {len(request.question)}.")
    if request.top_k is not None and request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer.")
        
    """
    Multimodal Question Answering endpoint. Retrieves matched page context and layout images,
    calls Google Gemini API to generate a grounded answer, and persists chat history.
    """
    from backend.generation.answer_generator import generate_grounded_answer
    from backend.retrieval import DocumentNotFoundError, IncompleteDocumentError
    
    # 1. Authorize document ownership before anything else
    assert_document_access(request.doc_id, current_user["uid"])
    
    # Determine or generate session ID
    session_id = request.session_id
    if not session_id:
        import uuid
        session_id = f"session_{uuid.uuid4()}"
        
    try:
        # Perform generation/retrieval first to validate request parameters & document existence
        response = generate_grounded_answer(
            doc_id=request.doc_id,
            question=request.question,
            top_k=request.top_k
        )
        
        # Save messages to database only after successful generation
        try:
            # Save user question to chat session
            repository.save_chat_message(
                session_id=session_id,
                role="user",
                content=request.question,
                doc_id=request.doc_id,
                owner_id=current_user["uid"]
            )
            
            # Build metadata for log
            first_evidence_score = 0.0
            if response.get("evidence"):
                first_ev = response["evidence"][0]
                if isinstance(first_ev, dict):
                    first_evidence_score = first_ev.get("score", 0.0)
                elif hasattr(first_ev, "score"):
                    first_evidence_score = getattr(first_ev, "score", 0.0)
                    
            meta = {
                "retrieved_pages": response.get("pages_used", []),
                "retrieval_score": first_evidence_score,
                "grounding_status": response.get("grounding_mode", "unverified"),
                "provider": settings.VLM_PROVIDER,
                "model": settings.GEMINI_MODEL if settings.VLM_PROVIDER == "gemini" else settings.OPENAI_MODEL
            }
            
            # Save assistant answer to chat session
            repository.save_chat_message(
                session_id=session_id,
                role="assistant",
                content=response.get("answer", ""),
                doc_id=request.doc_id,
                metadata=meta,
                owner_id=current_user["uid"]
            )
        except Exception as db_err:
            logger.error(f"Failed to save Q&A transaction in chat history for session {session_id}: {db_err}", exc_info=True)
            # Do not fail endpoint if database log fails (graceful degradation)
            
        return {
            "doc_id": request.doc_id,
            "question": request.question,
            "session_id": session_id,
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

class UserProfileSync(BaseModel):
    display_name: str | None = None

@app.post("/auth/sync", status_code=200)
def sync_user_profile(request: UserProfileSync, current_user: dict = Depends(get_current_user)):
    """
    Creates or updates the user profile record in Firestore/SQLite database.
    """
    try:
        repository.create_or_update_user_profile(
            uid=current_user["uid"],
            email=current_user.get("email"),
            display_name=request.display_name or current_user.get("display_name")
        )
        return {"status": "success", "message": "User profile synchronized successfully."}
    except Exception as e:
        logger.error(f"Failed to sync user profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to sync user profile.")

@app.get("/chat/history/{session_id}", status_code=200)
def get_session_history(session_id: str, current_user: dict = Depends(get_current_user)):
    """
    Retrieves the chat history/message logs for a specific session owned by the user.
    """
    session = repository.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
        
    if session.get("owner_id") != current_user["uid"]:
        raise HTTPException(status_code=403, detail="Access denied to this chat session.")
        
    try:
        history = repository.get_chat_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error retrieving chat history for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history.")

routes_registered_time = time.time() - start_import_time
logger.info(f"API routes registered: {routes_registered_time:.4f}s")


