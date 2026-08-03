import io
import json
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image, ImageOps
import fitz  # PyMuPDF
import numpy as np

from backend.config import settings
from backend.database import get_db_connection
from backend.ingestion.ocr import perform_ocr, PageOCRResult

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

class IngestionError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)

def validate_file(file_bytes: bytes, filename: str) -> str:
    """
    Validates file extension and readable content.
    Returns the normalized extension.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Validation failed: unsupported extension {ext} for file {filename}")
        raise IngestionError(400, "Unsupported file type.")
        
    # Check if empty file
    if len(file_bytes) == 0:
        logger.warning(f"Validation failed: file {filename} is empty")
        raise IngestionError(400, "Empty file.")

    if ext == ".pdf":
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page_count = doc.page_count
            if page_count == 0:
                logger.warning(f"Validation failed: PDF {filename} has 0 pages")
                raise IngestionError(400, "Invalid PDF: no pages found.")
            if doc.is_encrypted:
                logger.warning(f"Validation failed: PDF {filename} is encrypted")
                raise IngestionError(400, "Encrypted PDFs are not supported.")
        except Exception as e:
            if isinstance(e, IngestionError):
                raise e
            logger.warning(f"Validation failed: PDF {filename} is corrupted: {e}")
            raise IngestionError(400, "Corrupted PDF file.")
    else:  # Image formats
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()  # Verify image integrity
        except Exception as e:
            logger.warning(f"Validation failed: Image {filename} is corrupted: {e}")
            raise IngestionError(400, "Corrupted image file.")
            
    return ext

def preprocess_image(pil_img: Image.Image) -> Image.Image:
    """
    Applies minimal, safe image preprocessing operations:
    1. EXIF orientation correction.
    2. Convert to RGB mode.
    """
    # 1. EXIF orientation transpose
    pil_img = ImageOps.exif_transpose(pil_img)
    
    # 2. RGB conversion (removes alpha layer, handles grayscale cleanly)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
        
    return pil_img

def process_image_file(stored_path: Path, doc_id: str) -> List[PageOCRResult]:
    """
    Loads and runs OCR on a standard image file.
    """
    try:
        pil_img = Image.open(stored_path)
    except Exception as e:
        logger.error(f"Failed to open saved image: {e}")
        raise IngestionError(400, "Invalid image format on disk.")

    pil_img = preprocess_image(pil_img)
    width, height = pil_img.size
    
    # Save the preprocessed image to processed dir for Phase 3 visual embedding
    out_dir = settings.PROCESSED_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    page_image_path = out_dir / "page_1.jpg"
    pil_img.save(page_image_path, "JPEG", quality=85)
    
    img_arr = np.array(pil_img)
    
    logger.info(f"Running OCR on image {stored_path.name} ({width}x{height})...")
    ocr_start = time.perf_counter()
    page_result = perform_ocr(img_arr, page_number=1, width=width, height=height)
    ocr_time = time.perf_counter() - ocr_start
    
    logger.info(f"OCR page 1 completed. Blocks={len(page_result.blocks)}, processing_time={ocr_time:.2f}s")
    return [page_result]

def process_pdf_file(stored_path: Path, doc_id: str) -> List[PageOCRResult]:
    """
    Renders PDF pages to images at 200 DPI, runs preprocessing and OCR on each page.
    """
    try:
        doc = fitz.open(stored_path)
    except Exception as e:
        logger.error(f"Failed to open saved PDF: {e}")
        raise IngestionError(400, "Invalid PDF format on disk.")

    page_results = []
    
    # Render resolution matrix (200 DPI)
    zoom = 200.0 / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for i in range(len(doc)):
        page_number = i + 1
        page = doc[i]
        
        logger.info(f"Rendering PDF page {page_number}/{len(doc)} at 200 DPI...")
        render_start = time.perf_counter()
        try:
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")
            render_time = time.perf_counter() - render_start
            
            # Load in Pillow
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img = preprocess_image(pil_img)
            width, height = pil_img.size
            
            # Save preprocessed page image to processed dir for Phase 3 visual embedding
            out_dir = settings.PROCESSED_DIR / doc_id
            out_dir.mkdir(parents=True, exist_ok=True)
            page_image_path = out_dir / f"page_{page_number}.jpg"
            pil_img.save(page_image_path, "JPEG", quality=85)
            
            img_arr = np.array(pil_img)
        except Exception as render_err:
            logger.error(f"Failed to render page {page_number}: {render_err}")
            raise IngestionError(500, f"Error rendering PDF page {page_number}.")
            
        logger.info(f"Running OCR on PDF page {page_number} ({width}x{height}). Render time: {render_time:.2f}s")
        ocr_start = time.perf_counter()
        page_result = perform_ocr(img_arr, page_number=page_number, width=width, height=height)
        ocr_time = time.perf_counter() - ocr_start
        
        logger.info(f"OCR page {page_number} completed. Blocks={len(page_result.blocks)}, processing_time={ocr_time:.2f}s")
        page_results.append(page_result)
        
    return page_results

def save_to_database(doc_id: str, filename: str, stored_path: str, file_type: str, page_results: List[PageOCRResult]) -> List[dict]:
    """
    Saves document metadata and structured page OCR results inside a database transaction.
    Returns a list of dictionaries with inserted page primary key ids and page metadata.
    """
    inserted_pages = []
    with get_db_connection() as conn:
        # Update documents table status and file details
        conn.execute(
            """
            UPDATE documents 
            SET stored_path = ?, page_count = ?, status = 'completed'
            WHERE doc_id = ?
            """,
            (stored_path, len(page_results), doc_id)
        )
        
        # Save pages to pages table
        for pr in page_results:
            blocks_json = json.dumps([b.model_dump() for b in pr.blocks])
            cursor = conn.execute(
                """
                INSERT INTO pages (doc_id, page_number, width, height, ocr_text, ocr_blocks_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, pr.page_number, pr.width, pr.height, pr.full_text, blocks_json)
            )
            inserted_pages.append({
                "page_id": cursor.lastrowid,
                "page_number": pr.page_number,
                "full_text": pr.full_text
            })
    return inserted_pages

def ingest_document(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    E2E Ingestion Orchestration Pipeline:
    1. Validation (extension, size, corrupt check).
    2. Write original file securely using generated UUID name.
    3. Initialize document record in DB with 'processing' status.
    4. Execute page rendering / image preprocessing & OCR processing.
    5. Save results to SQLite database, updating status to 'completed'.
    6. Returns structured API response.
    """
    total_start = time.perf_counter()
    
    # 1. Size Validation (Check size first)
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if len(file_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        logger.warning(f"Upload rejected: {filename} size {file_size_mb:.2f}MB exceeds limit of {settings.MAX_UPLOAD_MB}MB.")
        raise IngestionError(413, f"File too large. Maximum upload limit is {settings.MAX_UPLOAD_MB}MB.")
        
    # 2. Format Validation
    val_start = time.perf_counter()
    ext = validate_file(file_bytes, filename)
    val_time = time.perf_counter() - val_start
    
    # 3. Secure File Name & Write
    doc_id = str(uuid.uuid4())
    stored_filename = f"{doc_id}{ext}"
    stored_path = settings.UPLOADS_DIR / stored_filename
    
    logger.info(f"Ingesting document: doc_id={doc_id}, original_filename={filename}, stored_path={stored_path}")
    
    try:
        # Write bytes securely to disk
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        with open(stored_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        logger.error(f"Failed to write uploaded file to disk: {e}", exc_info=True)
        raise IngestionError(500, "Failed to save uploaded file securely.")

    # 4. Insert initial DB row in 'processing' status
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, filename, stored_path, file_type, page_count, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, filename, str(stored_path), ext.replace(".", ""), 0, "processing")
            )
    except Exception as e:
        logger.critical(f"Database insertion failed: {e}", exc_info=True)
        # Delete file if DB insert fails
        if stored_path.exists():
            stored_path.unlink()
        raise IngestionError(500, "Database insertion error.")

    # 5. Process Ingestion & Run OCR
    try:
        if ext == ".pdf":
            page_results = process_pdf_file(stored_path, doc_id)
        else:
            page_results = process_image_file(stored_path, doc_id)
            
        # 6. Save OCR Pages to database & complete document status
        inserted_pages = save_to_database(doc_id, filename, str(stored_path), ext.replace(".", ""), page_results)
        
        # 7. Generate embeddings and index them (Phase 3)
        try:
            from backend.embeddings.text_embedder import TextEmbedder
            from backend.embeddings.image_embedder import ImageEmbedder
            from backend.vector_store import VectorStore
            
            for page_info in inserted_pages:
                page_id = page_info["page_id"]
                page_number = page_info["page_number"]
                full_text = page_info["full_text"]
                
                # Path to the saved page image
                page_image_path = settings.PROCESSED_DIR / doc_id / f"page_{page_number}.jpg"
                
                # Text Embedding
                text_indexed = 0
                if full_text and full_text.strip():
                    try:
                        text_vec = TextEmbedder.embed_text(full_text)
                        if text_vec is not None:
                            VectorStore.add_text_vector(page_id, text_vec)
                            text_indexed = 1
                    except Exception as te_err:
                        logger.error(f"Failed to generate text embedding for page {page_id}: {te_err}", exc_info=True)
                
                # Image Embedding
                image_indexed = 0
                if page_image_path.exists():
                    try:
                        pil_img = Image.open(page_image_path)
                        image_vec = ImageEmbedder.embed_image(pil_img)
                        VectorStore.add_image_vector(page_id, image_vec)
                        image_indexed = 1
                    except Exception as ie_err:
                        logger.error(f"Failed to generate image embedding for page {page_id}: {ie_err}", exc_info=True)
                else:
                    logger.warning(f"Page image path {page_image_path} does not exist. Skipping image embedding.")
                
                # Update database record with indexing status and models
                with get_db_connection() as conn:
                    conn.execute(
                        """
                        UPDATE pages
                        SET text_embedding_indexed = ?,
                            image_embedding_indexed = ?,
                            text_embedding_model = ?,
                            image_embedding_model = ?
                        WHERE id = ?
                        """,
                        (
                            text_indexed,
                            image_indexed,
                            settings.TEXT_EMBEDDING_MODEL if text_indexed else None,
                            settings.IMAGE_EMBEDDING_MODEL if image_indexed else None,
                            page_id
                        )
                    )
            
            # Save FAISS indices to disk after batch is complete
            VectorStore.save_indices()
        except Exception as emb_err:
            logger.error(f"Embedding/Vector indexing failed for doc_id={doc_id}: {emb_err}", exc_info=True)
        
    except Exception as process_error:
        # Update database status to failed
        logger.error(f"Ingestion process failed for doc_id={doc_id}: {process_error}", exc_info=True)
        try:
            with get_db_connection() as conn:
                conn.execute(
                    "UPDATE documents SET status = 'failed' WHERE doc_id = ?",
                    (doc_id,)
                )
        except Exception as db_err:
            logger.critical(f"Failed to set status to failed for doc_id={doc_id}: {db_err}")
            
        if isinstance(process_error, IngestionError):
            raise process_error
        raise IngestionError(500, f"Processing/OCR pipeline error: {str(process_error)}")
        
    total_time = time.perf_counter() - total_start
    logger.info(f"Document ingestion completed successfully for doc_id={doc_id} in {total_time:.2f}s (Validation: {val_time:.4f}s).")
    
    # Return structured API response matching requirements
    pages_response = []
    for pr in page_results:
        pages_response.append({
            "page_number": pr.page_number,
            "width": pr.width,
            "height": pr.height,
            "ocr": {
                "full_text": pr.full_text,
                "blocks": [
                    {
                        "text": b.text,
                        "confidence": b.confidence,
                        "bbox": b.bbox
                    } for b in pr.blocks
                ]
            }
        })
        
    return {
        "doc_id": doc_id,
        "filename": filename,
        "file_type": ext.replace(".", ""),
        "status": "completed",
        "pages": pages_response
    }
