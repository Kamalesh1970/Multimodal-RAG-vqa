import logging
import json
import time
import numpy as np
from backend.config import settings
from backend.database import get_db_connection
from backend.embeddings.text_embedder import TextEmbedder
from backend.embeddings.image_embedder import ImageEmbedder
from backend.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DocumentNotFoundError(Exception):
    """Raised when a doc_id does not exist in the database."""
    pass

class IncompleteDocumentError(Exception):
    """Raised when a document is in processing or failed state."""
    pass

def extract_evidence(question: str, blocks: list, max_blocks: int = 3) -> list[str]:
    """
    Selects blocks containing words relevant to the normalized question terms,
    excluding basic English stopwords.
    """
    if not blocks:
        return []
        
    # Basic normalization and stopword removal
    stopwords = {
        "what", "is", "the", "of", "in", "and", "to", "a", "for", "on", "with", 
        "at", "by", "an", "this", "which", "page", "contains", "are", "you", 
        "we", "he", "she", "it", "they", "how", "many", "show", "me", "find"
    }
    words = [w.strip("?,.:;!\"'").lower() for w in question.split()]
    keywords = [w for w in words if w and w not in stopwords]
    
    if not keywords:
        keywords = [w for w in words if w]
        
    scored_blocks = []
    for block in blocks:
        text = block.get("text", "")
        if not text:
            continue
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            scored_blocks.append((matches, text))
            
    # Sort blocks by matches descending
    scored_blocks.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored_blocks[:max_blocks]]

def retrieve_evidence(doc_id: str, question: str, top_k: int | None = None) -> list[dict]:
    """
    Performs document-isolated multimodal retrieval.
    Queries both text and image FAISS indexes, aligns candidate scores via SQLite,
    applies Weighted Available Modalities (WAM) fusion, deterministically ranks pages,
    and extracts relevant evidence blocks.
    """
    t_start = time.perf_counter()
    
    # 1. Query Validation
    if not question or not question.strip():
        raise ValueError("Search question cannot be empty or whitespace-only.")
        
    if len(question) > 500:
        raise ValueError(f"Search question length exceeds maximum limit of 500 characters. Got {len(question)}.")
        
    # 2. Verify document exists and is completed
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, filename FROM documents WHERE doc_id = ?", (doc_id,))
        doc_row = cursor.fetchone()
        
    if not doc_row:
        raise DocumentNotFoundError(f"Document with ID {doc_id} not found.")
        
    status = doc_row["status"]
    if status != "completed":
        raise IncompleteDocumentError(f"Document {doc_id} is in '{status}' status and cannot be searched yet.")
        
    # 3. Retrieve all pages for this document from SQLite
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, page_number, ocr_text, ocr_blocks_json FROM pages WHERE doc_id = ? ORDER BY page_number ASC",
            (doc_id,)
        )
        page_rows = cursor.fetchall()
        
    if not page_rows:
        logger.warning(f"Document {doc_id} has no pages in the database.")
        return []
        
    # Create page lookups
    pages_dict = {}
    for r in page_rows:
        pages_dict[r["id"]] = {
            "page_id": r["id"],
            "page_number": r["page_number"],
            "ocr_text": r["ocr_text"],
            "ocr_blocks_json": r["ocr_blocks_json"]
        }
        
    # 4. Generate Embeddings (measure performance)
    t_text_embed_start = time.perf_counter()
    text_query_vec = TextEmbedder.embed_text(question)
    t_text_embed = time.perf_counter() - t_text_embed_start
    
    t_clip_embed_start = time.perf_counter()
    clip_query_vec = ImageEmbedder.embed_text(question)
    t_clip_embed = time.perf_counter() - t_clip_embed_start
    
    # Expose total candidate capacity of the indexes for document-level filtering
    text_status = VectorStore.get_status()
    total_text_vectors = text_status["text_vectors"]
    total_image_vectors = text_status["image_vectors"]
    
    # 5. Search FAISS text index
    t_text_search_start = time.perf_counter()
    text_results = []
    if text_query_vec is not None and total_text_vectors > 0:
        text_results = VectorStore.search_text_index(text_query_vec, total_text_vectors)
    t_text_search = time.perf_counter() - t_text_search_start
        
    # 6. Search FAISS image index
    t_image_search_start = time.perf_counter()
    image_results = []
    if clip_query_vec is not None and total_image_vectors > 0:
        image_results = VectorStore.search_image_index(clip_query_vec, total_image_vectors)
    t_image_search = time.perf_counter() - t_image_search_start
        
    # Convert search results to lookups
    text_scores = {pid: score for pid, score in text_results}
    image_scores = {pid: score for pid, score in image_results}
    
    # 7. Perform Score Fusion & Validation
    t_fusion_start = time.perf_counter()
    
    # Calculate min-max boundaries within this document's candidate pages to calibrate distributions
    text_scores_list = [text_scores.get(pid) for pid in pages_dict if text_scores.get(pid) is not None]
    image_scores_list = [image_scores.get(pid) for pid in pages_dict if image_scores.get(pid) is not None]
    
    text_min = min(text_scores_list) if text_scores_list else 0.0
    text_max = max(text_scores_list) if text_scores_list else 1.0
    text_range = text_max - text_min if text_max != text_min else 1.0
    
    image_min = min(image_scores_list) if image_scores_list else 0.0
    image_max = max(image_scores_list) if image_scores_list else 1.0
    image_range = image_max - image_min if image_max != image_min else 1.0
    
    candidates = []
    
    for page_id, p_info in pages_dict.items():
        text_score = text_scores.get(page_id)
        image_score = image_scores.get(page_id)
        
        # Check index consistency and log orphans or mismatches
        # If DB says a page exists, but its expected vector is missing from FAISS indexes:
        if text_score is None and p_info["ocr_text"].strip():
            # Only log a warning if text index has items, indicating an index/db sync issue
            if total_text_vectors > 0:
                logger.warning(f"Database page ID {page_id} contains OCR text but is not found in FAISS text index (orphan/consistency error).")
        if image_score is None and total_image_vectors > 0:
            logger.warning(f"Database page ID {page_id} visual embedding is missing from FAISS image index (orphan/consistency error).")
            
        matched_modalities = []
        scores_dict = {}
        
        # Collect weights and normalized scores for WAM fusion
        weights = []
        scores = []
        
        if text_score is not None:
            matched_modalities.append("text")
            scores_dict["text"] = text_score
            norm_text = (text_score - text_min) / text_range
            weights.append(settings.TEXT_RETRIEVAL_WEIGHT)
            scores.append(norm_text)
        else:
            scores_dict["text"] = None
            
        if image_score is not None:
            matched_modalities.append("image")
            scores_dict["image"] = image_score
            norm_image = (image_score - image_min) / image_range
            weights.append(settings.IMAGE_RETRIEVAL_WEIGHT)
            scores.append(norm_image)
        else:
            scores_dict["image"] = None
            
        # WAM score fusion using calibrated normalized scores
        if weights:
            fused_score = sum(w * s for w, s in zip(weights, scores)) / sum(weights)
        else:
            fused_score = 0.0
            
        scores_dict["fused"] = fused_score
        
        # Parse blocks safely
        blocks = []
        try:
            if p_info["ocr_blocks_json"]:
                blocks = json.loads(p_info["ocr_blocks_json"])
        except Exception as e:
            logger.error(f"Failed to parse OCR blocks JSON for page ID {page_id}: {e}")
            
        evidence_text = extract_evidence(question, blocks)
        
        candidates.append({
            "page_id": page_id,
            "page_number": p_info["page_number"],
            "scores": scores_dict,
            "matched_modalities": matched_modalities,
            "evidence_text": evidence_text
        })
        
    # Filter candidates by minimum score
    filtered_candidates = [c for c in candidates if c["scores"]["fused"] >= settings.RETRIEVAL_MIN_SCORE]
    
    # 8. Sort deterministically: by fused score descending, then page number ascending, then page ID ascending
    filtered_candidates.sort(key=lambda x: (-x["scores"]["fused"], x["page_number"], x["page_id"]))
    t_fusion = time.perf_counter() - t_fusion_start
    
    t_total = time.perf_counter() - t_start
    
    # 9. Extract top_k
    k_limit = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
    top_results = filtered_candidates[:k_limit]
    
    # 10. Log diagnostic research metrics
    top_score = top_results[0]["scores"]["fused"] if top_results else 0.0
    logger.info(
        f"Retrieval completed: question_len={len(question)}, doc_id={doc_id}, "
        f"candidates={len(candidates)}, filtered={len(filtered_candidates)}, "
        f"top_score={top_score:.4f}, duration={t_total:.4f}s"
    )
    
    logger.info(
        f"Performance Breakdown: text_embed={t_text_embed:.4f}s, clip_embed={t_clip_embed:.4f}s, "
        f"text_search={t_text_search:.4f}s, image_search={t_image_search:.4f}s, "
        f"fusion={t_fusion:.4f}s"
    )
    
    return top_results
