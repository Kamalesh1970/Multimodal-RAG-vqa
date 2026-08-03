import logging
import json
import time
import os
import uuid
from PIL import Image
from pydantic import BaseModel, Field
from backend.config import settings
from backend.database import get_db_connection
from backend.retrieval import retrieve_evidence, DocumentNotFoundError, IncompleteDocumentError
from backend.generation.gemini_client import generate_content_with_retry
from backend.generation.openai_client import generate_openai_content_with_retry
from backend.generation.context import vlm_call_counter, request_id_var

logger = logging.getLogger(__name__)

# =====================================================================
# Structured Output Pydantic Schemas for VLM /ask response
# =====================================================================

class GeminiEvidenceItem(BaseModel):
    page_number: int = Field(..., description="The page number where the evidence text was found.")
    text: str = Field(..., description="The exact text snippet from the document page supporting the answer.")

class GeminiAnswerResponse(BaseModel):
    answer: str = Field(..., description="The direct grounded answer to the question.")
    answerable: bool = Field(..., description="True if context contains sufficient evidence, False otherwise.")
    grounding_explanation: str = Field(..., description="A short evidence-based explanation of how the answer was derived.")
    pages_used: list[int] = Field(..., description="List of page numbers cited to build this answer.")
    evidence: list[GeminiEvidenceItem] = Field(..., description="Extracts of raw text support from the document.")

# =====================================================================
# Grounding and Intent Utilities
# =====================================================================

from backend.generation.image_preprocessor import prepare_vlm_image, crop_evidence_region

def classify_question_intent(question: str) -> str:
    """
    Classifies question into 'VISUAL' or 'TEXT' using deterministic heuristics.
    """
    visual_keywords = {
        "color", "shape", "chart", "graph", "diagram", "image", "drawing", "plot", 
        "trend", "circle", "square", "rectangle", "ellipse", "blue", "red", "yellow", 
        "green", "black", "white", "line", "drawn", "visual", "look like", "logo", "icon"
    }
    words = [w.strip("?,.:;!\"'").lower() for w in question.split()]
    if any(w in visual_keywords for w in words):
        return "VISUAL"
    return "TEXT"

def select_pages_for_vlm(retrieval_results: list[dict], question: str) -> list[dict]:
    if not retrieval_results:
        return []
        
    summary_keywords = {"summarise", "summarize", "summary", "overview", "compare", "comparison", "both", "all", "entire", "whole", "between", "and", "contact", "signee", "robert"}
    q_lower = question.lower()
    is_multipage = any(kw in q_lower for kw in summary_keywords)
    
    max_pages = getattr(settings, "VLM_MAX_PAGES", 2)
    
    if is_multipage:
        return retrieval_results[:max_pages]
        
    if len(retrieval_results) == 1:
        return retrieval_results
        
    score1 = retrieval_results[0]["scores"]["fused"]
    score2 = retrieval_results[1]["scores"]["fused"]
    gap = score1 - score2
    
    gap_threshold = getattr(settings, "VLM_SCORE_GAP_THRESHOLD", 0.15)
    
    if gap >= gap_threshold:
        return retrieval_results[:1]
    else:
        return retrieval_results[:min(2, max_pages)]

def adjust_top_k_for_intent(question: str, doc_page_count: int, top_k: int | None) -> int:
    """
    Adjusts top_k dynamically based on search query intent.
    If the question implies a summary request, retrieves more pages.
    """
    summary_keywords = {"summarise", "summarize", "summary", "overview", "all pages", "whole document", "entire document"}
    q_lower = question.lower()
    is_summary = any(kw in q_lower for kw in summary_keywords)
    
    if is_summary:
        # Summary requests need more context (up to 10 pages for small documents)
        return max(top_k or 5, min(doc_page_count, 10))
    else:
        return top_k or settings.GENERATION_TOP_K

def verify_grounding_type(answer: str, answerable: bool, pages_used: list[int], retrieval_results: list[dict]) -> str:
    """
    Classifies the grounding source of the answer:
    - 'text-supported' if answer text correlates with OCR snippets
    - 'visual-supported' if answerable=True but answer not found in OCR text (solved visually)
    - 'multimodal-supported' if partially present in OCR text and page images cited
    - 'unverified' if answerable is false or empty context
    """
    if not answerable or not pages_used:
        return "unverified"
        
    ocr_snippets = []
    for r in retrieval_results:
        if r["page_number"] in pages_used:
            ocr_snippets.extend([line.lower() for line in r["evidence_text"]])
            
    ocr_combined = " ".join(ocr_snippets)
    ans_clean = answer.strip().lower()
    
    if ans_clean in ocr_combined:
        return "text-supported"
        
    words = [w.strip("?,.:;!\"'()") for w in ans_clean.split()]
    meaningful_words = [w for w in words if len(w) > 3]
    
    if not meaningful_words:
        return "visual-supported"
        
    matches = sum(1 for w in meaningful_words if w in ocr_combined)
    match_ratio = matches / len(meaningful_words)
    
    if match_ratio >= 0.8:
        return "text-supported"
    elif match_ratio > 0.0:
        return "multimodal-supported"
    else:
        return "visual-supported"

# =====================================================================
# Main RAG Generator Flow
# =====================================================================

def generate_grounded_answer(doc_id: str, question: str, top_k: int | None = None) -> dict:
    """
    Main Phase 5 grounded question generation pipeline:
    1. Validates document status.
    2. Runs Phase 4 retrieval to get top relevant context.
    3. Loads associated PIL page images (scaled to max dimensions).
    4. Formats structured instructions prompt with context budget limits.
    5. Submits to Gemini/OpenAI API requesting structured JSON output.
    6. Performs post-generation grounding check.
    """
    t_start = time.perf_counter()
    
    # Initialize request context
    req_id = str(uuid.uuid4())
    request_id_var.set(req_id)
    vlm_call_counter.set(0)
    
    # 1. Inputs validation
    if not question or not question.strip():
        raise ValueError("Search question cannot be empty or whitespace-only.")
        
    # Check if document exists and is completed
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, page_count FROM documents WHERE doc_id = ?", (doc_id,))
        doc_row = cursor.fetchone()
        
    if not doc_row:
        raise DocumentNotFoundError(f"Document with ID {doc_id} not found.")
        
    status = doc_row["status"]
    page_count = doc_row["page_count"]
    
    if status != "completed":
        raise IncompleteDocumentError(f"Document {doc_id} is in '{status}' status and cannot be questioned yet.")
        
    # 2. Adjust top_k and retrieve evidence
    adjusted_k = adjust_top_k_for_intent(question, page_count, top_k)
    all_retrieved = retrieve_evidence(doc_id, question, top_k=adjusted_k)
    
    # Apply adaptive page selection to restrict context size
    retrieval_results = select_pages_for_vlm(all_retrieved, question)
    
    if not retrieval_results:
        return {
            "answer": "The document contains no pages or retrieval returned no evidence.",
            "answerable": False,
            "grounding_explanation": "No evidence matches the query.",
            "pages_used": [],
            "evidence": [],
            "grounding_type": "unverified",
            "retrieval": {
                "top_score": 0.0,
                "pages_considered": []
            }
        }
        
    # Classify query intent and determine details/crop limits
    intent = classify_question_intent(question)
    if intent == "VISUAL":
        max_images = getattr(settings, "VLM_VISUAL_MAX_IMAGES", 1)
        detail_mode = getattr(settings, "VLM_VISUAL_DETAIL", "high")
    else:
        max_images = getattr(settings, "VLM_TEXT_MAX_IMAGES", 1)
        detail_mode = getattr(settings, "VLM_TEXT_DETAIL", "low")
        
    # Pre-load OCR block coordinates for potential crop operation
    retrieval_blocks = {}
    if intent == "TEXT" and getattr(settings, "VLM_CROP_EVIDENCE", False):
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT page_number, ocr_blocks_json FROM pages WHERE doc_id = ?", (doc_id,))
                for r in cursor.fetchall():
                    blocks = json.loads(r["ocr_blocks_json"]) if r["ocr_blocks_json"] else []
                    retrieval_blocks[r["page_number"]] = blocks
        except Exception as e:
            logger.error(f"Failed to fetch page block coordinates for cropping: {e}")
        
    # 3. Compile prompt text and Base64 images dictionary within budgets
    top_score = retrieval_results[0]["scores"]["fused"] if retrieval_results else 0.0
    pages_considered = [r["page_number"] for r in retrieval_results]
    
    t_api_start = time.perf_counter()
    if settings.VLM_PROVIDER == "local":
        from backend.generation.local_client import generate_local_content
        response_text = generate_local_content(
            question=question,
            retrieval_results=retrieval_results,
            response_schema=GeminiAnswerResponse
        )
    elif settings.VLM_PROVIDER in ("openai", "openrouter"):
        # Compile prompt text with MAX_OCR_CONTEXT_CHARS budget limit
        prompt_text = (
            f"DOCUMENT ID: {doc_id}\n"
            f"USER QUESTION: {question}\n\n"
            "Below are the retrieved context pages. Page numbers are 1-based.\n"
        )
        
        images_dict = {}
        images_attached = 0
        
        for result in retrieval_results:
            page_num = result["page_number"]
            evidence_text = "\n".join(result["evidence_text"])
            
            page_context = (
                f"--- CONTEXT PAGE {page_num} ---\n"
                f"Page Number: {page_num}\n"
                f"OCR Extracted Text:\n{evidence_text}\n\n"
            )
            
            # Stop loading context if it exceeds character limits
            if len(prompt_text) + len(page_context) > settings.MAX_OCR_CONTEXT_CHARS:
                rem = settings.MAX_OCR_CONTEXT_CHARS - len(prompt_text)
                if rem > 100:
                    prompt_text += page_context[:rem] + "\n... [OCR Context Truncated due to character budget] ...\n"
                break
            prompt_text += page_context
            
            # Conditionally attach images under max images limit
            if images_attached < max_images:
                image_path = settings.PROCESSED_DIR / doc_id / f"page_{page_num}.jpg"
                if image_path.exists():
                    try:
                        img = Image.open(image_path)
                        
                        # Apply layout cropping around matched lines if enabled
                        bboxes = []
                        if intent == "TEXT" and getattr(settings, "VLM_CROP_EVIDENCE", False):
                            page_blocks = retrieval_blocks.get(page_num, [])
                            for line in result["evidence_text"]:
                                line_lower = line.lower().strip()
                                for b in page_blocks:
                                    text_lower = b.get("text", "").lower().strip()
                                    if line_lower in text_lower or text_lower in line_lower:
                                        if b.get("bbox"):
                                            bboxes.append(b["bbox"])
                                            
                        if bboxes:
                            img = crop_evidence_region(img, bboxes)
                            
                        img = prepare_vlm_image(img, settings.MAX_IMAGE_DIMENSION)
                        images_dict[page_num] = img
                        images_attached += 1
                    except Exception as e:
                        logger.error(f"Failed to load/optimize image for Page {page_num}: {e}")
                else:
                    logger.warning(f"Expected page image file not found: {image_path}")
                    
        prompt_text += (
            "\nProvide your final answer as JSON matching the requested schema. "
            "Remember, do not answer from outside knowledge if evidence is missing."
        )
        
        # Invoke OpenAI / OpenRouter API passing details mode
        response_text = generate_openai_content_with_retry(
            prompt_text=prompt_text,
            images_dict=images_dict,
            response_schema=GeminiAnswerResponse,
            detail=detail_mode
        )
    else:
        # Compile Gemini multimodal prompt list
        contents = []
        contents.append(f"DOCUMENT ID: {doc_id}\n")
        contents.append(f"USER QUESTION: {question}\n\n")
        contents.append("Below are the retrieved context pages. Page numbers are 1-based.\n")
        
        prompt_char_count = 0
        images_attached = 0
        
        for result in retrieval_results:
            page_num = result["page_number"]
            evidence_text = "\n".join(result["evidence_text"])
            
            page_context = (
                f"--- CONTEXT PAGE {page_num} ---\n"
                f"Page Number: {page_num}\n"
                f"OCR Extracted Text:\n{evidence_text}\n\n"
            )
            
            # Budget check
            if prompt_char_count + len(page_context) > settings.MAX_OCR_CONTEXT_CHARS:
                rem = settings.MAX_OCR_CONTEXT_CHARS - prompt_char_count
                if rem > 100:
                    contents.append(page_context[:rem] + "\n... [OCR Context Truncated due to character budget] ...\n")
                break
                
            contents.append(page_context)
            prompt_char_count += len(page_context)
            
            # Conditionally attach images under max images limit
            if images_attached < max_images:
                image_path = settings.PROCESSED_DIR / doc_id / f"page_{page_num}.jpg"
                if image_path.exists():
                    try:
                        img = Image.open(image_path)
                        
                        # Apply layout cropping around matched lines if enabled
                        bboxes = []
                        if intent == "TEXT" and getattr(settings, "VLM_CROP_EVIDENCE", False):
                            page_blocks = retrieval_blocks.get(page_num, [])
                            for line in result["evidence_text"]:
                                line_lower = line.lower().strip()
                                for b in page_blocks:
                                    text_lower = b.get("text", "").lower().strip()
                                    if line_lower in text_lower or text_lower in line_lower:
                                        if b.get("bbox"):
                                            bboxes.append(b["bbox"])
                                            
                        if bboxes:
                            img = crop_evidence_region(img, bboxes)
                            
                        img = prepare_vlm_image(img, settings.MAX_IMAGE_DIMENSION)
                        contents.append(f"[Page {page_num} Layout Image]")
                        contents.append(img)
                        images_attached += 1
                    except Exception as e:
                        logger.error(f"Failed to load/optimize image for Page {page_num}: {e}")
                else:
                    logger.warning(f"Expected page image file not found: {image_path}")
                    
        contents.append(
            "\nProvide your final answer as JSON matching the requested schema. "
            "Remember, do not answer from outside knowledge if evidence is missing."
        )
        
        # Invoke Gemini API
        response_text = generate_content_with_retry(contents, response_schema=GeminiAnswerResponse)
        
    t_api = time.perf_counter() - t_api_start
    
    # 5. Parse and validate response text
    try:
        data = json.loads(response_text)
        validated_resp = GeminiAnswerResponse(**data)
    except Exception as e:
        logger.error(f"{settings.VLM_PROVIDER.upper()} structured JSON parsing or validation failed: {e}. Raw: {response_text}")
        raise ValueError(f"Failed to parse and validate {settings.VLM_PROVIDER.upper()} response: {e}")
        
    # 6. Verify grounding classification
    grounding_type = verify_grounding_type(
        answer=validated_resp.answer,
        answerable=validated_resp.answerable,
        pages_used=validated_resp.pages_used,
        retrieval_results=retrieval_results
    )
    
    # Align page citation evidence coordinates from original retrieval output
    evidence_output = []
    retrieval_blocks = {}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT page_number, ocr_blocks_json FROM pages WHERE doc_id = ?", (doc_id,))
        for r in cursor.fetchall():
            try:
                blocks = json.loads(r["ocr_blocks_json"]) if r["ocr_blocks_json"] else []
                retrieval_blocks[r["page_number"]] = blocks
            except Exception:
                retrieval_blocks[r["page_number"]] = []
                
    for item in validated_resp.evidence:
        bbox = None
        blocks = retrieval_blocks.get(item.page_number, [])
        item_text_lower = item.text.lower().strip()
        for b in blocks:
            text = b.get("text", "").lower().strip()
            if item_text_lower in text or text in item_text_lower:
                bbox = b.get("bbox")
                break
        evidence_output.append({
            "page_number": item.page_number,
            "text": item.text,
            "bbox": bbox
        })
        
    t_total = time.perf_counter() - t_start
    logger.info(
        f"Answer generation completed: doc_id={doc_id}, question_len={len(question)}, "
        f"answerable={validated_resp.answerable}, pages_used={validated_resp.pages_used}, "
        f"grounding={grounding_type}, latency={t_total:.4f}s (VLM API: {t_api:.4f}s)"
    )
    
    # Get request call counter count
    calls_count = vlm_call_counter.get()
    
    response_dict = {
        "answer": validated_resp.answer,
        "answerable": validated_resp.answerable,
        "grounding_explanation": validated_resp.grounding_explanation,
        "pages_used": validated_resp.pages_used,
        "grounding_type": grounding_type,
        "evidence": evidence_output,
        "retrieval": {
            "top_score": float(top_score),
            "pages_considered": pages_considered
        }
    }
    
    # Return debugging metadata if active
    # We check a setting property or config flag VLM_DEBUG
    # Let's map it via settings.VLM_DEBUG or a debug query setting.
    # To keep it generic, we check if settings has VLM_DEBUG or from environment.
    # Since VLM_DEBUG is optional, let's look at config or env directly:
    vlm_debug = os.getenv("VLM_DEBUG", "false").lower() == "true"
    if vlm_debug:
        response_dict["debug"] = {
            "provider_calls": calls_count,
            "request_id": req_id
        }
        
    return response_dict
