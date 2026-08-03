import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import logging
import numpy as np
from typing import List, Dict
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class OCRBlock(BaseModel):
    text: str
    confidence: float
    bbox: List[List[float]] = Field(..., description="List of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]")

class PageOCRResult(BaseModel):
    page_number: int
    width: int
    height: int
    blocks: List[OCRBlock]
    full_text: str

class OCREngine:
    _instances: Dict[str, object] = {}

    @classmethod
    def get_instance(cls, lang: str = "en"):
        """
        Lazy-initializes and caches a PaddleOCR engine instance for the requested language.
        """
        if lang not in cls._instances:
            logger.info(f"Initializing PaddleOCR engine for language '{lang}' (lazy-initialization)...")
            try:
                # Import locally to prevent model download/startup errors at global import time
                from paddleocr import PaddleOCR
                cls._instances[lang] = PaddleOCR(use_angle_cls=True, lang=lang, enable_mkldnn=False)
                logger.info("OCR engine initialized")
            except Exception as e:
                logger.critical(f"Failed to initialize PaddleOCR engine for '{lang}': {e}", exc_info=True)
                raise e
        return cls._instances[lang]

def perform_ocr(img_arr: np.ndarray, page_number: int, width: int, height: int, lang: str = "en") -> PageOCRResult:
    """
    Runs PaddleOCR on the provided image numpy array and normalizes results into the standard PageOCRResult format.
    """
    engine = OCREngine.get_instance(lang=lang)
    
    try:
        # Run PaddleOCR with classification enabled to handle orientation issues
        result = engine.ocr(img_arr)
    except Exception as e:
        logger.error(f"PaddleOCR execution failed on page {page_number}: {e}", exc_info=True)
        raise e

    blocks = []
    full_text_parts = []

    logger.info(f"Raw PaddleOCR result type: {type(result)}")
    if isinstance(result, list) and len(result) > 0:
        first_res = result[0]
        logger.info(f"Raw PaddleOCR result[0] type: {type(first_res)}")

        # Determine format
        if isinstance(first_res, (list, tuple)):
            # Legacy format parsing
            logger.info("Parsing PaddleOCR legacy list-of-lines format.")
            for line in first_res:
                try:
                    bbox = [[float(coord[0]), float(coord[1])] for coord in line[0]]
                    text = str(line[1][0])
                    confidence = float(line[1][1])
                    
                    blocks.append(OCRBlock(
                        text=text,
                        confidence=confidence,
                        bbox=bbox
                    ))
                    full_text_parts.append(text)
                except (IndexError, ValueError, TypeError) as parse_err:
                    logger.warning(f"Failed to parse legacy OCR line {line} on page {page_number}: {parse_err}")
        else:
            logger.info("Parsing PaddleOCR new format (dict-like or custom object).")
            
            # Try to get the dict representation
            result_dict = None
            if isinstance(first_res, dict):
                result_dict = first_res
            elif hasattr(first_res, "json"):
                json_val = first_res.json
                if callable(json_val):
                    try:
                        json_val = json_val()
                    except Exception:
                        pass
                if isinstance(json_val, dict):
                    result_dict = json_val
                elif isinstance(json_val, str):
                    try:
                        import json
                        result_dict = json.loads(json_val)
                    except Exception:
                        pass
            
            if result_dict is None:
                result_dict = first_res
            
            # Read fields from result_dict
            try:
                if isinstance(result_dict, dict):
                    rec_texts = result_dict.get("rec_texts", [])
                    rec_scores = result_dict.get("rec_scores", [])
                    rec_boxes = result_dict.get("rec_boxes", result_dict.get("rec_polys", []))
                else:
                    rec_texts = getattr(result_dict, "rec_texts", [])
                    rec_scores = getattr(result_dict, "rec_scores", [])
                    rec_boxes = getattr(result_dict, "rec_boxes", getattr(result_dict, "rec_polys", []))
                
                logger.info(f"Extracted from new format: texts_len={len(rec_texts)}, scores_len={len(rec_scores)}, boxes_len={len(rec_boxes)}")
                
                for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                    bbox = []
                    if isinstance(box, list):
                        if len(box) == 4 and all(isinstance(pt, list) and len(pt) == 2 for pt in box):
                            bbox = [[float(pt[0]), float(pt[1])] for pt in box]
                        elif len(box) == 8:
                            bbox = [
                                [float(box[0]), float(box[1])],
                                [float(box[2]), float(box[3])],
                                [float(box[4]), float(box[5])],
                                [float(box[6]), float(box[7])]
                            ]
                        else:
                            bbox = [[float(pt[0]), float(pt[1])] for pt in box if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                    
                    if not bbox or len(bbox) != 4:
                        bbox = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
                    
                    blocks.append(OCRBlock(
                        text=str(text),
                        confidence=float(score),
                        bbox=bbox
                    ))
                    full_text_parts.append(str(text))
            except Exception as parse_err:
                logger.error(f"Failed to parse new format OCR results: {parse_err}", exc_info=True)
    else:
        logger.warning(f"PaddleOCR returned empty or unexpected result type: {result}")

    # Combine blocks to page level full_text in reading order
    full_text = "\n".join(full_text_parts)
    
    logger.info(f"OCR execution completed: extracted {len(blocks)} blocks. Full text length: {len(full_text)}")
    if len(blocks) == 0:
        logger.warning(f"OCR extracted 0 blocks. Raw result was: {result}")

    return PageOCRResult(
        page_number=page_number,
        width=width,
        height=height,
        blocks=blocks,
        full_text=full_text
    )
