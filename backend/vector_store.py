import logging
import os
from pathlib import Path
import faiss
import numpy as np
from backend.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    _text_index = None
    _image_index = None
    _text_dim = None
    _image_dim = None

    @classmethod
    def get_text_index_path(cls) -> Path:
        return settings.VECTOR_INDEX_DIR / "text.index"

    @classmethod
    def get_image_index_path(cls) -> Path:
        return settings.VECTOR_INDEX_DIR / "image.index"

    @classmethod
    def initialize(cls, text_dim: int, image_dim: int):
        """
        Initializes or loads FAISS indexes. Must be called with the expected dimensions.
        """
        cls._text_dim = text_dim
        cls._image_dim = image_dim
        
        # Ensure target index directories exist
        settings.VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        
        cls.load_indices()

    @classmethod
    def load_indices(cls):
        """
        Loads indexes from disk if they exist, otherwise creates fresh flat exact ID indexes.
        """
        text_path = cls.get_text_index_path()
        image_path = cls.get_image_index_path()
        
        cls._text_index = None
        cls._image_index = None
        
        # Load text index
        if text_path.exists():
            logger.info(f"Loading existing FAISS text index from {text_path}...")
            try:
                cls._text_index = faiss.read_index(str(text_path))
                loaded_dim = cls._text_index.d
                if loaded_dim != cls._text_dim:
                    logger.error(f"Text index dimension mismatch! Expected {cls._text_dim}, loaded {loaded_dim}. Recreating index.")
                    cls._text_index = None
            except Exception as e:
                logger.error(f"Failed to load text index from disk: {e}. Recreating index.")
                cls._text_index = None
                
        if cls._text_index is None:
            logger.info(f"Creating new FAISS text index (IndexFlatIP + IndexIDMap2) with dimension {cls._text_dim}...")
            quantizer = faiss.IndexFlatIP(cls._text_dim)
            cls._text_index = faiss.IndexIDMap2(quantizer)
            
        # Load image index
        if image_path.exists():
            logger.info(f"Loading existing FAISS image index from {image_path}...")
            try:
                cls._image_index = faiss.read_index(str(image_path))
                loaded_dim = cls._image_index.d
                if loaded_dim != cls._image_dim:
                    logger.error(f"Image index dimension mismatch! Expected {cls._image_dim}, loaded {loaded_dim}. Recreating index.")
                    cls._image_index = None
            except Exception as e:
                logger.error(f"Failed to load image index from disk: {e}. Recreating index.")
                cls._image_index = None
                
        if cls._image_index is None:
            logger.info(f"Creating new FAISS image index (IndexFlatIP + IndexIDMap2) with dimension {cls._image_dim}...")
            quantizer = faiss.IndexFlatIP(cls._image_dim)
            cls._image_index = faiss.IndexIDMap2(quantizer)

    @classmethod
    def save_indices(cls):
        """
        Saves both FAISS indexes to disk.
        """
        text_path = cls.get_text_index_path()
        image_path = cls.get_image_index_path()
        
        settings.VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        
        if cls._text_index is not None:
            logger.info(f"Saving FAISS text index to {text_path}...")
            faiss.write_index(cls._text_index, str(text_path))
            
        if cls._image_index is not None:
            logger.info(f"Saving FAISS image index to {image_path}...")
            faiss.write_index(cls._image_index, str(image_path))

    @classmethod
    def validate_vector(cls, vector: np.ndarray, expected_dim: int) -> np.ndarray:
        """
        Validates vector type, values, norm, and dimensions.
        Converts input to float32 2D NumPy array with shape (1, dim).
        """
        if not isinstance(vector, np.ndarray):
            raise TypeError("Vector must be a numpy ndarray.")
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        if vector.ndim != 1 and (vector.ndim != 2 or vector.shape[0] != 1):
            raise ValueError(f"Vector must be 1D or 2D with shape (1, dim). Got shape {vector.shape}.")
        
        v = vector.flatten()
        
        if len(v) != expected_dim:
            raise ValueError(f"Vector dimension mismatch. Expected {expected_dim}, got {len(v)}.")
        if np.isnan(v).any():
            raise ValueError("Vector contains NaN values.")
        if np.isinf(v).any():
            raise ValueError("Vector contains Inf values.")
            
        norm = np.linalg.norm(v)
        if np.isclose(norm, 0.0):
            raise ValueError("Vector has zero norm.")
            
        return v.reshape(1, -1)

    @classmethod
    def add_text_vector(cls, page_id: int, vector: np.ndarray):
        """
        Validates and adds a text vector mapped to pages.id integer key in FAISS text index.
        """
        if cls._text_index is None:
            raise RuntimeError("Text index is not initialized. Call initialize() first.")
            
        v = cls.validate_vector(vector, cls._text_dim)
        ids = np.array([page_id], dtype=np.int64)
        
        if cls.has_text_vector(page_id):
            logger.warning(f"Page ID {page_id} already exists in FAISS text index. Overwriting vector.")
            cls.remove_text_vector(page_id)
            
        cls._text_index.add_with_ids(v, ids)
        cls.save_indices()

    @classmethod
    def add_image_vector(cls, page_id: int, vector: np.ndarray):
        """
        Validates and adds an image vector mapped to pages.id integer key in FAISS image index.
        """
        if cls._image_index is None:
            raise RuntimeError("Image index is not initialized. Call initialize() first.")
            
        v = cls.validate_vector(vector, cls._image_dim)
        ids = np.array([page_id], dtype=np.int64)
        
        if cls.has_image_vector(page_id):
            logger.warning(f"Page ID {page_id} already exists in FAISS image index. Overwriting vector.")
            cls.remove_image_vector(page_id)
            
        cls._image_index.add_with_ids(v, ids)
        cls.save_indices()

    @classmethod
    def has_text_vector(cls, page_id: int) -> bool:
        """
        Checks if a given page_id integer key is present in FAISS text index.
        """
        if cls._text_index is None:
            return False
        try:
            ids = faiss.vector_to_array(cls._text_index.id_map)
            return page_id in ids
        except Exception:
            # Fallback iteration for compatibility
            try:
                for i in range(cls._text_index.id_map.size()):
                    if cls._text_index.id_map.at(i) == page_id:
                        return True
            except Exception:
                pass
            return False

    @classmethod
    def has_image_vector(cls, page_id: int) -> bool:
        """
        Checks if a given page_id integer key is present in FAISS image index.
        """
        if cls._image_index is None:
            return False
        try:
            ids = faiss.vector_to_array(cls._image_index.id_map)
            return page_id in ids
        except Exception:
            try:
                for i in range(cls._image_index.id_map.size()):
                    if cls._image_index.id_map.at(i) == page_id:
                        return True
            except Exception:
                pass
            return False

    @classmethod
    def remove_text_vector(cls, page_id: int):
        """
        Removes a page_id mapping and its vector from FAISS text index.
        """
        if cls._text_index is not None:
            ids_to_remove = np.array([page_id], dtype=np.int64)
            cls._text_index.remove_ids(ids_to_remove)

    @classmethod
    def remove_image_vector(cls, page_id: int):
        """
        Removes a page_id mapping and its vector from FAISS image index.
        """
        if cls._image_index is not None:
            ids_to_remove = np.array([page_id], dtype=np.int64)
            cls._image_index.remove_ids(ids_to_remove)

    @classmethod
    def get_status(cls) -> dict:
        """
        Returns index status metadata, vector counts, and dimensions.
        """
        text_count = cls._text_index.ntotal if cls._text_index is not None else 0
        image_count = cls._image_index.ntotal if cls._image_index is not None else 0
        return {
            "text_vectors": text_count,
            "image_vectors": image_count,
            "text_dimension": cls._text_dim,
            "image_dimension": cls._image_dim,
            "index_status": "healthy" if cls._text_index is not None and cls._image_index is not None else "uninitialized"
        }

    @classmethod
    def search_text_index(cls, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """
        Searches the FAISS text index. Returns a list of (page_id, score) tuples.
        """
        if cls._text_index is None:
            logger.warning("Text index is not initialized.")
            return []
        
        ntotal = cls._text_index.ntotal
        if ntotal == 0:
            return []
            
        try:
            v = cls.validate_vector(vector, cls._text_dim)
        except Exception as e:
            logger.error(f"Text vector validation failed during search: {e}")
            raise
            
        search_k = min(k, ntotal)
        if search_k <= 0:
            return []
            
        scores, ids = cls._text_index.search(v, search_k)
        
        results = []
        for score, pid in zip(scores[0], ids[0]):
            if pid != -1:
                results.append((int(pid), float(score)))
        return results

    @classmethod
    def search_image_index(cls, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """
        Searches the FAISS image index. Returns a list of (page_id, score) tuples.
        """
        if cls._image_index is None:
            logger.warning("Image index is not initialized.")
            return []
            
        ntotal = cls._image_index.ntotal
        if ntotal == 0:
            return []
            
        try:
            v = cls.validate_vector(vector, cls._image_dim)
        except Exception as e:
            logger.error(f"Image vector validation failed during search: {e}")
            raise
            
        search_k = min(k, ntotal)
        if search_k <= 0:
            return []
            
        scores, ids = cls._image_index.search(v, search_k)
        
        results = []
        for score, pid in zip(scores[0], ids[0]):
            if pid != -1:
                results.append((int(pid), float(score)))
        return results
