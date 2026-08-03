import logging
import numpy as np
from backend.config import settings

logger = logging.getLogger(__name__)

class TextEmbedder:
    _model = None

    @classmethod
    def get_model(cls):
        """
        Lazy-loads the SentenceTransformer model on the appropriate device.
        """
        if cls._model is None:
            device = settings.EMBEDDING_DEVICE.lower()
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device == "cuda":
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA was explicitly requested but is not available.")
            
            logger.info(f"Loading Text Embedding Model '{settings.TEXT_EMBEDDING_MODEL}' on device: {device}...")
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL, device=device)
            logger.info("Text Embedding Model loaded successfully.")
        return cls._model

    @classmethod
    def get_dimension(cls) -> int:
        """
        Returns the embedding dimension of the text model.
        """
        model = cls.get_model()
        return model.get_sentence_embedding_dimension()

    @classmethod
    def embed_text(cls, text: str) -> np.ndarray | None:
        """
        Embeds a single string. Returns a normalized float32 numpy array or None if text is invalid.
        """
        if not text or not text.strip():
            logger.info("Skipping text embedding for empty or whitespace-only text.")
            return None
            
        model = cls.get_model()
        
        import torch
        with torch.inference_mode():
            # encode returns a numpy array if convert_to_numpy=True
            embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            
        # Validate embedding values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            raise ValueError("Generated text embedding contains NaN or Inf values.")
            
        # Ensure strict L2 normalization
        norm = np.linalg.norm(embedding)
        if not np.isclose(norm, 1.0, atol=1e-4):
            embedding = embedding / (norm + 1e-12)
            
        return embedding.astype(np.float32)

    @classmethod
    def embed_texts(cls, texts: list[str]) -> list[np.ndarray | None]:
        """
        Embeds a batch of texts.
        """
        return [cls.embed_text(t) for t in texts]
