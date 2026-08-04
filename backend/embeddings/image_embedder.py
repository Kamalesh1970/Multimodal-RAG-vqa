import logging
import numpy as np
from PIL import Image
from backend.config import settings

logger = logging.getLogger(__name__)

class ImageEmbedder:
    _model = None
    _preprocess = None
    _tokenizer = None
    _device = None

    @classmethod
    def get_model_and_transforms(cls):
        """
        Lazy-loads the OpenCLIP model and transforms.
        """
        if cls._model is None:
            import torch
            import open_clip
            
            # Determine device
            device = settings.EMBEDDING_DEVICE.lower()
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA was explicitly requested but is not available.")
            
            cls._device = device
            
            model_name = settings.IMAGE_EMBEDDING_MODEL
            pretrained = settings.IMAGE_EMBEDDING_PRETRAINED
            
            logger.info(f"Loading CLIP model '{model_name}' with weights '{pretrained}' on device '{device}'...")
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained,
                    device=device
                )
                tokenizer = open_clip.get_tokenizer(model_name)
            except Exception as e:
                logger.critical(f"Failed to load OpenCLIP model: {e}", exc_info=True)
                raise RuntimeError(f"OpenCLIP initialization error: {e}") from e
                
            cls._model = model
            cls._preprocess = preprocess
            cls._tokenizer = tokenizer
            logger.info("[MODEL_LOAD] CLIP initialized once")
            
        return cls._model, cls._preprocess, cls._tokenizer, cls._device

    @classmethod
    def get_dimension(cls) -> int:
        """
        Returns the output dimension of the CLIP visual encoder.
        """
        if settings.IMAGE_EMBEDDING_MODEL == "ViT-B-32":
            return 512
        model, _, _, _ = cls.get_model_and_transforms()
        if hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
            return model.visual.output_dim
        # Default fallback for ViT-B-32
        return 512

    @classmethod
    def embed_image(cls, image: Image.Image) -> np.ndarray:
        """
        Embeds a PIL Image. Returns L2 normalized float32 numpy array.
        """
        import torch
        model, preprocess, _, device = cls.get_model_and_transforms()
        
        try:
            img_tensor = preprocess(image).unsqueeze(0).to(device)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Invalid image format: {e}") from e
            
        with torch.inference_mode():
            features = model.encode_image(img_tensor)
            # Normalize embedding
            features /= features.norm(dim=-1, keepdim=True)
            embedding = features.cpu().numpy()[0]
            
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            raise ValueError("Generated image embedding contains NaN or Inf values.")
            
        return embedding.astype(np.float32)

    @classmethod
    def embed_text(cls, text: str) -> np.ndarray:
        """
        Embeds a query text using CLIP text encoder. Returns L2 normalized float32 numpy array.
        """
        import torch
        model, _, tokenizer, device = cls.get_model_and_transforms()
        
        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace text.")
            
        try:
            tokens = tokenizer([text]).to(device)
        except Exception as e:
            logger.error(f"Tokenization failed for text '{text}': {e}")
            raise ValueError(f"Text tokenization error: {e}") from e
            
        with torch.inference_mode():
            features = model.encode_text(tokens)
            # Normalize embedding
            features /= features.norm(dim=-1, keepdim=True)
            embedding = features.cpu().numpy()[0]
            
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            raise ValueError("Generated CLIP text embedding contains NaN or Inf values.")
            
        return embedding.astype(np.float32)

    @classmethod
    def embed_images(cls, images: list[Image.Image]) -> list[np.ndarray]:
        """
        Embeds a list of PIL Images in a single batch.
        """
        if not images:
            return []
            
        import torch
        model, preprocess, _, device = cls.get_model_and_transforms()
        
        try:
            # Preprocess all images and stack them into a single batch tensor
            tensors = [preprocess(img) for img in images]
            img_tensor = torch.stack(tensors).to(device)
        except Exception as e:
            logger.error(f"Batch image preprocessing failed: {e}")
            raise ValueError(f"Invalid image format: {e}") from e
            
        with torch.inference_mode():
            features = model.encode_image(img_tensor)
            # Normalize embedding along features dimension
            features /= features.norm(dim=-1, keepdim=True)
            embeddings = features.cpu().numpy()
            
        results = []
        for embedding in embeddings:
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                raise ValueError("Generated image embedding contains NaN or Inf values.")
            results.append(embedding.astype(np.float32))
            
        return results
