import logging
from PIL import Image

logger = logging.getLogger(__name__)

def prepare_vlm_image(img: Image.Image, max_dim: int) -> Image.Image:
    """
    Resizes layout screenshots to keep max dimension under max_dim
    while preserving aspect ratio and ensuring readability for VLM.
    Strips EXIF/metadata (implicitly handled by not saving metadata).
    """
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h} (max_dim={max_dim})")
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img

def crop_evidence_region(img: Image.Image, bboxes: list, padding: int = 40) -> Image.Image:
    """
    Crops the page image around the given bounding boxes of matching OCR evidence.
    Each box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    Applies safety padding and returns the cropped image.
    If bboxes is empty, returns original image.
    """
    if not bboxes:
        return img
        
    w, h = img.size
    
    xs = []
    ys = []
    for bbox in bboxes:
        if isinstance(bbox, list) and len(bbox) == 4:
            for pt in bbox:
                if isinstance(pt, list) and len(pt) >= 2:
                    xs.append(pt[0])
                    ys.append(pt[1])
                    
    if not xs or not ys:
        return img
        
    min_x = max(0, int(min(xs)) - padding)
    min_y = max(0, int(min(ys)) - padding)
    max_x = min(w, int(max(xs)) + padding)
    max_y = min(h, int(max(ys)) + padding)
    
    # Verify valid crop dimensions
    if (max_x - min_x) > 10 and (max_y - min_y) > 10:
        logger.info(f"Cropping image to evidence region: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        return img.crop((min_x, min_y, max_x, max_y))
        
    return img
