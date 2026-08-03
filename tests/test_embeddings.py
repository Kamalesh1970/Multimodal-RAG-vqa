import io
import pytest
import numpy as np
from PIL import Image, ImageDraw
import fitz
from backend.embeddings.text_embedder import TextEmbedder
from backend.embeddings.image_embedder import ImageEmbedder
from backend.vector_store import VectorStore
from backend.database import get_db_connection
from backend.config import settings

def generate_test_image():
    """Generates a PIL image containing known text."""
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "Connection Name: TEST-CONNECTION-01", fill=(0, 0, 0))
    draw.text((10, 60), "Total Amount: 1500", fill=(0, 0, 0))
    return img

def generate_test_image_bytes():
    """Generates bytes of a PNG image containing known text."""
    img = generate_test_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_red_circle_image():
    """Generates a PIL Image showing a red circle on a white background."""
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((50, 50, 174, 174), fill=(255, 0, 0), outline=(255, 0, 0))
    return img

def generate_multipage_pdf_bytes():
    """Generates a 2-page PDF containing text on each page."""
    doc = fitz.open()
    page1 = doc.new_page(width=400, height=200)
    page1.insert_text((10, 30), "Connection Name: TEST-CONNECTION-01", fontsize=12)
    page2 = doc.new_page(width=400, height=200)
    page2.insert_text((10, 70), "Total Amount: 1500", fontsize=12)
    return doc.write()

def generate_blank_image_bytes():
    """Generates bytes for a completely blank white image."""
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def calculate_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity of two normalized vectors via dot product."""
    return float(np.dot(v1, v2))

# =====================================================================
# Sanity & Unit Tests (Models & Vector Store)
# =====================================================================

def test_text_embedding_generation():
    """Verify SentenceTransformer outputs formatted, L2-normalized float32 vectors."""
    text = "Connection Name: TEST-CONNECTION-01"
    vec = TextEmbedder.embed_text(text)
    
    assert vec is not None
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert len(vec) == TextEmbedder.get_dimension()
    assert np.isfinite(vec).all()
    
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-4)

def test_text_semantic_sanity():
    """Verify semantic similarity relationship is logical."""
    vec_a = TextEmbedder.embed_text("The invoice total is 1500 rupees.")
    vec_b = TextEmbedder.embed_text("Total amount due: ₹1500.")
    vec_c = TextEmbedder.embed_text("A dog is running through a park.")
    
    sim_ab = calculate_cosine_similarity(vec_a, vec_b)
    sim_ac = calculate_cosine_similarity(vec_a, vec_c)
    
    assert sim_ab > sim_ac
    logger_msg = f"Semantic sanity test: sim(A, B) = {sim_ab:.4f} > sim(A, C) = {sim_ac:.4f}"
    print(logger_msg)

def test_image_embedding_generation():
    """Verify OpenCLIP visual encoder outputs formatted, L2-normalized vectors."""
    img = generate_test_image()
    vec = ImageEmbedder.embed_image(img)
    
    assert vec is not None
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert len(vec) == ImageEmbedder.get_dimension()
    assert np.isfinite(vec).all()
    
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-4)

def test_clip_cross_modal_sanity():
    """Verify text queries and images share the same embedding space."""
    img = generate_red_circle_image()
    img_vec = ImageEmbedder.embed_image(img)
    
    text_relevant_vec = ImageEmbedder.embed_text("a red circle")
    text_unrelated_vec = ImageEmbedder.embed_text("a photograph of a forest")
    
    sim_relevant = calculate_cosine_similarity(img_vec, text_relevant_vec)
    sim_unrelated = calculate_cosine_similarity(img_vec, text_unrelated_vec)
    
    assert sim_relevant > sim_unrelated
    print(f"Cross-modal sanity test: sim(red circle image, 'a red circle') = {sim_relevant:.4f} > sim(red circle image, 'forest') = {sim_unrelated:.4f}")

# =====================================================================
# Ingestion, Integration, and DB Mapping Tests
# =====================================================================

def test_real_ingestion_and_indexing_pipeline(client):
    """Verify end-to-end upload generates database entries and registers FAISS vectors correctly."""
    png_bytes = generate_test_image_bytes()
    response = client.post(
        "/documents/upload",
        files={"file": ("test.png", png_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    
    doc_id = data["doc_id"]
    
    # 1. Verify database mappings and indexing markers
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, text_embedding_indexed, image_embedding_indexed FROM pages WHERE doc_id = ?", (doc_id,))
        page_row = cursor.fetchone()
        
    assert page_row is not None
    page_id = page_row["id"]
    assert page_row["text_embedding_indexed"] == 1
    assert page_row["image_embedding_indexed"] == 1
    
    # 2. Verify FAISS vector presence
    assert VectorStore.has_text_vector(page_id) is True
    assert VectorStore.has_image_vector(page_id) is True
    
    # 3. Verify metadata API response structure
    meta_response = client.get(f"/documents/{doc_id}/embeddings")
    assert meta_response.status_code == 200
    meta_data = meta_response.json()
    assert meta_data["doc_id"] == doc_id
    assert len(meta_data["pages"]) == 1
    
    page_meta = meta_data["pages"][0]
    assert page_meta["page_id"] == page_id
    assert page_meta["text_embedding"]["indexed"] is True
    assert page_meta["text_embedding"]["dimension"] == TextEmbedder.get_dimension()
    assert page_meta["image_embedding"]["indexed"] is True
    assert page_meta["image_embedding"]["dimension"] == ImageEmbedder.get_dimension()

def test_multipage_pdf_indexing(client):
    """Verify multi-page PDF generates unique page-level vectors and indexes each correctly."""
    pdf_bytes = generate_multipage_pdf_bytes()
    response = client.post(
        "/documents/upload",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    
    doc_id = data["doc_id"]
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, page_number FROM pages WHERE doc_id = ? ORDER BY page_number ASC", (doc_id,))
        rows = cursor.fetchall()
        
    assert len(rows) == 2
    page1_id = rows[0]["id"]
    page2_id = rows[1]["id"]
    
    # Verify both pages are indexed separately in FAISS
    assert VectorStore.has_text_vector(page1_id) is True
    assert VectorStore.has_text_vector(page2_id) is True
    assert VectorStore.has_image_vector(page1_id) is True
    assert VectorStore.has_image_vector(page2_id) is True
    
    # Check counts endpoint
    status_response = client.get("/system/vector-status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["text_vectors"] >= 2
    assert status_data["image_vectors"] >= 2

def test_blank_page_indexing_behavior(client):
    """Verify blank page generates an image vector but skips generating text vector."""
    blank_bytes = generate_blank_image_bytes()
    response = client.post(
        "/documents/upload",
        files={"file": ("blank.png", blank_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    
    doc_id = data["doc_id"]
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, text_embedding_indexed, image_embedding_indexed FROM pages WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        
    assert row is not None
    page_id = row["id"]
    
    # Verify image is indexed but text is skipped
    assert row["image_embedding_indexed"] == 1
    assert row["text_embedding_indexed"] == 0
    assert VectorStore.has_image_vector(page_id) is True
    assert VectorStore.has_text_vector(page_id) is False

def test_index_persistence_and_reload(client):
    """Verify FAISS indexes can reload from disk and counts remain consistent."""
    # Write a test vector
    text_dim = TextEmbedder.get_dimension()
    image_dim = ImageEmbedder.get_dimension()
    
    dummy_text_vec = np.random.rand(text_dim).astype(np.float32)
    dummy_text_vec /= np.linalg.norm(dummy_text_vec)
    dummy_image_vec = np.random.rand(image_dim).astype(np.float32)
    dummy_image_vec /= np.linalg.norm(dummy_image_vec)
    
    # Initialize and add
    VectorStore.initialize(text_dim, image_dim)
    initial_text_count = VectorStore.get_status()["text_vectors"]
    initial_image_count = VectorStore.get_status()["image_vectors"]
    
    test_id = 99999
    VectorStore.add_text_vector(test_id, dummy_text_vec)
    VectorStore.add_image_vector(test_id, dummy_image_vec)
    
    assert VectorStore.get_status()["text_vectors"] == initial_text_count + 1
    assert VectorStore.get_status()["image_vectors"] == initial_image_count + 1
    
    # Save, clear cache and force reload
    VectorStore.save_indices()
    
    # Re-initialize to reload from disk files
    VectorStore._text_index = None
    VectorStore._image_index = None
    VectorStore.initialize(text_dim, image_dim)
    
    assert VectorStore.get_status()["text_vectors"] == initial_text_count + 1
    assert VectorStore.get_status()["image_vectors"] == initial_image_count + 1
    assert VectorStore.has_text_vector(test_id) is True
    assert VectorStore.has_image_vector(test_id) is True
    
    # Cleanup
    VectorStore.remove_text_vector(test_id)
    VectorStore.remove_image_vector(test_id)
    VectorStore.save_indices()
