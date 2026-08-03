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
from backend.retrieval import retrieve_evidence

# =====================================================================
# Programmatic Test Fixture Generators
# =====================================================================

def generate_text_image_bytes(text_lines: list[str]) -> bytes:
    """Generates a PNG image containing specific text lines."""
    img = Image.new("RGB", (500, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for idx, line in enumerate(text_lines):
        draw.text((20, 30 + idx * 40), line, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_shape_image_bytes(color: str, shape: str) -> bytes:
    """Generates a PNG image with a distinct visual shape."""
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse((40, 40, 184, 184), fill=color, outline=color)
    elif shape == "square":
        draw.rectangle((40, 40, 184, 184), fill=color, outline=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_multipage_pdf_bytes(pages_content: list[list[str]]) -> bytes:
    """Generates a multi-page PDF with text lines on each page."""
    doc = fitz.open()
    for idx, lines in enumerate(pages_content):
        page = doc.new_page(width=500, height=300)
        for line_idx, line in enumerate(lines):
            page.insert_text((20, 40 + line_idx * 40), line, fontsize=14)
    return doc.write()

# =====================================================================
# Unit & Sanity Tests
# =====================================================================

def test_query_validation(client):
    """Verify that empty, whitespace-only, or overly long queries are rejected with HTTP 400."""
    # Create a dummy doc to pass general checks
    png_bytes = generate_text_image_bytes(["Dummy"])
    response = client.post("/documents/upload", files={"file": ("dummy.png", png_bytes, "image/png")})
    doc_id = response.json()["doc_id"]
    
    # None/Empty question
    response = client.post("/retrieve", json={"doc_id": doc_id, "question": ""})
    assert response.status_code == 400
    
    # Whitespace question
    response = client.post("/retrieve", json={"doc_id": doc_id, "question": "   "})
    assert response.status_code == 400
    
    # Overly long question
    long_question = "a" * 501
    response = client.post("/retrieve", json={"doc_id": doc_id, "question": long_question})
    assert response.status_code == 400

def test_document_validation(client):
    """Verify that retrieval checks if doc_id exists and is in completed state."""
    # Non-existent doc_id
    response = client.post("/retrieve", json={"doc_id": "non-existent-uuid", "question": "test query"})
    assert response.status_code == 404
    
    # Registering processing document manually
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, filename, file_type, status) VALUES (?, ?, ?, ?)",
            ("processing-uuid", "test.png", "png", "processing")
        )
    response = client.post("/retrieve", json={"doc_id": "processing-uuid", "question": "test query"})
    assert response.status_code == 400
    assert "processing" in response.json()["detail"]

def test_text_retrieval(client):
    """Verify text-only semantic search ranks the page containing matching text highest."""
    # Upload multi-page document
    pdf_bytes = generate_multipage_pdf_bytes([
        ["Connection Name: TEST-CONNECTION-01"],
        ["Total Amount: 1500"],
        ["Employee Name: Alice"]
    ])
    response = client.post("/documents/upload", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
    doc_id = response.json()["doc_id"]
    
    # Query for Page 2 content
    response = client.post("/retrieve", json={"doc_id": doc_id, "question": "What is the total amount?"})
    assert response.status_code == 200
    results = response.json()["results"]
    
    assert len(results) > 0
    # Page 2 (index 1) should be ranked first
    assert results[0]["page_number"] == 2
    assert "1500" in results[0]["evidence_text"][0]

def test_visual_retrieval(client):
    """Verify visual search using CLIP ranks pages with the matched shape higher."""
    # Page 1 has red circle, Page 2 has blue square
    pdf_doc = fitz.open()
    
    # Generate page 1 (visual circle)
    p1_img_bytes = generate_shape_image_bytes("red", "circle")
    p1 = pdf_doc.new_page(width=224, height=224)
    p1.insert_image(p1.rect, stream=p1_img_bytes)
    
    # Generate page 2 (visual square)
    p2_img_bytes = generate_shape_image_bytes("blue", "square")
    p2 = pdf_doc.new_page(width=224, height=224)
    p2.insert_image(p2.rect, stream=p2_img_bytes)
    
    pdf_bytes = pdf_doc.write()
    response = client.post("/documents/upload", files={"file": ("visual.pdf", pdf_bytes, "application/pdf")})
    doc_id = response.json()["doc_id"]
    
    # Query for red circle visual with weights set to prioritize the visual channel
    orig_text_w = settings.TEXT_RETRIEVAL_WEIGHT
    orig_image_w = settings.IMAGE_RETRIEVAL_WEIGHT
    try:
        settings.TEXT_RETRIEVAL_WEIGHT = 0.0
        settings.IMAGE_RETRIEVAL_WEIGHT = 1.0
        
        response = client.post("/retrieve", json={"doc_id": doc_id, "question": "Which page contains a red circle?"})
        assert response.status_code == 200
        results = response.json()["results"]
        
        assert len(results) > 0
        assert results[0]["page_number"] == 1
    finally:
        settings.TEXT_RETRIEVAL_WEIGHT = orig_text_w
        settings.IMAGE_RETRIEVAL_WEIGHT = orig_image_w

def test_document_isolation(client):
    """Verify that retrieval strictly isolates results to the queried doc_id."""
    # Document A
    pdf_a = generate_multipage_pdf_bytes([["Total Amount: 1500"]])
    response_a = client.post("/documents/upload", files={"file": ("doc_a.pdf", pdf_a, "application/pdf")})
    doc_a_id = response_a.json()["doc_id"]
    
    # Document B
    pdf_b = generate_multipage_pdf_bytes([["Total Amount: 9000"]])
    response_b = client.post("/documents/upload", files={"file": ("doc_b.pdf", pdf_b, "application/pdf")})
    doc_b_id = response_b.json()["doc_id"]
    
    # Search document A
    response = client.post("/retrieve", json={"doc_id": doc_a_id, "question": "What is the total amount?"})
    assert response.status_code == 200
    results = response.json()["results"]
    
    # Ensure no pages from document B leaked in
    for r in results:
        assert r["scores"]["fused"] >= 0.0
        # The endpoint only returns relative page metadata belonging to the requested document. 
        # Check that page mappings resolve to document A.
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT doc_id FROM pages WHERE id = ?", (r["page_id"],))
            db_row = cursor.fetchone()
        assert db_row["doc_id"] == doc_a_id

def test_visual_only_page_indexing_and_retrieval(client):
    """Verify that visual-only page without text can be retrieved using image encoder."""
    # Upload blank image with visual circle drawing (which might hallucinate tiny text blocks in OCR)
    circle_bytes = generate_shape_image_bytes("red", "circle")
    response = client.post("/documents/upload", files={"file": ("circle.png", circle_bytes, "image/png")})
    doc_id = response.json()["doc_id"]
    
    # Resolve page database primary key ID
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM pages WHERE doc_id = ?", (doc_id,))
        page_id = cursor.fetchone()["id"]
        
    # Simulate a true visual-only page by stripping OCR text / text embedding markers
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE pages SET ocr_text = '', text_embedding_indexed = 0, text_embedding_model = NULL WHERE id = ?",
            (page_id,)
        )
    VectorStore.remove_text_vector(page_id)
    VectorStore.save_indices()
    
    # Query metadata to verify text embedding is skipped/removed but image embedding remains indexed
    meta_resp = client.get(f"/documents/{doc_id}/embeddings")
    assert meta_resp.status_code == 200
    meta_data = meta_resp.json()
    assert meta_data["pages"][0]["text_embedding"]["indexed"] is False
    assert meta_data["pages"][0]["image_embedding"]["indexed"] is True
    
    # Retrieve visual element
    response = client.post("/retrieve", json={"doc_id": doc_id, "question": "red circle element"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["page_number"] == 1
    assert "image" in results[0]["matched_modalities"]
    assert "text" not in results[0]["matched_modalities"]

def test_fused_multimodal_ranking(client):
    """Verify that tie-breaking and weights influence ranking deterministically."""
    # Page 1: strong text match ("Employee: Charlie"), weak visual.
    # Page 2: strong visual match (blue square), weak text.
    pdf_doc = fitz.open()
    
    p1 = pdf_doc.new_page(width=500, height=300)
    p1.insert_text((20, 40), "Employee: Charlie", fontsize=14)
    # Weak visual (blank area)
    
    p2 = pdf_doc.new_page(width=224, height=224)
    p2_img_bytes = generate_shape_image_bytes("blue", "square")
    p2.insert_image(p2.rect, stream=p2_img_bytes)
    # Weak text (none)
    
    pdf_bytes = pdf_doc.write()
    response = client.post("/documents/upload", files={"file": ("fused.pdf", pdf_bytes, "application/pdf")})
    doc_id = response.json()["doc_id"]
    
    # Scenario A: Higher text weight (0.8 text, 0.2 image)
    orig_text_w = settings.TEXT_RETRIEVAL_WEIGHT
    orig_image_w = settings.IMAGE_RETRIEVAL_WEIGHT
    
    try:
        settings.TEXT_RETRIEVAL_WEIGHT = 0.9
        settings.IMAGE_RETRIEVAL_WEIGHT = 0.1
        
        # Query that matches text "Charlie" and image "blue square"
        # We expect Page 1 (text match) to rank first because text weight is high.
        response = client.post("/retrieve", json={"doc_id": doc_id, "question": "Charlie's blue square"})
        results = response.json()["results"]
        assert results[0]["page_number"] == 1
        
        # Scenario B: Higher image weight (0.1 text, 0.9 image)
        settings.TEXT_RETRIEVAL_WEIGHT = 0.1
        settings.IMAGE_RETRIEVAL_WEIGHT = 0.9
        
        response = client.post("/retrieve", json={"doc_id": doc_id, "question": "Charlie's blue square"})
        results = response.json()["results"]
        # We expect Page 2 (image match) to rank first because image weight is high.
        assert results[0]["page_number"] == 2
        
    finally:
        # Restore configuration
        settings.TEXT_RETRIEVAL_WEIGHT = orig_text_w
        settings.IMAGE_RETRIEVAL_WEIGHT = orig_image_w

# =====================================================================
# Controlled Evaluation Dataset & Metrics Comparison
# =====================================================================

def test_controlled_retrieval_evaluation(client):
    """
    Evaluates Recall@1, Recall@3, and MRR for Text-only, Image-only, and Fused modes
    on a controlled dataset of 10 query cases.
    """
    # Create visual/text multipage document containing various distinct pages
    pdf_doc = fitz.open()
    
    # 1. Connection invoice page
    p1 = pdf_doc.new_page(width=500, height=300)
    p1.insert_text((20, 40), "Connection Name: ALPHA-CONNECTION", fontsize=14)
    p1.insert_text((20, 80), "Status: Active", fontsize=14)
    
    # 2. Total amount invoice page
    p2 = pdf_doc.new_page(width=500, height=300)
    p2.insert_text((20, 40), "Total Amount: 1500 rupees", fontsize=14)
    p2.insert_text((20, 80), "VAT: 18%", fontsize=14)
    
    # 3. Employee Alice page
    p3 = pdf_doc.new_page(width=500, height=300)
    p3.insert_text((20, 40), "Employee Name: Alice Cooper", fontsize=14)
    p3.insert_text((20, 80), "Department: Engineering", fontsize=14)
    
    # 4. Red circle page
    p4 = pdf_doc.new_page(width=224, height=224)
    p4.insert_image(p4.rect, stream=generate_shape_image_bytes("red", "circle"))
    
    # 5. Blue square page
    p5 = pdf_doc.new_page(width=224, height=224)
    p5.insert_image(p5.rect, stream=generate_shape_image_bytes("blue", "square"))
    
    # 6. Invoice number page
    p6 = pdf_doc.new_page(width=500, height=300)
    p6.insert_text((20, 40), "Invoice Number: INV-001", fontsize=14)
    
    # 7. Employee Charlie page
    p7 = pdf_doc.new_page(width=500, height=300)
    p7.insert_text((20, 40), "Employee Name: Charlie Brown", fontsize=14)
    
    # 8. Service fee page
    p8 = pdf_doc.new_page(width=500, height=300)
    p8.insert_text((20, 40), "Service Fee: 300 rupees", fontsize=14)
    
    # 9. Blue circle page
    p9 = pdf_doc.new_page(width=224, height=224)
    p9.insert_image(p9.rect, stream=generate_shape_image_bytes("blue", "circle"))
    
    # 10. Red square page
    p10 = pdf_doc.new_page(width=224, height=224)
    p10.insert_image(p10.rect, stream=generate_shape_image_bytes("red", "square"))
    
    pdf_bytes = pdf_doc.write()
    response = client.post("/documents/upload", files={"file": ("evaluation.pdf", pdf_bytes, "application/pdf")})
    doc_id = response.json()["doc_id"]
    
    # 10 query test cases mapping (question -> expected_page_number)
    test_cases = [
        ("What is the connection name?", 1),
        ("total amount is 1500", 2),
        ("employee named Alice Cooper", 3),
        ("page with the red circle drawing", 4),
        ("Which page shows a blue square?", 5),
        ("invoice number INV-001", 6),
        ("Employee Charlie Brown", 7),
        ("service fee is 300 rupees", 8),
        ("visual blue circle drawing", 9),
        ("visual red square drawing", 10),
    ]
    
    orig_text_w = settings.TEXT_RETRIEVAL_WEIGHT
    orig_image_w = settings.IMAGE_RETRIEVAL_WEIGHT
    
    evaluation_results = {}
    
    # Modes to compare: (name, text_weight, image_weight)
    modes = [
        ("Text-only", 1.0, 0.0),
        ("Image-only", 0.0, 1.0),
        ("Fused", 0.65, 0.35)
    ]
    
    for mode_name, text_w, image_w in modes:
        settings.TEXT_RETRIEVAL_WEIGHT = text_w
        settings.IMAGE_RETRIEVAL_WEIGHT = image_w
        
        recalls_1 = []
        recalls_3 = []
        rr_scores = []
        
        for question, expected_page in test_cases:
            res = retrieve_evidence(doc_id, question, top_k=3)
            page_ranks = [item["page_number"] for item in res]
            
            # Recall@1
            r1 = 1 if page_ranks and page_ranks[0] == expected_page else 0
            recalls_1.append(r1)
            
            # Recall@3
            r3 = 1 if expected_page in page_ranks[:3] else 0
            recalls_3.append(r3)
            
            # Reciprocal Rank (MRR)
            if expected_page in page_ranks:
                rank = page_ranks.index(expected_page) + 1
                rr_scores.append(1.0 / rank)
            else:
                rr_scores.append(0.0)
                
        evaluation_results[mode_name] = {
            "Recall@1": np.mean(recalls_1),
            "Recall@3": np.mean(recalls_3),
            "MRR": np.mean(rr_scores)
        }
        
    # Restore weights
    settings.TEXT_RETRIEVAL_WEIGHT = orig_text_w
    settings.IMAGE_RETRIEVAL_WEIGHT = orig_image_w
    
    # Print evaluation comparison table to stdout (inspected via -s pytest flag)
    print("\n" + "=" * 50)
    print("      RETRIEVAL MODE METRICS EVALUATION")
    print("=" * 50)
    print(f"{'Mode':<15} | {'Recall@1':<10} | {'Recall@3':<10} | {'MRR':<10}")
    print("-" * 50)
    for mode, metrics in evaluation_results.items():
        print(f"{mode:<15} | {metrics['Recall@1']:<10.2f} | {metrics['Recall@3']:<10.2f} | {metrics['MRR']:<10.2f}")
    print("=" * 50 + "\n")
    
    # Assert basic target levels for Fused retrieval model
    # (engineering acceptance checks on this synthetic controlled set)
    fused = evaluation_results["Fused"]
    assert fused["Recall@1"] >= 0.70
    assert fused["Recall@3"] >= 0.90
