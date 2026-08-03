import io
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import fitz

from backend.main import app
from backend.config import settings
from backend.database import get_db_connection
from backend.ingestion.ocr import PageOCRResult, OCRBlock


def generate_test_image_bytes():
    """Generates a test PNG image containing known text."""
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "Connection Name: TEST-CONNECTION-01", fill=(0, 0, 0))
    draw.text((10, 60), "Total Amount: 1500", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_test_pdf_bytes():
    """Generates a simple single-page PDF containing known text."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((10, 30), "Connection Name: TEST-CONNECTION-01", fontsize=12)
    page.insert_text((10, 70), "Total Amount: 1500", fontsize=12)
    return doc.write()

@pytest.fixture
def mock_perform_ocr():
    """Fixture to mock OCR calls for fast unit testing."""
    dummy_result = PageOCRResult(
        page_number=1,
        width=400,
        height=200,
        blocks=[
            OCRBlock(text="Connection Name: TEST-CONNECTION-01", confidence=0.99, bbox=[[10.0, 20.0], [300.0, 20.0], [300.0, 40.0], [10.0, 40.0]]),
            OCRBlock(text="Total Amount: 1500", confidence=0.98, bbox=[[10.0, 60.0], [200.0, 60.0], [200.0, 80.0], [10.0, 80.0]])
        ],
        full_text="Connection Name: TEST-CONNECTION-01\nTotal Amount: 1500"
    )
    with patch("backend.ingestion.processor.perform_ocr", return_value=dummy_result) as mock:
        yield mock

def test_health_still_works(client):
    """Verify health and root endpoints return status code 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["phase"] == 2

    response = client.get("/")
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text or "html" in response.text

def test_unsupported_extension(client):
    """Verify upload endpoint rejects unsupported extensions with HTTP 400."""
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", b"some dummy text", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file type." in response.json()["detail"]

def test_empty_upload(client):
    """Verify upload endpoint rejects empty file uploads with HTTP 400."""
    response = client.post(
        "/documents/upload",
        files={"file": ("test.png", b"", "image/png")}
    )
    assert response.status_code == 400
    assert "Empty file." in response.json()["detail"]

def test_corrupted_image(client):
    """Verify upload endpoint rejects invalid/corrupted image bytes with HTTP 400."""
    response = client.post(
        "/documents/upload",
        files={"file": ("test.png", b"corrupted image bytes", "image/png")}
    )
    assert response.status_code == 400
    assert "Corrupted image file." in response.json()["detail"]

def test_corrupted_pdf(client):
    """Verify upload endpoint rejects corrupted PDF bytes with HTTP 400."""
    response = client.post(
        "/documents/upload",
        files={"file": ("test.pdf", b"corrupted pdf bytes", "application/pdf")}
    )
    assert response.status_code == 400
    assert "Corrupted PDF file." in response.json()["detail"]

def test_oversized_upload(client):
    """Verify upload endpoint rejects files exceeding settings limit with HTTP 413."""
    # Set limit temporarily to 0 MB (exceeds all files)
    with patch.object(settings, "MAX_UPLOAD_MB", 0):
        png_bytes = generate_test_image_bytes()
        response = client.post(
            "/documents/upload",
            files={"file": ("test.png", png_bytes, "image/png")}
        )
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

def test_nonexistent_doc_id(client):
    """Verify non-existent doc_id retrieval returns HTTP 404."""
    response = client.get("/documents/nonexistent-uuid")
    assert response.status_code == 404
    
    response = client.get("/documents/nonexistent-uuid/ocr")
    assert response.status_code == 404

def test_valid_png_upload_flow(client, mock_perform_ocr):
    """Verify E2E flow for uploading a valid PNG file under mock conditions."""
    png_bytes = generate_test_image_bytes()
    response = client.post(
        "/documents/upload",
        files={"file": ("test.png", png_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert data["filename"] == "test.png"
    assert data["status"] == "completed"
    assert len(data["pages"]) == 1
    
    doc_id = data["doc_id"]
    
    # 1. Fetch metadata and check persistence
    meta_response = client.get(f"/documents/{doc_id}")
    assert meta_response.status_code == 200
    meta_data = meta_response.json()
    assert meta_data["doc_id"] == doc_id
    assert meta_data["status"] == "completed"
    assert meta_data["page_count"] == 1
    
    # 2. Fetch OCR results
    ocr_response = client.get(f"/documents/{doc_id}/ocr")
    assert ocr_response.status_code == 200
    ocr_data = ocr_response.json()
    assert ocr_data["doc_id"] == doc_id
    assert len(ocr_data["pages"]) == 1
    assert "Connection Name: TEST-CONNECTION-01" in ocr_data["pages"][0]["ocr"]["full_text"]

def test_valid_pdf_upload_flow(client, mock_perform_ocr):
    """Verify E2E flow for uploading a valid PDF file under mock conditions."""
    pdf_bytes = generate_test_pdf_bytes()
    response = client.post(
        "/documents/upload",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert data["filename"] == "test.pdf"
    assert data["status"] == "completed"
    assert len(data["pages"]) == 1
    
    doc_id = data["doc_id"]
    
    # Check database
    meta_response = client.get(f"/documents/{doc_id}")
    assert meta_response.status_code == 200
    assert meta_response.json()["page_count"] == 1

@pytest.mark.ocr
def test_real_ocr_acceptance(client):
    """Mandatory acceptance test using real PaddleOCR engine execution."""
    png_bytes = generate_test_image_bytes()
    response = client.post(
        "/documents/upload",
        files={"file": ("test.png", png_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    
    doc_id = data["doc_id"]
    ocr_response = client.get(f"/documents/{doc_id}/ocr")
    assert ocr_response.status_code == 200
    ocr_data = ocr_response.json()
    
    full_text = ocr_data["pages"][0]["ocr"]["full_text"]
    # Check that meaningful text was extracted by PaddleOCR
    assert "TEST-CONNECTION-01" in full_text or "TEST" in full_text
    assert "1500" in full_text

    # Verify structured block coordinates, text, and confidence levels
    blocks = ocr_data["pages"][0]["ocr"]["blocks"]
    assert len(blocks) > 0
    for block in blocks:
        assert isinstance(block["text"], str)
        assert len(block["text"]) > 0
        assert isinstance(block["confidence"], float)
        assert 0.0 <= block["confidence"] <= 1.0
        assert isinstance(block["bbox"], list)
        assert len(block["bbox"]) == 4
        for pt in block["bbox"]:
            assert len(pt) == 2
            assert isinstance(pt[0], float)
            assert isinstance(pt[1], float)
