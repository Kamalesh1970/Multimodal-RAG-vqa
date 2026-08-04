import pytest
import json
import io
import fitz
from unittest.mock import patch
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from backend.main import app
from backend.auth.dependencies import get_current_user, security_scheme
from backend.storage import repository
from backend.config import settings

@pytest.fixture(autouse=True)
def force_sqlite_backend():
    """
    Force FIREBASE_ENABLED = False during auth isolation tests to guarantee 
    absolute database isolation in the local SQLite test DB.
    """
    orig = settings.FIREBASE_ENABLED
    settings.FIREBASE_ENABLED = False
    yield
    settings.FIREBASE_ENABLED = orig

# Global text variable to communicate the expected OCR text to the mock OCR runner
current_ocr_text = "Default Mocked OCR Text"

def generate_pdf_with_text(text: str) -> bytes:
    """Generates a valid single-page PDF containing the given text."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((10, 30), text, fontsize=12)
    return doc.write()

@pytest.fixture(autouse=True)
def mock_ocr_runs():
    """
    Fixture to intercept PaddleOCR calls, returning custom text mock results instantly.
    Prevents heavy PaddleOCR models from initializing during isolation tests.
    """
    from backend.ingestion.ocr import PageOCRResult, OCRBlock
    with patch("backend.ingestion.processor.perform_ocr") as mock_ocr:
        def side_effect(image_array, *args, **kwargs):
            page_num = kwargs.get("page_number", 1)
            return PageOCRResult(
                page_number=page_num,
                width=image_array.shape[1],
                height=image_array.shape[0],
                blocks=[
                    OCRBlock(text=current_ocr_text, confidence=0.99, bbox=[[10.0, 20.0], [300.0, 20.0], [300.0, 40.0], [10.0, 40.0]])
                ],
                full_text=current_ocr_text
            )
        mock_ocr.side_effect = side_effect
        yield mock_ocr

# Mock authentication dependency that inspects the Authorization Bearer header
def mock_get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme)):
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing or invalid scheme.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    if token == "token_user_a":
        return {"uid": "user_a", "email": "user_a@example.com", "display_name": "Alice"}
    elif token == "token_user_b":
        return {"uid": "user_b", "email": "user_b@example.com", "display_name": "Bob"}
    elif token == "token_invalid" or token == "token_expired":
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired credentials: Mocked error.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    else:
         raise HTTPException(
            status_code=401,
            detail="Invalid token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

@pytest.fixture(autouse=True)
def setup_auth_override():
    app.dependency_overrides[get_current_user] = mock_get_current_user
    yield
    app.dependency_overrides.pop(get_current_user, None)

def test_missing_authorization_header_unauthorized(client):
    """1. Test that requests without Authorization header are rejected with 401."""
    response = client.get("/documents")
    assert response.status_code == 401
    assert "Authorization header missing" in response.json()["detail"]

def test_invalid_bearer_token_unauthorized(client):
    """2. Test that requests with invalid Bearer token are rejected with 401."""
    headers = {"Authorization": "Bearer token_invalid"}
    response = client.get("/documents", headers=headers)
    assert response.status_code == 401
    assert "Invalid or expired credentials" in response.json()["detail"]

def test_valid_token_allowed(client):
    """3. Test that requests with a valid token are allowed."""
    headers = {"Authorization": "Bearer token_user_a"}
    response = client.get("/documents", headers=headers)
    assert response.status_code == 200
    assert response.json() == []

def test_user_document_upload_and_list_isolation(client):
    """4. Test that uploaded documents are owned by the uploader, and listings are isolated."""
    global current_ocr_text
    
    # User A uploads a document
    headers_a = {"Authorization": "Bearer token_user_a"}
    current_ocr_text = "This is Alice's private invoice data. Code: PRIVATE_A_728491"
    pdf_bytes_a = generate_pdf_with_text(current_ocr_text)
    
    response_upload_a = client.post(
        "/documents/upload",
        files={"file": ("invoice_alice.pdf", pdf_bytes_a, "application/pdf")},
        headers=headers_a
    )
    assert response_upload_a.status_code == 200
    doc_a_id = response_upload_a.json()["doc_id"]
    assert doc_a_id is not None
    
    # User B uploads a document
    headers_b = {"Authorization": "Bearer token_user_b"}
    current_ocr_text = "This is Bob's private report data. Code: PRIVATE_B_913742"
    pdf_bytes_b = generate_pdf_with_text(current_ocr_text)
    
    response_upload_b = client.post(
        "/documents/upload",
        files={"file": ("report_bob.pdf", pdf_bytes_b, "application/pdf")},
        headers=headers_b
    )
    assert response_upload_b.status_code == 200
    doc_b_id = response_upload_b.json()["doc_id"]
    assert doc_b_id is not None
    
    # Verify User A's document list contains only User A's document
    response_list_a = client.get("/documents", headers=headers_a)
    assert response_list_a.status_code == 200
    docs_a = response_list_a.json()
    assert len(docs_a) == 1
    assert docs_a[0]["doc_id"] == doc_a_id
    assert docs_a[0]["owner_id"] == "user_a"
    
    # Verify User B's document list contains only User B's document
    response_list_b = client.get("/documents", headers=headers_b)
    assert response_list_b.status_code == 200
    docs_b = response_list_b.json()
    assert len(docs_b) == 1
    assert docs_b[0]["doc_id"] == doc_b_id
    assert docs_b[0]["owner_id"] == "user_b"

def test_cross_user_document_access_denied(client):
    """5. Test that User A cannot read User B's document metadata, OCR, or embeddings."""
    global current_ocr_text
    
    # Setup: User B uploads a document
    headers_b = {"Authorization": "Bearer token_user_b"}
    current_ocr_text = "Private Bob document contents."
    pdf_bytes_b = generate_pdf_with_text(current_ocr_text)
    
    res = client.post(
        "/documents/upload",
        files={"file": ("bob.pdf", pdf_bytes_b, "application/pdf")},
        headers=headers_b
    )
    assert res.status_code == 200
    doc_b_id = res.json()["doc_id"]
    
    headers_a = {"Authorization": "Bearer token_user_a"}
    
    # User A requests Bob's document metadata -> Denied
    res_meta = client.get(f"/documents/{doc_b_id}", headers=headers_a)
    assert res_meta.status_code == 403
    
    # User A requests Bob's document OCR -> Denied
    res_ocr = client.get(f"/documents/{doc_b_id}/ocr", headers=headers_a)
    assert res_ocr.status_code == 403
    
    # User A requests Bob's document embeddings -> Denied
    res_emb = client.get(f"/documents/{doc_b_id}/embeddings", headers=headers_a)
    assert res_emb.status_code == 403

def test_cross_user_ask_retrieval_denied_before_execution(client):
    """6. Test that User A cannot ask questions about User B's document, and that access check runs before retrieval/VLM."""
    global current_ocr_text
    
    # Setup: User B uploads a document
    headers_b = {"Authorization": "Bearer token_user_b"}
    current_ocr_text = "Bob private code: PRIVATE_B_913742"
    pdf_bytes_b = generate_pdf_with_text(current_ocr_text)
    
    res = client.post(
        "/documents/upload",
        files={"file": ("bob.pdf", pdf_bytes_b, "application/pdf")},
        headers=headers_b
    )
    assert res.status_code == 200
    doc_b_id = res.json()["doc_id"]
    
    # User A tries to ask Bob's document
    headers_a = {"Authorization": "Bearer token_user_a"}
    ask_req = {
        "doc_id": doc_b_id,
        "question": "What is the private code?"
    }
    res_ask = client.post("/ask", json=ask_req, headers=headers_a)
    assert res_ask.status_code == 403
    assert "Access denied" in res_ask.json()["detail"]

def test_cross_user_delete_denied(client):
    """7. Test that User A cannot delete User B's document."""
    global current_ocr_text
    
    # Setup: User B uploads a document
    headers_b = {"Authorization": "Bearer token_user_b"}
    current_ocr_text = "bob data"
    pdf_bytes_b = generate_pdf_with_text(current_ocr_text)
    
    res = client.post(
        "/documents/upload",
        files={"file": ("bob.pdf", pdf_bytes_b, "application/pdf")},
        headers=headers_b
    )
    assert res.status_code == 200
    doc_b_id = res.json()["doc_id"]
    
    # User A deletes -> Denied
    headers_a = {"Authorization": "Bearer token_user_a"}
    res_del = client.delete(f"/documents/{doc_b_id}", headers=headers_a)
    assert res_del.status_code == 403

def test_cross_user_chat_history_isolation(client):
    """8. Test that User A cannot fetch chat history from User B's session."""
    global current_ocr_text
    
    # Setup: User B uploads doc and gets session ID from asking
    headers_b = {"Authorization": "Bearer token_user_b"}
    current_ocr_text = "Some Bob text."
    pdf_bytes_b = generate_pdf_with_text(current_ocr_text)
    
    res_upload = client.post(
        "/documents/upload",
        files={"file": ("bob.pdf", pdf_bytes_b, "application/pdf")},
        headers=headers_b
    )
    assert res_upload.status_code == 200
    doc_b_id = res_upload.json()["doc_id"]
    
    res_ask = client.post(
        "/ask",
        json={"doc_id": doc_b_id, "question": "What is this?"},
        headers=headers_b
    )
    assert res_ask.status_code == 200
    session_id = res_ask.json()["session_id"]
    
    # User B can view their history
    res_hist_b = client.get(f"/chat/history/{session_id}", headers=headers_b)
    assert res_hist_b.status_code == 200
    
    # User A cannot view User B's history
    headers_a = {"Authorization": "Bearer token_user_a"}
    res_hist_a = client.get(f"/chat/history/{session_id}", headers=headers_a)
    assert res_hist_a.status_code == 403

def test_cross_user_processed_image_serving_isolation(client):
    """9. Test that User A cannot serve/view preprocessed page images belonging to User B's document."""
    global current_ocr_text
    
    # Setup: User B uploads a document
    headers_b = {"Authorization": "Bearer token_user_b"}
    current_ocr_text = "Content for Bob page rendering."
    pdf_bytes_b = generate_pdf_with_text(current_ocr_text)
    
    res_upload = client.post(
        "/documents/upload",
        files={"file": ("bob.pdf", pdf_bytes_b, "application/pdf")},
        headers=headers_b
    )
    assert res_upload.status_code == 200
    doc_b_id = res_upload.json()["doc_id"]
    
    # Serving url
    img_url = f"/processed/{doc_b_id}/page_1.jpg"
    
    # User B gets it
    res_img_b = client.get(img_url, headers=headers_b)
    # Could be 200 if files exists or 404 (file not found) but NEVER 403
    assert res_img_b.status_code in (200, 404)
    
    # User A gets it -> Denied (403)
    headers_a = {"Authorization": "Bearer token_user_a"}
    res_img_a = client.get(img_url, headers=headers_a)
    assert res_img_a.status_code == 403

def test_legacy_unassigned_document_protection(client):
    """10. Test that legacy documents (owner_id = null) are inaccessible to standard ordinary users."""
    doc_id = "legacy-doc-123"
    repository.create_document(
        doc_id=doc_id,
        filename="legacy.pdf",
        stored_path="data/uploads/legacy.pdf",
        file_type="pdf",
        page_count=1,
        status="completed",
        owner_id=None # Legacy / Unassigned
    )
    
    headers_a = {"Authorization": "Bearer token_user_a"}
    
    # User A requests metadata -> Denied
    res = client.get(f"/documents/{doc_id}", headers=headers_a)
    assert res.status_code == 403

def test_authenticated_owner_validation_errors(client):
    """11. Test that authenticated owners still receive correct 400 validation codes for invalid inputs."""
    global current_ocr_text
    
    headers_a = {"Authorization": "Bearer token_user_a"}
    current_ocr_text = "Alice's invoice data."
    pdf_bytes_a = generate_pdf_with_text(current_ocr_text)
    
    # Upload doc
    res = client.post(
        "/documents/upload",
        files={"file": ("alice.pdf", pdf_bytes_a, "application/pdf")},
        headers=headers_a
    )
    assert res.status_code == 200
    doc_id = res.json()["doc_id"]
    
    # 1. Empty question from owner -> 400
    res_empty = client.post("/ask", json={"doc_id": doc_id, "question": ""}, headers=headers_a)
    assert res_empty.status_code == 400
    
    # 2. Whitespace question from owner -> 400
    res_space = client.post("/ask", json={"doc_id": doc_id, "question": "   "}, headers=headers_a)
    assert res_space.status_code == 400
    
    # 3. Create document in processing state manually, owned by User A
    proc_id = "proc-doc-123"
    repository.create_document(
        doc_id=proc_id,
        filename="processing.pdf",
        stored_path="data/uploads/processing.pdf",
        file_type="pdf",
        page_count=0,
        status="processing",
        owner_id="user_a"
    )
    
    # Querying incomplete document state from owner -> 400
    res_proc = client.post("/retrieve", json={"doc_id": proc_id, "question": "test query"}, headers=headers_a)
    assert res_proc.status_code == 400

