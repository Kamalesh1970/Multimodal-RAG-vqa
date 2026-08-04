import pytest
from unittest.mock import patch, MagicMock
from backend.config import settings
from backend.storage import repository

@pytest.fixture(autouse=True)
def reset_firebase_settings():
    """Fixture to ensure settings are reset after each test."""
    orig_enabled = settings.FIREBASE_ENABLED
    orig_path = settings.FIREBASE_CREDENTIALS_PATH
    orig_project_id = settings.FIREBASE_PROJECT_ID
    
    yield
    
    settings.FIREBASE_ENABLED = orig_enabled
    settings.FIREBASE_CREDENTIALS_PATH = orig_path
    settings.FIREBASE_PROJECT_ID = orig_project_id

def test_firebase_disabled():
    """1. Test Firebase disabled mode."""
    settings.FIREBASE_ENABLED = False
    assert repository.get_db_provider() == "local"
    assert repository.is_db_connected() is True  # SQLite is always connected locally

def test_missing_credentials():
    """2. Test missing credentials path."""
    settings.FIREBASE_ENABLED = True
    settings.FIREBASE_CREDENTIALS_PATH = ""
    
    from backend.firebase.client import initialize_firebase
    with patch("backend.firebase.client._firebase_app", None):
        with patch("backend.firebase.client._firestore_client", None):
            success = initialize_firebase()
            assert success is False
            assert repository.is_db_connected() is False

def test_firebase_initialization_mocked():
    """3. Test successful initialization with mocked credentials Certificate."""
    settings.FIREBASE_ENABLED = True
    settings.FIREBASE_CREDENTIALS_PATH = "secrets/firebase-service-account.json"
    
    from backend.firebase.client import initialize_firebase
    with patch("backend.firebase.client.credentials.Certificate") as mock_cert:
        with patch("backend.firebase.client.firebase_admin.initialize_app") as mock_init:
            with patch("backend.firebase.client.firestore.client") as mock_firestore:
                with patch("backend.firebase.client._firebase_app", None):
                    with patch("backend.firebase.client._firestore_client", None):
                        success = initialize_firebase()
                        assert success is True
                        mock_cert.assert_called_once()
                        mock_init.assert_called_once()
                        mock_firestore.assert_called_once()

def test_duplicate_initialization_safety():
    """4. Test duplicate initialization does not re-initialize app."""
    settings.FIREBASE_ENABLED = True
    
    from backend.firebase.client import initialize_firebase
    with patch("backend.firebase.client.firebase_admin._apps", ["mock_app"]):
        with patch("backend.firebase.client.firebase_admin.get_app") as mock_get:
            with patch("backend.firebase.client.firestore.client") as mock_firestore:
                with patch("backend.firebase.client._firebase_app", None):
                    with patch("backend.firebase.client._firestore_client", None):
                        success = initialize_firebase()
                        assert success is True
                        mock_get.assert_called_once()
                        mock_firestore.assert_called_once()

def test_firestore_repository_mocked():
    """5. Test Firestore document repository save/retrieve functions under mocked database."""
    settings.FIREBASE_ENABLED = True
    
    mock_db = MagicMock()
    mock_doc_ref = MagicMock()
    mock_db.collection.return_value.document.return_value = mock_doc_ref
    
    with patch("backend.firebase.client.get_firestore_client", return_value=mock_db):
        # Test create document
        repository.create_document("test-doc-1", "test.pdf", "data/test.pdf", "pdf", 0, "processing")
        mock_db.collection.assert_called_with("documents")
        mock_db.collection.return_value.document.assert_called_with("test-doc-1")
        mock_doc_ref.set.assert_called_once()
        
        # Test retrieve document
        mock_snapshot = MagicMock()
        mock_snapshot.exists = True
        mock_snapshot.to_dict.return_value = {
            "doc_id": "test-doc-1",
            "filename": "test.pdf",
            "status": "processing",
            "created_at": "2026-08-04"
        }
        mock_doc_ref.get.return_value = mock_snapshot
        
        doc = repository.get_document("test-doc-1")
        assert doc is not None
        assert doc["doc_id"] == "test-doc-1"
        assert doc["filename"] == "test.pdf"

def test_page_save_read_mocked():
    """6. Test page metadata save and read via mocked Firestore."""
    settings.FIREBASE_ENABLED = True
    
    mock_db = MagicMock()
    mock_page_ref = MagicMock()
    # Chain: db.collection("documents").document(doc_id).collection("pages").document(...)
    mock_db.collection.return_value.document.return_value.collection.return_value.document.return_value = mock_page_ref
    
    # Mock order_by().get()
    mock_pages_collection = MagicMock()
    mock_db.collection.return_value.document.return_value.collection.return_value = mock_pages_collection
    mock_doc_snap = MagicMock()
    mock_doc_snap.to_dict.return_value = {
        "page_id": 12345,
        "doc_id": "test-doc-1",
        "page_number": 1,
        "width": 100,
        "height": 200,
        "ocr_text": "hello"
    }
    mock_pages_collection.order_by.return_value.get.return_value = [mock_doc_snap]
    
    with patch("backend.firebase.client.get_firestore_client", return_value=mock_db):
        # Save page
        pid = repository.save_page("test-doc-1", 1, 100, 200, "hello", "[]")
        assert pid is not None
        
        # Get pages
        pages = repository.get_pages("test-doc-1")
        assert len(pages) == 1
        assert pages[0]["ocr_text"] == "hello"
        assert pages[0]["page_id"] == 12345

def test_chat_message_persistence_mocked():
    """7. Test chat session and message logging via mocked Firestore."""
    settings.FIREBASE_ENABLED = True
    
    mock_db = MagicMock()
    mock_session_ref = MagicMock()
    mock_msg_ref = MagicMock()
    
    mock_db.collection.return_value.document.return_value = mock_session_ref
    mock_session_ref.collection.return_value.document.return_value = mock_msg_ref
    mock_session_ref.get.return_value.exists = True
    
    with patch("backend.firebase.client.get_firestore_client", return_value=mock_db):
        msg_id = repository.save_chat_message(
            session_id="session-1",
            role="user",
            content="What is this?",
            doc_id="doc-1"
        )
        assert msg_id is not None
        mock_msg_ref.set.assert_called_once()

def test_firestore_failure_handling():
    """8. Test graceful handling when Firestore connection / API call fails."""
    settings.FIREBASE_ENABLED = True
    
    # Mock collection method to raise exception
    mock_db = MagicMock()
    mock_db.collection.side_effect = Exception("Firestore API unavailable")
    
    with patch("backend.firebase.client.get_firestore_client", return_value=mock_db):
        with pytest.raises(Exception) as exc_info:
            repository.create_document("test-doc-1", "test.pdf", "data/test.pdf", "pdf", 0, "processing")
        assert "Firestore API unavailable" in str(exc_info.value)
