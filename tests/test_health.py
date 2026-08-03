# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from backend.main import app
from backend.config import settings
from backend.database import get_db_connection

def test_read_root():
    """
    Test that GET / returns the correct status and project info.
    """
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        json_data = response.json()
        assert "project" in json_data
        assert json_data["phase"] == 1
        assert json_data["status"] == "running"

def test_health_check():
    """
    Test that GET /health returns the correct phase 1 healthy state.
    """
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        json_data = response.json()
        assert json_data["status"] == "healthy"
        assert json_data["service"] == "Multimodal RAG VQA"
        assert json_data["phase"] == 1

def test_database_initialization():
    """
    Verify that the SQLite database exists and schema is correctly created.
    """
    with TestClient(app) as _client:
        assert settings.DATABASE_PATH.exists()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents';")
            table = cursor.fetchone()
            assert table is not None
            assert table["name"] == "documents"
            
            # Verify columns of the 'documents' table
            cursor.execute("PRAGMA table_info(documents);")
            columns = [row["name"] for row in cursor.fetchall()]
            assert "id" in columns
            assert "doc_id" in columns
            assert "filename" in columns
            assert "file_type" in columns
            assert "created_at" in columns
            assert "status" in columns
