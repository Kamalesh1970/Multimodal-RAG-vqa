import pytest
from pathlib import Path
from backend.config import settings
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(autouse=True)
def test_db(tmp_path):
    """
    Fixture to isolate database and file storage for each test.
    Overrides backend settings to use a temporary directory.
    """
    # Create temp paths
    temp_db_path = tmp_path / "test_app.db"
    temp_uploads_dir = tmp_path / "uploads"
    temp_processed_dir = tmp_path / "processed"
    temp_indexes_dir = tmp_path / "indexes"
    
    # Save original settings
    orig_db_path = settings.DATABASE_PATH
    orig_uploads_dir = settings.UPLOADS_DIR
    orig_processed_dir = settings.PROCESSED_DIR
    orig_indexes_dir = settings.INDEXES_DIR
    orig_vector_index_dir = settings.VECTOR_INDEX_DIR
    
    # Apply overrides
    settings.DATABASE_PATH = temp_db_path
    settings.UPLOADS_DIR = temp_uploads_dir
    settings.PROCESSED_DIR = temp_processed_dir
    settings.INDEXES_DIR = temp_indexes_dir
    settings.VECTOR_INDEX_DIR = temp_indexes_dir
    
    yield temp_db_path
    
    # Restore original settings
    settings.DATABASE_PATH = orig_db_path
    settings.UPLOADS_DIR = orig_uploads_dir
    settings.PROCESSED_DIR = orig_processed_dir
    settings.INDEXES_DIR = orig_indexes_dir
    settings.VECTOR_INDEX_DIR = orig_vector_index_dir

@pytest.fixture
def client():
    """
    Fixture to provide a TestClient instance within the lifespan context.
    """
    with TestClient(app) as client:
        yield client

@pytest.fixture(autouse=True)
def force_local_vlm():
    """
    Forces VLM_PROVIDER to 'local' for all tests unless FORCE_LIVE_API is set to true.
    """
    import os
    if os.getenv("FORCE_LIVE_API", "false").lower() == "true":
        pass
    else:
        settings.VLM_PROVIDER = "local"
