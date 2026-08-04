import sqlite3
import logging
from contextlib import contextmanager
from backend.config import settings

logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """
    Context manager to safely acquire and release a database connection.
    Ensures rollback on exceptions and commit on success.
    """
    # Ensure database directory exists before connecting
    settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(settings.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    try:
        # Enable Foreign Key support in SQLite
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction failed: {e}")
        raise e
    finally:
        conn.close()

def init_db() -> None:
    """
    Initializes the database schema.
    Creates tables 'documents' and 'pages' if they do not exist.
    Performs safe migration additions for existing databases.
    """
    logger.info(f"Initializing database at: {settings.DATABASE_PATH}")
    
    create_documents_sql = """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT UNIQUE NOT NULL,
        filename TEXT NOT NULL,
        stored_path TEXT,
        file_type TEXT NOT NULL,
        page_count INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT NOT NULL
    );
    """
    
    create_pages_sql = """
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        page_number INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        ocr_text TEXT NOT NULL,
        ocr_blocks_json TEXT NOT NULL,
        text_embedding_indexed INTEGER DEFAULT 0,
        image_embedding_indexed INTEGER DEFAULT 0,
        text_embedding_model TEXT,
        image_embedding_model TEXT,
        FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE,
        UNIQUE (doc_id, page_number)
    );
    """

    create_chat_sessions_sql = """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        owner_id TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
    );
    """

    create_chat_messages_sql = """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata_json TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id) ON DELETE CASCADE
    );
    """
    
    try:
        with get_db_connection() as conn:
            # 1. Create documents table if not exists
            conn.execute(create_documents_sql)
            
            # 2. Perform migrations for Phase 1 databases to add missing columns
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(documents);")
            existing_columns = [row["name"] for row in cursor.fetchall()]
            
            if "stored_path" not in existing_columns:
                logger.info("Migrating database: adding 'stored_path' to documents table.")
                conn.execute("ALTER TABLE documents ADD COLUMN stored_path TEXT;")
                
            if "page_count" not in existing_columns:
                logger.info("Migrating database: adding 'page_count' to documents table.")
                conn.execute("ALTER TABLE documents ADD COLUMN page_count INTEGER DEFAULT 0;")
            
            # 3. Create pages table if not exists
            conn.execute(create_pages_sql)
            
            # 4. Perform migrations for pages table to add missing embedding columns if needed
            cursor.execute("PRAGMA table_info(pages);")
            existing_pages_columns = [row["name"] for row in cursor.fetchall()]
            
            if "text_embedding_indexed" not in existing_pages_columns:
                logger.info("Migrating database: adding 'text_embedding_indexed' to pages table.")
                conn.execute("ALTER TABLE pages ADD COLUMN text_embedding_indexed INTEGER DEFAULT 0;")
                
            if "image_embedding_indexed" not in existing_pages_columns:
                logger.info("Migrating database: adding 'image_embedding_indexed' to pages table.")
                conn.execute("ALTER TABLE pages ADD COLUMN image_embedding_indexed INTEGER DEFAULT 0;")
                
            if "text_embedding_model" not in existing_pages_columns:
                logger.info("Migrating database: adding 'text_embedding_model' to pages table.")
                conn.execute("ALTER TABLE pages ADD COLUMN text_embedding_model TEXT;")
                
            if "image_embedding_model" not in existing_pages_columns:
                logger.info("Migrating database: adding 'image_embedding_model' to pages table.")
                conn.execute("ALTER TABLE pages ADD COLUMN image_embedding_model TEXT;")

            # 5. Create chat tables if not exist
            conn.execute(create_chat_sessions_sql)
            conn.execute(create_chat_messages_sql)
            
        logger.info("Database initialization and migrations completed successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise e

