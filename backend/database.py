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
    Creates table 'documents' if it does not exist.
    """
    logger.info(f"Initializing database at: {settings.DATABASE_PATH}")
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT UNIQUE NOT NULL,
        filename TEXT NOT NULL,
        file_type TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT NOT NULL
    );
    """
    try:
        with get_db_connection() as conn:
            conn.execute(create_table_sql)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise e
