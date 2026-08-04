import logging
import hashlib
import struct
from backend.config import settings

logger = logging.getLogger(__name__)

def generate_deterministic_page_id(doc_id: str, page_number: int) -> int:
    """
    Generates a deterministic positive 63-bit integer page ID from (doc_id, page_number).
    This ID fits into signed 64-bit integers mapped in FAISS (np.int64).
    """
    h = hashlib.sha256(f"{doc_id}_{page_number}".encode()).digest()
    val = struct.unpack(">q", h[:8])[0]
    return val & 0x7FFFFFFFFFFFFFFF

def serialize_firestore_value(val):
    """
    Serializes standard python/numpy types to Firestore-compatible structures.
    """
    if hasattr(val, "item"):  # numpy scalars
        return val.item()
    if isinstance(val, dict):
        return {k: serialize_firestore_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [serialize_firestore_value(x) for x in val]
    return val

def init_storage():
    """
    Initializes active storage provider database.
    """
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import initialize_firebase
        initialize_firebase()
    else:
        from backend.database import init_db
        init_db()

def is_db_connected() -> bool:
    """
    Returns True if the database provider is active and successfully connected.
    """
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import is_firebase_available
        return is_firebase_available()
    else:
        try:
            from backend.database import get_db_connection
            with get_db_connection() as _conn:
                pass
            return True
        except Exception:
            return False

def get_db_provider() -> str:
    """
    Returns 'firestore' if firebase is enabled, 'local' (or 'sqlite') otherwise.
    """
    return "firestore" if settings.FIREBASE_ENABLED else "local"

def create_document(doc_id: str, filename: str, stored_path: str, file_type: str, page_count: int, status: str) -> None:
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import get_firestore_client
        from google.cloud import firestore
        db = get_firestore_client()
        doc_ref = db.collection("documents").document(doc_id)
        doc_ref.set({
            "doc_id": doc_id,
            "filename": filename,
            "stored_path": stored_path,
            "file_type": file_type,
            "page_count": page_count,
            "status": status,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "owner_id": None
        })
    else:
        from backend.database import get_db_connection
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, filename, stored_path, file_type, page_count, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, filename, stored_path, file_type, page_count, status)
            )

def get_document(doc_id: str) -> dict | None:
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import get_firestore_client
        db = get_firestore_client()
        doc_ref = db.collection("documents").document(doc_id)
        snapshot = doc_ref.get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict()
        if "created_at" in data and data["created_at"]:
            if hasattr(data["created_at"], "isoformat"):
                data["created_at"] = data["created_at"].isoformat()
            else:
                data["created_at"] = str(data["created_at"])
        return data
    else:
        from backend.database import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT doc_id, filename, stored_path, file_type, page_count, status, created_at FROM documents WHERE doc_id = ?",
                (doc_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "stored_path": row["stored_path"],
                "file_type": row["file_type"],
                "page_count": row["page_count"],
                "status": row["status"],
                "created_at": row["created_at"]
            }

def update_document(doc_id: str, status: str, page_count: int | None = None, stored_path: str | None = None) -> None:
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import get_firestore_client
        from google.cloud import firestore
        db = get_firestore_client()
        doc_ref = db.collection("documents").document(doc_id)
        updates = {
            "status": status,
            "updated_at": firestore.SERVER_TIMESTAMP
        }
        if page_count is not None:
            updates["page_count"] = page_count
        if stored_path is not None:
            updates["stored_path"] = stored_path
        doc_ref.update(updates)
    else:
        from backend.database import get_db_connection
        with get_db_connection() as conn:
            if page_count is not None and stored_path is not None:
                conn.execute(
                    "UPDATE documents SET status = ?, page_count = ?, stored_path = ? WHERE doc_id = ?",
                    (status, page_count, stored_path, doc_id)
                )
            elif page_count is not None:
                conn.execute(
                    "UPDATE documents SET status = ?, page_count = ? WHERE doc_id = ?",
                    (status, page_count, doc_id)
                )
            else:
                conn.execute(
                    "UPDATE documents SET status = ? WHERE doc_id = ?",
                    (status, doc_id)
                )

def save_page(doc_id: str, page_number: int, width: int, height: int, ocr_text: str, ocr_blocks_json: str) -> int:
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import get_firestore_client
        from google.cloud import firestore
        db = get_firestore_client()
        page_id = generate_deterministic_page_id(doc_id, page_number)
        page_ref = db.collection("documents").document(doc_id).collection("pages").document(f"page_{page_number}")
        page_ref.set({
            "page_id": page_id,
            "doc_id": doc_id,
            "page_number": page_number,
            "width": width,
            "height": height,
            "ocr_text": ocr_text,
            "ocr_blocks_json": ocr_blocks_json,
            "text_embedding_indexed": 0,
            "image_embedding_indexed": 0,
            "text_embedding_model": None,
            "image_embedding_model": None,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        return page_id
    else:
        from backend.database import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO pages (doc_id, page_number, width, height, ocr_text, ocr_blocks_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, page_number, width, height, ocr_text, ocr_blocks_json)
            )
            return cursor.lastrowid

def update_page_indexing(doc_id: str, page_number: int, text_indexed: bool, image_indexed: bool, text_model: str | None, image_model: str | None) -> None:
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import get_firestore_client
        db = get_firestore_client()
        page_ref = db.collection("documents").document(doc_id).collection("pages").document(f"page_{page_number}")
        page_ref.update({
            "text_embedding_indexed": 1 if text_indexed else 0,
            "image_embedding_indexed": 1 if image_indexed else 0,
            "text_embedding_model": text_model if text_indexed else None,
            "image_embedding_model": image_model if image_indexed else None
        })
    else:
        from backend.database import get_db_connection
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE pages
                SET text_embedding_indexed = ?,
                    image_embedding_indexed = ?,
                    text_embedding_model = ?,
                    image_embedding_model = ?
                WHERE doc_id = ? AND page_number = ?
                """,
                (
                    1 if text_indexed else 0,
                    1 if image_indexed else 0,
                    text_model if text_indexed else None,
                    image_model if image_indexed else None,
                    doc_id,
                    page_number
                )
            )

def get_pages(doc_id: str) -> list[dict]:
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import get_firestore_client
        db = get_firestore_client()
        pages_ref = db.collection("documents").document(doc_id).collection("pages")
        docs = pages_ref.order_by("page_number").get()
        results = []
        for doc in docs:
            data = doc.to_dict()
            results.append({
                "id": data.get("page_id"),
                "page_id": data.get("page_id"),
                "doc_id": data.get("doc_id"),
                "page_number": data.get("page_number"),
                "width": data.get("width"),
                "height": data.get("height"),
                "ocr_text": data.get("ocr_text"),
                "ocr_blocks_json": data.get("ocr_blocks_json"),
                "text_embedding_indexed": data.get("text_embedding_indexed"),
                "image_embedding_indexed": data.get("image_embedding_indexed"),
                "text_embedding_model": data.get("text_embedding_model"),
                "image_embedding_model": data.get("image_embedding_model")
            })
        return results
    else:
        from backend.database import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, doc_id, page_number, width, height, ocr_text, ocr_blocks_json,
                       text_embedding_indexed, image_embedding_indexed, 
                       text_embedding_model, image_embedding_model 
                FROM pages WHERE doc_id = ? ORDER BY page_number ASC
                """,
                (doc_id,)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "page_id": row["id"],
                    "doc_id": row["doc_id"],
                    "page_number": row["page_number"],
                    "width": row["width"],
                    "height": row["height"],
                    "ocr_text": row["ocr_text"],
                    "ocr_blocks_json": row["ocr_blocks_json"],
                    "text_embedding_indexed": row["text_embedding_indexed"],
                    "image_embedding_indexed": row["image_embedding_indexed"],
                    "text_embedding_model": row["text_embedding_model"],
                    "image_embedding_model": row["image_embedding_model"]
                })
            return results

def get_all_pages_sync_info() -> list[dict]:
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import get_firestore_client
        db = get_firestore_client()
        docs = db.collection_group("pages").get()
        results = []
        for doc in docs:
            data = doc.to_dict()
            results.append({
                "id": data.get("page_id"),
                "doc_id": data.get("doc_id"),
                "page_number": data.get("page_number"),
                "ocr_text": data.get("ocr_text")
            })
        return results
    else:
        from backend.database import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, doc_id, page_number, ocr_text FROM pages")
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "doc_id": row["doc_id"],
                    "page_number": row["page_number"],
                    "ocr_text": row["ocr_text"]
                })
            return results

def create_chat_session(session_id: str, doc_id: str, owner_id: str | None = None) -> None:
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import get_firestore_client
        from google.cloud import firestore
        db = get_firestore_client()
        doc_ref = db.collection("chat_sessions").document(session_id)
        doc_ref.set({
            "session_id": session_id,
            "doc_id": doc_id,
            "owner_id": owner_id,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
    else:
        from backend.database import get_db_connection
        with get_db_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO chat_sessions (session_id, doc_id, owner_id) VALUES (?, ?, ?)",
                (session_id, doc_id, owner_id)
            )

def save_chat_message(session_id: str, role: str, content: str, doc_id: str, metadata: dict | None = None) -> str:
    import uuid
    message_id = str(uuid.uuid4())
    serialized_meta = serialize_firestore_value(metadata or {})
    
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import get_firestore_client
        from google.cloud import firestore
        db = get_firestore_client()
        
        session_ref = db.collection("chat_sessions").document(session_id)
        if not session_ref.get().exists:
            session_ref.set({
                "session_id": session_id,
                "doc_id": doc_id,
                "owner_id": None,
                "created_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP
            })
        else:
            session_ref.update({"updated_at": firestore.SERVER_TIMESTAMP})
            
        msg_ref = session_ref.collection("messages").document(message_id)
        msg_ref.set({
            "message_id": message_id,
            "role": role,
            "content": content,
            "metadata": serialized_meta,
            "created_at": firestore.SERVER_TIMESTAMP
        })
    else:
        from backend.database import get_db_connection
        import json
        with get_db_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO chat_sessions (session_id, doc_id, owner_id) VALUES (?, ?, ?)",
                (session_id, doc_id, None)
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, metadata_json) VALUES (?, ?, ?, ?)",
                (session_id, role, content, json.dumps(serialized_meta))
            )
    return message_id

def get_chat_history(session_id: str) -> list[dict]:
    if settings.FIREBASE_ENABLED:
        from backend.firebase.client import get_firestore_client
        db = get_firestore_client()
        session_ref = db.collection("chat_sessions").document(session_id)
        docs = session_ref.collection("messages").order_by("created_at").get()
        results = []
        for doc in docs:
            data = doc.to_dict()
            if "created_at" in data and data["created_at"]:
                if hasattr(data["created_at"], "isoformat"):
                    data["created_at"] = data["created_at"].isoformat()
                else:
                    data["created_at"] = str(data["created_at"])
            results.append(data)
        return results
    else:
        from backend.database import get_db_connection
        import json
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content, metadata_json, created_at FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                try:
                    meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                except Exception:
                    meta = {}
                results.append({
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": meta,
                    "created_at": row["created_at"]
                })
            return results
