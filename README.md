# Multimodal RAG for Visual Question Answering (VQA)

A Multimodal Retrieval-Augmented Generation (RAG) system built with FastAPI and SQLite, designed to perform Visual Question Answering (VQA) on document screenshots, figures, and charts using Google's Gemini API.

## Development Phase Status
- **Phase 1 — Project Foundation (Completed)**
- **Phase 2 — Document Ingestion & OCR (Completed)**
- **Phase 3 — Multimodal Embeddings & Vector Storage (Completed)**
- **Phase 4 — Multimodal Retrieval (Not Started)**

---

## Features (Phase 2 Ingestion & OCR)
- **Supported Formats**: `.pdf`, `.png`, `.jpg`, `.jpeg`
- **Configurable Upload Limit**: Defaulted to 20MB (configured via `MAX_UPLOAD_MB` in `.env`).
- **OCR Engine**: English text extraction powered by `PaddleOCR` (lazy-initialized singleton).
- **PDF Rendering**: Multi-page PDF document split and page rendering to raw image bytes at 200 DPI using `PyMuPDF`.
- **Relational Persistence**: Original files stored securely via UUID names in `data/uploads/`, and metadata/structural bounding box coordinates persisted in SQLite tables (`documents` & `pages`).

## Features (Phase 3 Multimodal Embeddings & Vector Storage)
- **Page-Level Embedding Design**: Vector representations are generated independently for each page. Multi-page PDFs generate a text vector + image vector per individual page to ensure retrieval granularity.
- **Text Representation**: Captures OCR semantics using `sentence-transformers/all-MiniLM-L6-v2` producing L2-normalized float32 vectors of dimension **384**.
- **Visual Representation**: Captures document visual structures/layouts using OpenCLIP `ViT-B-32` (`laion2b_s34b_b79k`) producing L2-normalized float32 vectors of dimension **512**.
- **Vector Storage**: Maintains two independent indexes using `FAISS` (`IndexFlatIP` exact flat indexes wrapped with `IndexIDMap2` to map SQLite `pages.id` primary keys deterministically).
- **Device Support**: Set via `EMBEDDING_DEVICE` (`auto` / `cpu` / `cuda`).
- **Disk Persistence**: FAISS indexes are saved to `data/indexes/` and reloaded dynamically on restart.

---

## Requirements
- Python 3.12+
- SQLite3
- System dependencies for PaddleOCR (e.g., C++ redistributable packages on Windows or libgomp/glib on Linux if running non-headless, but managed using `opencv-python-headless`).

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Kamalesh1970/Multimodal-RAG-vqa.git
cd Multimodal-RAG-vqa
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Copy the template configuration file:
```bash
cp .env.example .env
```
Ensure that the local `.env` is updated if necessary. E.g.:
```ini
MAX_UPLOAD_MB=20
```

---

## Running the Application

Start the FastAPI backend server using Uvicorn:
```bash
python -m uvicorn backend.main:app --reload
```

- **API Base URL**: `http://127.0.0.1:8000`
- **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Endpoints

### 1. Health Check
* **GET** `/health`
  * Returns JSON stating the active phase is 2 and status is healthy.

### 2. Document Ingestion Upload
* **POST** `/documents/upload`
  * Accepts `multipart/form-data` upload containing a `file` field.
  * Example Curl Command:
    ```bash
    curl -X POST -F "file=@/path/to/invoice.pdf" http://127.0.0.1:8000/documents/upload
    ```
  * Response Format:
    ```json
    {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "invoice.pdf",
      "file_type": "pdf",
      "status": "completed",
      "pages": [
        {
          "page_number": 1,
          "width": 1654,
          "height": 2339,
          "ocr": {
            "full_text": "Connection Name: TEST-CONNECTION-01\nTotal Amount: 1500",
            "blocks": [
              {
                "text": "Connection Name: TEST-CONNECTION-01",
                "confidence": 0.99,
                "bbox": [[10.0, 20.0], [300.0, 20.0], [300.0, 40.0], [10.0, 40.0]]
              }
            ]
          }
        }
      ]
    }
    ```

### 3. Document Metadata
* **GET** `/documents/{doc_id}`
  * Retrieves stored database metadata (original filename, file type, status, page count, and creation timestamp).

### 4. Fetch Stored Document OCR
* **GET** `/documents/{doc_id}/ocr`
  * Returns saved page-by-page structural OCR text and bounding boxes from the database (does not re-execute OCR).

### 5. Document Embeddings Metadata
* **GET** `/documents/{doc_id}/embeddings`
  * Retrieves embedding metadata (indexing status, dimensions, and active models) for a specific document's pages. Does not return actual vector values.

### 6. System Vector Status
* **GET** `/system/vector-status`
  * Exposes global status information about the text and image vector indexes.

---

## Running Tests

Run the full automated test suite using pytest:
```bash
# Run all tests (including mock and real OCR/embeddings E2E tests)
pytest -v -s

# Run only fast unit tests (skips slow real PaddleOCR and model inference)
pytest -m "not ocr"
```

---

## Project Structure
```
Multimodal-RAG-vqa/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── vector_store.py   # FAISS double-index vector store
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── text_embedder.py  # SentenceTransformers L2 normalized text embeddings
│   │   └── image_embedder.py # OpenCLIP visual layout & query text embeddings
│   └── ingestion/
│       ├── __init__.py
│       ├── ocr.py        # PaddleOCR engine singleton & normalizers
│       └── processor.py  # File validation, image/PDF processing & ingestion orchestration
├── data/
│   ├── uploads/          # Original files stored with UUID names
│   ├── processed/        # Processed page-level visual images (Phase 3)
│   └── indexes/          # Persistent FAISS index files (Phase 3)
├── tests/
│   ├── __init__.py
│   ├── test_health.py
│   ├── test_ingestion.py # Ingestion flow and error handling E2E tests
│   └── test_embeddings.py # Embedding unit, sanity, and persistence tests
├── reports/
│   └── PROJECT_PROGRESS.md
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── pytest.ini
```

---

## Development Roadmap
- [x] **Phase 1 — Project Foundation**: Setup backend server, database, configuration, and logging.
- [x] **Phase 2 — Ingestion & OCR**: PDF rendering, image orientation correction, PaddleOCR integration, metadata persistence, and upload endpoints.
- [x] **Phase 3 — Embeddings & Vector Storage**: Page-level visual and text embedding generation using OpenCLIP/SentenceTransformers and FAISS indexing.
- [ ] **Phase 4 — Retrieval**: FAISS vector search implementation (Not started - Retrieval is Phase 4 and is not yet implemented).
- [ ] **Phase 5 — Gemini VLM**: VLM inference logic using Gemini API.
- [ ] **Phase 6 — Accuracy & Grounding**: Reranking and confidence estimation.
- [ ] **Phase 7 — Custom Frontend**: React/Streamlit interface.
- [ ] **Phase 8 — Evaluation**: Systematic testing of retrieval accuracy and answer quality.
