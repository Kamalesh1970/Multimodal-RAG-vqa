# Multimodal RAG for Visual Question Answering (VQA)

A Multimodal Retrieval-Augmented Generation (RAG) system built with FastAPI and SQLite, designed to perform Visual Question Answering (VQA) on document screenshots, figures, and charts using Google's Gemini API.

## Development Phase Status
- **Phase 1 — Project Foundation (Completed)**
- **Phase 2 — Document Ingestion & OCR (Completed)**
- **Phase 3 — Multimodal Embeddings & Vector Storage (Completed)**
- **Phase 4 — Multimodal Retrieval (Completed)**
- **Phase 5 — Gemini Multimodal Grounded Answer Generation (Completed)**

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

## Features (Phase 4 Multimodal Retrieval)
- **Document-Isolated Retrieval**: Limits candidate queries to the specific document ID requested, preventing cross-tenant or cross-document data leakage.
- **Calibrated Score Fusion**: Evaluates raw similarity scores on a min-max normalized scale within the document candidate set to handle model calibration differences, fusing modalities using configurable text/image weights.
- **Weighted Available Modalities (WAM)**: Dynamically adjusts score weighting when a specific modality is missing (e.g. blank page with no OCR text) to avoid penalizing visual-only pages.
- **Lexical Evidence Selection**: Normalizes query terms to identify and extract the top 3 most relevant localized text blocks from raw OCR structures.
- **Deterministic Tie-Breaking**: Ranks candidates based on fused score descending, then page number ascending, and finally page ID ascending to guarantee stable result lists.

## Features (Phase 5 Gemini Multimodal Grounded Answer Generation)
- **Lazy client connection**: Initialized on first call using Google GenAI SDK (`google-genai`), preventing boot crashes if configuration is absent.
- **Grounded multimodal prompt**: Feeds both OCR textual evidence and corresponding PIL page layout images to Gemini (e.g., `gemini-2.5-flash`), leveraging both visual contexts (slopes, shapes, charts) and textual parameters.
- **Query intent adapter**: Dynamically adjusts retrieval `top_k` depending on query semantics (e.g., retrieving up to 10 context pages for broad summaries, or 3 pages for factual queries).
- **Post-generation grounding checker**: Compares generated text against source OCR coordinates to evaluate source authenticity and classifies grounding mode (`text-supported`, `visual-supported`, or `multimodal-supported`).
- **Prompt injection defense**: Includes explicit system prompt directives classifying document contents as raw DATA blocks only, preventing malicious OCR injection instructions from hijacking LLM reasoning.
- **No-answer rejection**: Enforces clean unanswerable responses (setting `answerable=False`) when context contains insufficient factual data, completely mitigating model hallucinations.

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

### 7. Multimodal Retrieval
* **POST** `/retrieve`
  * Submits query question text to get matching document page evidence (does not generate natural language text answers).
  * Request Format:
    ```json
    {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "question": "What is the total amount?",
      "top_k": 3
    }
    ```
  * Response Format:
    ```json
    {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "question": "What is the total amount?",
      "results": [
        {
          "page_id": 12,
          "page_number": 2,
          "scores": {
            "text": 0.8245,
            "image": 0.2201,
            "fused": 0.6548
          },
          "matched_modalities": ["text", "image"],
          "evidence_text": ["Total Amount", "1500 rupees"]
        }
      ]
    }
    ```

### 8. Multimodal Question Answering (VQA)
* **POST** `/ask`
  * Submits query question to retrieve relevant pages, passes visual layouts and OCR text to Gemini, and returns a fully grounded, cited answer.
  * Request Format:
    ```json
    {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "question": "What plan does Alice Johnson subscribe to?",
      "top_k": 3
    }
    ```
  * Response Format:
    ```json
    {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "question": "What plan does Alice Johnson subscribe to?",
      "answer": "Premium Package",
      "answerable": true,
      "grounding_explanation": "Alice Johnson's subscription details are visible on Page 2.",
      "pages_used": [2],
      "grounding_type": "text-supported",
      "evidence": [
        {
          "page_number": 2,
          "text": "Plan Subscription: Premium Package",
          "bbox": [[20.0, 50.0], [300.0, 50.0], [300.0, 70.0], [20.0, 70.0]]
        }
      ],
      "retrieval": {
        "top_score": 0.8245,
        "pages_considered": [1, 2, 3]
      }
    }
    ```

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

## Evaluation & Benchmarking (Phase 6)

The RAG system contains offline evaluation and optimization scripts to run retrieval and VQA benchmarks:

### 1. Run Retrieval Evaluation (Mode A)
Executes offline retrieval recall and fusion experiments without requiring paid VLM API credentials:
```bash
python scripts/evaluate_retrieval.py
```
This script runs a grid search on text/image weights, min-max score normalization, and Reciprocal Rank Fusion (RRF), saving metrics to [reports/evaluation/retrieval_optimized.json](file:///home/kamalesh/RAG_Project/reports/evaluation/retrieval_optimized.json).

### 2. Run VQA Generation Evaluation (Mode B & C)
Executes the visual question answering pipeline, scoring exact-value accuracy, unanswerable queries, and prompt injection resistance:
```bash
# Mode B: Simulated Local VLM
python scripts/evaluate_vqa.py --provider simulated

# Mode C: Real Gemini/OpenAI API (Requires active API keys and ENABLE_LIVE_VLM_TESTS=true)
python scripts/evaluate_vqa.py --provider gemini
```
All results are saved to [reports/evaluation/generation_results.json](file:///home/kamalesh/RAG_Project/reports/evaluation/generation_results.json).
Detailed analysis is documented in [reports/PHASE6_EVALUATION.md](file:///home/kamalesh/RAG_Project/reports/PHASE6_EVALUATION.md).

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
│   ├── retrieval.py      # Multimodal query retrieval, score fusion, & evidence extraction
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── text_embedder.py  # SentenceTransformers L2 normalized text embeddings
│   │   └── image_embedder.py # OpenCLIP visual layout & query text embeddings
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── gemini_client.py  # Lazy API client config and retries
│   │   └── answer_generator.py # Multimodal context formatting, WAM, and post-grounding check
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
│   ├── test_embeddings.py # Embedding unit, sanity, and persistence tests
│   ├── test_retrieval.py # Query retrieval validation and performance metrics comparison
│   └── test_generation.py # Grounded VQA answer generation unit, visual, and prompt-injection tests
├── reports/
│   ├── PROJECT_PROGRESS.md
│   └── PHASE6_EVALUATION.md # Phase 6 evaluation report
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
- [x] **Phase 4 — Multimodal Retrieval**: Exact search, document-isolation filtering, min-max score calibration, and WAM score fusion.
- [x] **Phase 5 — Gemini Multimodal Grounded Answer Generation**: Structured VQA answers extraction, prompt-injection defense, no-answer checks, and citation metadata.
- [x] **Phase 6 — Evaluation, Accuracy Improvement & RAG Optimization**: Systematically evaluate and optimize retrieval using Recall@K, MRR, min-max score calibration, query-aware weight adapters, and RRF.
- [ ] **Phase 7 — Custom Frontend**: React/Streamlit interface.
- [ ] **Phase 8 — Evaluation**: Systematic testing of retrieval accuracy and answer quality.
