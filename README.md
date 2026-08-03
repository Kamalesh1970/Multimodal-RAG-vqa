# Multimodal RAG for Visual Question Answering (VQA)

A Multimodal Retrieval-Augmented Generation (RAG) system built with FastAPI and SQLite, designed to perform Visual Question Answering (VQA) on document screenshots, figures, and charts using Google's Gemini API.

## Current Development Phase
- **Phase 1 — Project Foundation (Completed)**

---

## Requirements
- Python 3.12+
- SQLite3

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
Ensure that the local `.env` is updated if necessary. (Note: `GEMINI_API_KEY` is a placeholder for future phases).

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

## Running Tests

Run the automated test suite using pytest:
```bash
pytest
```

---

## Project Structure
```
Multimodal-RAG-vqa/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   └── database.py
├── data/
│   ├── uploads/          # User-uploaded documents/images
│   ├── processed/        # Extracted pages/images
│   └── indexes/          # Search indexes
├── tests/
│   ├── __init__.py
│   └── test_health.py
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
- [ ] **Phase 2 — Ingestion & OCR**: PDF splitting, extraction, and PaddleOCR integration.
- [ ] **Phase 3 — Embeddings**: Multimodal embedding generation for text/images.
- [ ] **Phase 4 — Retrieval**: FAISS vector search implementation.
- [ ] **Phase 5 — Gemini VLM**: VLM inference logic using Gemini API.
- [ ] **Phase 6 — Accuracy & Grounding**: Reranking and confidence estimation.
- [ ] **Phase 7 — Custom Frontend**: React/Streamlit interface.
- [ ] **Phase 8 — Evaluation**: Systematic testing of retrieval accuracy and answer quality.

