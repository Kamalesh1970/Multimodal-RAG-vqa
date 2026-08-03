# Phase 7: Performance Optimization & Stability Baseline

This report documents the baseline performance and stability metrics of the Multimodal Retrieval-Augmented Generation (RAG) system for Visual Question Answering (VQA) prior to optimization.

## Historical Baseline Profile

The following baseline metrics are derived from the actual observed runtime logs on the standard evaluation queries:

### 1. Ingestion / Upload Baseline
- **PDF Render DPI:** 200 DPI (Default)
- **OCR Engine:** PaddleOCR (Lazy singleton but with duplicate DB connection overhead)
- **FAISS Disk Writes:** Serialized to disk after *every single* page vector addition, creating high write amplification.
- **Upload Latency:** TBD (dependent on document size)
- **RAM Usage:** ~5 - 6 GB peak RAM on CPU during model loading and ingestion.

### 2. Retrieval Baseline
- **Retrieval Latency:** ~42 ms (extremely fast)
- **Text Embeddings Latency:** ~20 ms
- **Image Embeddings Latency:** ~9 ms
- **FAISS Index Lookup:** ~10 ms
- **WAM Fusion:** ~0.6 ms

### 3. Answer Generation (VLM API) Baseline
- **VLM API Calls:** 1
- **Retrieved Pages Sent:** 2 (always top-k pages sent, regardless of confidence or query intent)
- **Images Sent:** 2 (sent at max 1024 dimension, but without specifying visual detail mode, defaulting to `"auto"` high-detail mode)
- **Original Image Dimensions:** 2480x3508 (A4 300 DPI)
- **Prompt/Input Tokens:** ~51,455 tokens (vision tokens inflated by OpenAI for `gpt-4o-mini` by ~33.3x to match `gpt-4o` image pricing)
- **Output Tokens:** 60 – 316 tokens
- **VLM Latency:** ~5.4 – 6.9 seconds
- **Total Answer Latency:** ~6.9 seconds

---

## Stability Issues Identified

### A. Uvicorn Reload Loop (Backend false-offline)
- **Confirmed Root Cause:** Uvicorn is started with `--reload` in the project root. The application writes database updates, file uploads, preprocessed page images, and FAISS vector indices to the `data/` subdirectory. Since Uvicorn watches the entire workspace, these file writes trigger immediate Uvicorn reloads.
- **Impact:** Reloader terminates the active Python process mid-request or during ingestion, causing temporary unresponsiveness (5-10 seconds) while PyTorch, SentenceTransformers, OpenCLIP, and PaddleOCR are reloaded at boot. The client experiences this as "Backend Offline".

### B. Event-Loop Blocking
- **Confirmed Root Cause:** Ingestion endpoint `/documents/upload` is defined as `async def`, but executes PDF rendering, Pillow preprocessing, PaddleOCR, and local embedding generation synchronously on the main thread.
- **Impact:** The single-threaded FastAPI event loop is blocked for the entire duration of document ingestion. Health check (`/system/status`) and Q&A (`/ask`) requests are queued and appear dead until ingestion completes.

---

## Representative Test Cases for Optimization Verification

We will evaluate the system against the following test cases to ensure that accuracy is preserved and performance targets are met:

| Category | Query | Expected Page | Expected Answer |
|----------|-------|---------------|-----------------|
| **A. Exact OCR** | "What is the account type for Bob Cooper?" | Page 1 | "Enterprise Membership" |
| **B. Document Q**| "Is the payment term net 30 or net 60?" | Page 10 | "Net 30 Days" |
| **C. Visual-only**| "What shape and color is on page 3?" | Page 3 | "yellow circle" |
| **D. Chart** | "What trend does the graph show on page 4?" | Page 4 | "upward" |
| **E. Multi-page** | "What contact details exist for Robert Smith's contract signee?" | Page 5 & 6 | "support@enterprise.com" |
| **F. Unanswerable**| "What is the customer's blood group?" | None | (answerable=false) |

---

## Performance Targets

- **VLM Input Tokens:** Reduce from ~51,440 tokens to `<10,000` tokens (Stretch target `<5,000` tokens).
- **Images per Request:** Reduce from `2` to `1` by default using adaptive Page selection.
- **Total Latency:** Maintain local retrieval latency, reduce VLM answering latency to `2.0 - 4.0` seconds.
- **Ingestion Speed:** Accelerate by using batch embedding generation and single FAISS index save per document.
- **Backend Stability:** Maintain 100% online status by moving ingestion to threadpool and recommending reload exclusions.
