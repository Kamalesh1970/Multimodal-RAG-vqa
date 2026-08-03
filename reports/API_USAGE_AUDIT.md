# VLM API Usage Audit

## Executive Summary
This document provides a comprehensive API usage and token/cost efficiency audit for the Multimodal RAG VQA application. We inspect every file, function, and component to identify all external API triggers, establish an API call map, trace complete request cycles, analyze retry behaviors, identify redundant calls, and propose optimizations.

---

## Current Architecture
The system employs a hybrid pipeline:
1. **Document Ingestion (Local)**: PDFs and images are processed, text extracted via local PaddleOCR, text/image embeddings generated locally via SentenceTransformers and OpenCLIP, and indexed locally in FAISS.
2. **Retrieval (Local)**: Search queries are embedded locally and mapped to FAISS indexes; score fusion is performed locally.
3. **Answer Generation (External)**: Grounded VQA prompts and images are sent to Gemini or OpenAI for structured output generation.

---

## Every External API Call Site

| File | Function | Provider | Trigger | Calls per operation | Necessary? |
|------|----------|----------|---------|---------------------|------------|
| `backend/generation/gemini_client.py` | `generate_content_with_retry()` | Gemini | `POST /ask` | 1 initial + `GEMINI_MAX_RETRIES` (default 2) | Yes (when VLM_PROVIDER=gemini) |
| `backend/generation/openai_client.py` | `generate_openai_content_with_retry()` | OpenAI | `POST /ask` | 1 initial + `OPENAI_MAX_RETRIES` (default 2) | Yes (when VLM_PROVIDER=openai) |

---

## Upload Call Trace
**Endpoint**: `POST /documents/upload`
1. Request received at `backend/main.py:upload_document`.
2. File saved to disk at `backend/ingestion/processor.py:save_to_database`.
3. PaddleOCR processes images locally at `backend/ingestion/ocr.py`.
4. Page images saved locally to `data/processed/{doc_id}/page_{page_number}.jpg`.
5. Local embedding vectors generated at `backend/embeddings/text_embedder.py` and `backend/embeddings/image_embedder.py`.
6. FAISS index updated locally at `backend/vector_store.py`.

* **Gemini calls**: 0
* **OpenAI calls**: 0
* **Other paid API calls**: 0

*Conclusion*: Upload is fully local. No VLM calls are made. No captions are generated.

---

## Retrieval Call Trace
**Endpoint**: `POST /retrieve`
1. Request received at `backend/main.py:retrieve_document_evidence`.
2. Input question text embedded locally via `TextEmbedder` and `ImageEmbedder`.
3. Search flat L2/inner product distance computed on FAISS text and image indexes.
4. Min-max normalization and Weighted Available Modalities (WAM) score fusion performed.
5. Deterministic tie-breaker ranking applied.

* **Gemini calls**: 0
* **OpenAI calls**: 0
* **Other paid API calls**: 0

*Conclusion*: Retrieval is fully local. No VLM calls are made.

---

## Ask Call Trace
**Endpoint**: `POST /ask`

### Call Graph
```
/ask (main.py:ask_grounded_question)
  ↓
generate_grounded_answer() (answer_generator.py)
  ↓
  ├─ retrieve_evidence() (retrieval.py) [Local: 0 calls]
  ├─ load PIL image(s) [Local: 0 calls]
  └─ VLM API Call:
       ├─ generate_openai_content_with_retry() (openai_client.py) [OpenAI API]
       └─ OR generate_content_with_retry() (gemini_client.py) [Gemini API]
```

### Call Budget
* **Minimum calls**: 1
* **Normal calls**: 1
* **Maximum attempts (including retries)**: 3 (1 initial + 2 retries)

---

## Retry Analysis
Both clients implement a simple retry loop:
- `for attempt in range(max_retries + 1):`
If `max_retries = 2`, the loop runs up to 3 times (1 initial call, plus up to 2 retries).
Currently, the clients retry **all** exceptions that match transient codes (429, 500, 503).
However, an OpenAI `insufficient_quota` (billing exhaustion) or a Gemini `limit: 0` error is also a 429, which gets retried. This wastes time, increases log size, and delays returning the error to the user.

---

## Duplicate Call Analysis
- No duplicate VLM calls are present in the core flow.
- No LLM-based query classification, LLM-based page filtering, or LLM-based answer verification exists in the production code path.
- The `pytest` suite, however, runs live tests automatically if API keys are set, consuming paid tokens.

---

## Provider Fallback Analysis
- There is no automatic cross-provider fallback. If the active provider fails, the error is immediately logged and returned to the user.

---

## Test Suite API Consumption
- `tests/test_generation.py` contains:
  - 2 mocked offline tests.
  - 6 live VLM tests (exact OCR, visual shape, chart, unanswerable, multipage, prompt injection).
  - 1 live evaluation metrics test (making 20 sequential requests).
- Total live API calls: **26** per pytest run (if VLM keys are configured).
- Standard `pytest` runs these live tests automatically without asking, risking high API consumption.

---

## Prompt Size Analysis
- Current prompt includes long Pydantic JSON schema descriptions and double instructions.
- All OCR text is sent for the retrieved pages. Under broad summary queries, pages can scale up to 10 pages, increasing prompt token count.

---

## OCR Context Analysis
- Currently, the application sends the *entire* OCR text of every retrieved page. If `top_k` is 3 or more, large documents can produce very long prompts.

---

## Image Payload Analysis
- The application reads original page JPEG layout images and attaches them. The images are stored at their default rendered resolutions (up to 224x224 or fitz default pages). No limit on image sizes is currently enforced during API transmission.

---

## Output Token Analysis
- Output tokens are not currently capped via `max_tokens` or `max_output_tokens` in the client calls, relying on default provider settings which may return verbose answers.

---

## Problems Found
1. **Quota / Billing Exhaustion Retries**: Quota limits (HTTP 429 with `insufficient_quota` or `limit: 0`) are retried up to 2 times, causing unnecessary delay.
2. **Implicit Pytest API Consumption**: Standard `pytest` automatically invokes live tests if keys are present in `.env`.
3. **No Request ID Propagation**: Lack of a request ID makes it difficult to verify in logs that one `/ask` query maps to exactly one API call.
4. **Lack of Token Usage Metadata**: Token usage metrics (input, output, total tokens) returned by providers are not captured or logged.
5. **No Image Resizing Limit**: Large high-resolution images are sent raw, increasing base64 payload size and VLM input token cost.
6. **No Output Token Limit**: Verbose answers can be generated since no max token limits are set on completions.

---

## Recommendations / Optimizations Plan
1. **Disable Quota Retries**: Check for `insufficient_quota` (OpenAI) and `"limit: 0"` / `"quota"` (Gemini) messages, throwing immediately without retrying.
2. **Add `ENABLE_LIVE_VLM_TESTS` safety guard**: Skip all live tests unless `ENABLE_LIVE_VLM_TESTS=true` is set.
3. **Inject Request ID**: Create a UUID request ID for each `/ask` query, passing it through to client calls.
4. **Log Token Usage**: Extract input/output tokens from completions response and print a clean development telemetry summary.
5. **Add Call Counter**: Count VLM calls per ask request and return it in the response under a debug flag.
6. **Configurable Prompt and Context Limits**: Introduce `MAX_OCR_CONTEXT_CHARS`, `MAX_VLM_IMAGES`, and image resolution bounds.
