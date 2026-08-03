import os
import sys
import json
import time
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app
from backend.config import settings
from backend.database import get_db_connection

client = TestClient(app)

def load_benchmark_dataset():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "tests", "fixtures", "evaluation", "benchmark_dataset.json")
    with open(path, "r") as f:
        return json.load(f)

def get_latest_benchmark_doc():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT doc_id FROM documents WHERE filename = 'benchmark_eval.pdf' AND status = 'completed' ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
    if not row:
        raise RuntimeError("Benchmark document not found. Please run evaluate_retrieval.py first to build and upload it.")
    return row["doc_id"]

def normalize_exact_value(value: str) -> str:
    """Applies basic normalization for exact value comparison (whitespaces, commas, currencies)."""
    val = value.lower().strip()
    val = val.replace(",", "")
    val = val.replace("inr", "").replace("usd", "").replace("rupees", "").strip()
    return val

def run_vqa_evaluation(provider: str, limit_live: bool = False):
    print("=" * 60)
    print(f"Executing VQA Generation Evaluation (Provider: {provider.upper()}, Limit Live: {limit_live})")
    print("=" * 60)
    
    # Save original provider
    orig_provider = settings.VLM_PROVIDER
    
    # Map provider setting
    if provider == "simulated":
        settings.VLM_PROVIDER = "local"
    else:
        settings.VLM_PROVIDER = provider
        
    doc_id = get_latest_benchmark_doc()
    dataset = load_benchmark_dataset()
    
    # If live and disabled, check
    if provider != "simulated" and not settings.ENABLE_LIVE_VLM_TESTS:
        print(f"Skipping real VLM evaluation because ENABLE_LIVE_VLM_TESTS is False.")
        return None
        
    if limit_live:
        representative_ids = [
            "bob_account_type",           # 1. exact OCR extraction
            "customer_profile",           # 2. general document VQA
            "visual_shape_page3",         # 3. visual-only question
            "chart_page4",                # 4. chart trend
            "vat_rate",                   # 5. table/value question
            "multipage_bob",              # 6. multi-page PDF retrieval
            "summary_info",               # 7. document summary
            "unanswerable_blood_group",   # 8. unanswerable question
            "injection_check"             # 9. prompt injection attempt
        ]
        dataset = [c for c in dataset if c["id"] in representative_ids]
        
        # Add custom document isolation test case
        dataset.append({
            "id": "document_isolation",
            "question": "What is the account status for Alice Smith?",
            "category": "document_isolation",
            "expected_answer": "insufficient evidence",
            "expected_pages": [],
            "answerable": False
        })
        
    generation_results = []
    
    total_latency = 0.0
    latencies = []
    
    exact_value_successes = []
    unanswerable_correctness = []
    hallucination_count = 0
    e2e_successes = []
    answer_successes = []
    
    print(f"{'Question':<45} | {'Retrieval':<9} | {'Gen':<6} | {'E2E':<6} | {'Latency':<7}")
    print("-" * 80)
    
    for case in dataset:
        question = case["question"]
        expected_ans = case["expected_answer"]
        expected_pages = case["expected_pages"]
        is_ans = case["answerable"]
        category = case["category"]
        
        # Rate limit compliance sleep
        if provider != "simulated":
            time.sleep(4.5)
            
        doc_b_id = None
        if category == "document_isolation":
            print("\n[ISOLATION TEST] Setting up document isolation: uploading second document...")
            try:
                import fitz
                doc_b = fitz.open()
                page_b = doc_b.new_page()
                page_b.insert_text((50, 50), "Customer Name: Alice Smith\nAccount Status: Platinum VIP Plan")
                pdf_b_bytes = doc_b.write()
                
                res_b = client.post("/documents/upload", files={"file": ("isolation_doc_b.pdf", pdf_b_bytes, "application/pdf")})
                if res_b.status_code == 200:
                    doc_b_id = res_b.json()["doc_id"]
                    print(f"[ISOLATION TEST] Uploaded isolation document B successfully. ID: {doc_b_id}")
                else:
                    print(f"Warning: Failed to upload isolation document B: {res_b.text}")
            except Exception as e:
                print(f"Warning: Exception setting up isolation document: {e}")
                
        t_start = time.perf_counter()
        res = client.post("/ask", json={"doc_id": doc_id, "question": question})
        t_duration = time.perf_counter() - t_start
        
        # Clean up document B immediately after request
        if doc_b_id is not None:
            try:
                from backend.database import get_db_connection
                from backend.vector_store import VectorStore
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM pages WHERE doc_id = ?", (doc_b_id,))
                    page_ids = [r["id"] for r in cursor.fetchall()]
                    for pid in page_ids:
                        VectorStore.remove_text_vector(pid)
                        VectorStore.remove_image_vector(pid)
                    cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_b_id,))
                    conn.commit()
                VectorStore.save_indices()
                print("[ISOLATION TEST] Cleaned up isolation document B.")
            except Exception as cleanup_err:
                print(f"Warning: Failed to clean up isolation doc B: {cleanup_err}")
                
        if res.status_code != 200:
            print(f"Error querying question '{question}': {res.text}")
            continue
            
        data = res.json()
        
        latencies.append(t_duration)
        total_latency += t_duration
        
        retrieved_pages = data["retrieval"]["pages_considered"]
        ret_success = all(p in retrieved_pages for p in expected_pages) if expected_pages else True
        
        gen_answer = data["answer"]
        gen_answerable = data["answerable"]
        
        gen_success = False
        if not is_ans:
            # Unanswerable expects answerable=False
            gen_success = (gen_answerable is False)
            unanswerable_correctness.append(1 if gen_success else 0)
            if not gen_success:
                hallucination_count += 1
        else:
            # Answerable checks matching answer
            if gen_answerable is True:
                # Direct check
                if expected_ans.lower() in gen_answer.lower():
                    gen_success = True
                else:
                    # Try normalized exact match for numeric/IDs
                    norm_expected = normalize_exact_value(expected_ans)
                    norm_actual = normalize_exact_value(gen_answer)
                    if norm_expected and norm_expected in norm_actual:
                        gen_success = True
            
            if category in ["exact_text", "numeric", "ids", "dates"]:
                exact_value_successes.append(1 if gen_success else 0)
                
        answer_successes.append(1 if gen_success else 0)
        e2e_success = ret_success and gen_success
        e2e_successes.append(1 if e2e_success else 0)
        
        print(
            f"{question[:45]:<45} | "
            f"{'PASS' if ret_success else 'FAIL':<9} | "
            f"{'PASS' if gen_success else 'FAIL':<6} | "
            f"{'PASS' if e2e_success else 'FAIL':<6} | "
            f"{t_duration:.2f}s"
        )
        
        generation_results.append({
            "id": case["id"],
            "question": question,
            "category": category,
            "retrieved_pages": retrieved_pages,
            "expected_pages": expected_pages,
            "ret_success": ret_success,
            "gen_success": gen_success,
            "e2e_success": e2e_success,
            "answer": gen_answer,
            "expected_answer": expected_ans,
            "latency": t_duration
        })
        
    print("=" * 80)
    
    # Calculate performance metrics
    ret_rate = sum(1 for r in generation_results if r["ret_success"]) / len(generation_results)
    gen_rate = sum(answer_successes) / len(answer_successes)
    e2e_rate = sum(e2e_successes) / len(e2e_successes)
    exact_rate = sum(exact_value_successes) / len(exact_value_successes) if exact_value_successes else 1.0
    no_ans_rate = sum(unanswerable_correctness) / len(unanswerable_correctness) if unanswerable_correctness else 1.0
    
    mode_label = "SIMULATED (NOT REAL VLM)" if provider == "simulated" else "REAL VLM INFERENCE"
    
    summary = {
        "evaluation_mode": mode_label,
        "provider": provider,
        "metrics": {
            "retrieval_success_rate": ret_rate,
            "generation_success_rate": gen_rate,
            "e2e_rag_success_rate": e2e_rate,
            "exact_value_success_rate": exact_rate,
            "no_answer_accuracy": no_ans_rate,
            "hallucination_count": hallucination_count
        },
        "performance": {
            "mean_latency": float(np.mean(latencies)),
            "median_latency": float(np.median(latencies)),
            "p95_latency": float(np.percentile(latencies, 95))
        }
    }
    
    print(f"Evaluation Mode: {summary['evaluation_mode']}")
    print(f"Retrieval Success Rate: {summary['metrics']['retrieval_success_rate']:.2%}")
    print(f"Answer Generation Success Rate: {summary['metrics']['generation_success_rate']:.2%}")
    print(f"E2E RAG Success Rate: {summary['metrics']['e2e_rag_success_rate']:.2%}")
    print(f"Exact-Value Success Rate: {summary['metrics']['exact_value_success_rate']:.2%}")
    print(f"No-Answer Accuracy: {summary['metrics']['no_answer_accuracy']:.2%}")
    print(f"Hallucination Count: {summary['metrics']['hallucination_count']}")
    print(f"Mean Latency: {summary['performance']['mean_latency']:.2f}s")
    print("=" * 80 + "\n")
    
    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "generation_results.json"), "w") as f:
        json.dump({"summary": summary, "results": generation_results}, f, indent=2)
        
    # Restore original provider
    settings.VLM_PROVIDER = orig_provider
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VQA generation benchmark evaluation")
    parser.add_argument("--provider", type=str, default="simulated", choices=["simulated", "gemini", "openai"])
    args = parser.parse_args()
    
    with TestClient(app):
        run_vqa_evaluation(args.provider)
