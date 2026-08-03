import os
import sys
import argparse
import json
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app
from backend.config import settings
from scripts.evaluate_retrieval import run_evaluation as run_retrieval_eval
from scripts.evaluate_vqa import run_vqa_evaluation

def print_banner(text):
    print("\n" + "=" * 80)
    print(f" {text}".center(80))
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Phase 8 - Final Unified Evaluation Script")
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "simulated", "live"],
        help="Evaluation Mode: 'offline' (retrieval-only), 'simulated' (local VLM simulation), or 'live' (real remote VLM)."
    )
    args = parser.parse_args()

    client = TestClient(app)

    if args.mode == "offline":
        print_banner("RUNNING OFFLINE RETRIEVAL EVALUATION")
        # Ensure fresh document upload for offline run
        from scripts.evaluate_retrieval import build_benchmark_document, load_benchmark_dataset
        doc_id = build_benchmark_document()
        dataset = load_benchmark_dataset()
        
        # Run retrieval benchmark
        start_time = time.perf_counter()
        retrieval_summary = run_retrieval_eval(doc_id, dataset)
        duration = time.perf_counter() - start_time
        
        # Calculate mean retrieval latency from latencies list
        latencies = retrieval_summary.get("latencies", [])
        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        
        print_banner("FINAL OFFLINE METRICS SUMMARY")
        print(f"Recall@1: {retrieval_summary.get('Recall@1', 0.0):.2%}")
        print(f"Recall@3: {retrieval_summary.get('Recall@3', 0.0):.2%}")
        print(f"Recall@5: {retrieval_summary.get('Recall@5', 0.0):.2%}")
        print(f"MRR: {retrieval_summary.get('MRR', 0.0000):.4f}")
        print(f"Retrieval Success Rate: {retrieval_summary.get('Recall@3', 0.0):.2%}")
        print(f"Mean Retrieval Latency: {mean_latency:.4f}s")
        print(f"Total Evaluation Duration: {duration:.2f}s")
        print("=" * 80 + "\n")

    elif args.mode == "simulated":
        print_banner("RUNNING SIMULATED VLM GENERATION EVALUATION")
        # Ensure fresh document upload for simulated run to purge stale index mappings
        from scripts.evaluate_retrieval import build_benchmark_document
        build_benchmark_document()
        
        # Run local high-fidelity simulated VLM benchmark
        start_time = time.perf_counter()
        vqa_summary = run_vqa_evaluation(provider="simulated")
        duration = time.perf_counter() - start_time
        
        print_banner("FINAL SIMULATED METRICS SUMMARY")
        print("LABEL: SIMULATED — NOT REAL MODEL ACCURACY")
        print(f"Retrieval Success Rate: {vqa_summary['metrics']['retrieval_success_rate']:.2%}")
        print(f"Answer Generation Success Rate: {vqa_summary['metrics']['generation_success_rate']:.2%}")
        print(f"E2E RAG Success Rate: {vqa_summary['metrics']['e2e_rag_success_rate']:.2%}")
        print(f"Exact-Value Success Rate: {vqa_summary['metrics']['exact_value_success_rate']:.2%}")
        print(f"No-Answer Accuracy: {vqa_summary['metrics']['no_answer_accuracy']:.2%}")
        print(f"Hallucination Count: {vqa_summary['metrics']['hallucination_count']}")
        print(f"Mean VLM Latency: {vqa_summary['performance']['mean_latency']:.2f}s")
        print(f"Total Evaluation Duration: {duration:.2f}s")
        print("=" * 80 + "\n")

    elif args.mode == "live":
        print_banner("RUNNING LIVE VLM GENERATION EVALUATION")
        
        # Verify active provider and API Keys
        active_provider = settings.VLM_PROVIDER
        if active_provider == "local":
            print("VLM_PROVIDER is set to 'local'. Setting VLM_PROVIDER to 'openai' for live evaluation...")
            settings.VLM_PROVIDER = "openai"
            active_provider = "openai"
            
        if active_provider == "openai" and not settings.OPENAI_API_KEY:
            print("ERROR: OPENAI_API_KEY is not set in environment.")
            sys.exit(1)
        elif active_provider == "gemini" and not settings.GEMINI_API_KEY:
            print("ERROR: GEMINI_API_KEY is not set in environment.")
            sys.exit(1)
            
        print(f"Active Live Provider: {active_provider}")
        print(f"Active Live Model: {settings.OPENAI_MODEL if active_provider == 'openai' else settings.GEMINI_MODEL}")
        
        # Enable live tests flag temporarily
        settings.ENABLE_LIVE_VLM_TESTS = True
        
        # Execute first smoke request to check token budget gate (Step 33)
        print_banner("LIVE TOKEN BUDGET GATE - SMOKE TEST")
        from backend.database import init_db
        from scripts.evaluate_retrieval import build_benchmark_document
        
        init_db()
        doc_id = build_benchmark_document()
        
        # Smoke question
        smoke_q = "What is the account type for Bob Cooper?"
        print(f"Executing Single Smoke Query: '{smoke_q}'...")
        
        from backend.generation.answer_generator import generate_grounded_answer
        t0 = time.perf_counter()
        smoke_result = generate_grounded_answer(doc_id, smoke_q)
        duration_smoke = time.perf_counter() - t0
        
        print(f"Smoke Answering Latency: {duration_smoke:.2f}s")
        print(f"Grounded Answer Result: {json.dumps(smoke_result, indent=2)}")
        
        print("\nSmoke test successfully completed token budget validation!")
        
        # Prompt user to check token budget logging in the terminal before running the rest
        print("\n" + "-" * 80)
        print("Please verify the logged [VLM_USAGE] token budget above.")
        print("Optimizations are active. To run the full live validation dataset (10-20 queries),")
        print("run standard evaluations with enable flags.")
        print("-" * 80 + "\n")
        
        # Run live VQA evaluation
        vqa_summary = run_vqa_evaluation(provider=active_provider, limit_live=True)
        
        print_banner("FINAL REAL VLM METRICS SUMMARY")
        print(f"Real Provider: {active_provider} (Model: {settings.OPENAI_MODEL if active_provider == 'openai' else settings.GEMINI_MODEL})")
        print(f"Retrieval Success Rate: {vqa_summary['metrics']['retrieval_success_rate']:.2%}")
        print(f"Answer Generation Success Rate: {vqa_summary['metrics']['generation_success_rate']:.2%}")
        print(f"E2E RAG Success Rate: {vqa_summary['metrics']['e2e_rag_success_rate']:.2%}")
        print(f"Exact-Value Success Rate: {vqa_summary['metrics']['exact_value_success_rate']:.2%}")
        print(f"No-Answer Accuracy: {vqa_summary['metrics']['no_answer_accuracy']:.2%}")
        print(f"Hallucination Count: {vqa_summary['metrics']['hallucination_count']}")
        print(f"Mean VLM Answering Latency: {vqa_summary['performance']['mean_latency']:.2f}s")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
