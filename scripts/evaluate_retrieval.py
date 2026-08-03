import os
import sys
import json
import time
import numpy as np
import fitz
from PIL import Image, ImageDraw
import io

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app
from backend.config import settings
from backend.retrieval import retrieve_evidence
from backend.database import get_db_connection
from backend.vector_store import VectorStore

# Set up test client
client = TestClient(app)

# Helper function to generate shape drawings for visual VQA
def draw_visual_shape(color: str, shape: str) -> bytes:
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse((40, 40, 184, 184), fill=color, outline=color)
    elif shape == "square":
        draw.rectangle((40, 40, 184, 184), fill=color, outline=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# Helper function to generate a simple trend line chart
def draw_trend_chart(trend: str) -> bytes:
    img = Image.new("RGB", (300, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line((40, 160, 260, 160), fill=(0, 0, 0), width=2) # X Axis
    draw.line((40, 20, 40, 160), fill=(0, 0, 0), width=2)  # Y Axis
    if trend == "upward":
        draw.line((50, 140, 250, 40), fill=(255, 0, 0), width=4)
    else:
        draw.line((50, 40, 250, 140), fill=(0, 0, 255), width=4)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def build_benchmark_document() -> str:
    """Builds and uploads the 10-page benchmark document, returning the doc_id."""
    print("Building synthetic benchmark PDF document...")
    doc = fitz.open()
    
    # Page 1: Profile & Account Type
    p1 = doc.new_page(width=500, height=300)
    p1.insert_text((20, 40), "Customer Profile: Bob Cooper", fontsize=14)
    p1.insert_text((20, 80), "Account Type: Enterprise Membership", fontsize=14)
    
    # Page 2: Balance, Due Date, VAT
    p2 = doc.new_page(width=500, height=300)
    p2.insert_text((20, 40), "Total Balance Due: USD 24,500.00", fontsize=14)
    p2.insert_text((20, 80), "Due Date: 2026-10-15", fontsize=14)
    p2.insert_text((20, 120), "VAT: 18%", fontsize=14)
    
    # Page 3: Yellow circle drawing (image only)
    p3 = doc.new_page(width=224, height=224)
    p3.insert_image(p3.rect, stream=draw_visual_shape("yellow", "circle"))
    
    # Page 4: Upward trend line chart (image only)
    p4 = doc.new_page(width=300, height=200)
    p4.insert_image(p4.rect, stream=draw_trend_chart("upward"))
    
    # Page 5: Contact details
    p5 = doc.new_page(width=500, height=300)
    p5.insert_text((20, 40), "Contact Details: support@enterprise.com", fontsize=14)
    
    # Page 6: Contract terms & Signee
    p6 = doc.new_page(width=500, height=300)
    p6.insert_text((20, 40), "Contract Term: 24 Months", fontsize=14)
    p6.insert_text((20, 80), "Signee Name: Robert Smith", fontsize=14)
    p6.insert_text((20, 120), "Date Signed: 2026-10-15", fontsize=14)
    
    # Page 7: Black square drawing (image only)
    p7 = doc.new_page(width=224, height=224)
    p7.insert_image(p7.rect, stream=draw_visual_shape("black", "square"))
    
    # Page 8: Invoice number, Reference ID, Service fee
    p8 = doc.new_page(width=500, height=300)
    p8.insert_text((20, 40), "Invoice Number: INV-2026-X12", fontsize=14)
    p8.insert_text((20, 80), "Reference ID: INV-2026-X12", fontsize=14)
    p8.insert_text((20, 120), "Service Fee: 300 rupees", fontsize=14)
    
    # Page 9: Downward trend line chart (image only)
    p9 = doc.new_page(width=300, height=200)
    p9.insert_image(p9.rect, stream=draw_trend_chart("downward"))
    
    # Page 10: Payment terms
    p10 = doc.new_page(width=500, height=300)
    p10.insert_text((20, 40), "Payment Terms: Net 30 Days", fontsize=14)
    
    pdf_bytes = doc.write()
    response = client.post("/documents/upload", files={"file": ("benchmark_eval.pdf", pdf_bytes, "application/pdf")})
    if response.status_code != 200:
        raise RuntimeError(f"Failed to upload document: {response.text}")
    doc_id = response.json()["doc_id"]
    print(f"Document uploaded successfully. ID: {doc_id}")
    return doc_id

def load_benchmark_dataset():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "tests", "fixtures", "evaluation", "benchmark_dataset.json")
    with open(path, "r") as f:
        return json.load(f)

def run_evaluation(doc_id, dataset, top_k=3, text_w=0.65, image_w=0.35, 
                   fusion_method="normalized", query_aware=False, rrf_constant=60):
    # Save original settings
    orig_top_k = settings.RETRIEVAL_TOP_K
    orig_text_w = settings.TEXT_RETRIEVAL_WEIGHT
    orig_image_w = settings.IMAGE_RETRIEVAL_WEIGHT
    orig_fusion = settings.FUSION_METHOD
    orig_query_aware = settings.QUERY_AWARE_FUSION
    orig_rrf_k = settings.RRF_CONSTANT
    
    # Apply experiment parameters
    settings.RETRIEVAL_TOP_K = top_k
    settings.TEXT_RETRIEVAL_WEIGHT = text_w
    settings.IMAGE_RETRIEVAL_WEIGHT = image_w
    settings.FUSION_METHOD = fusion_method
    settings.QUERY_AWARE_FUSION = query_aware
    settings.RRF_CONSTANT = rrf_constant
    
    recalls_1 = []
    recalls_3 = []
    recalls_5 = []
    rr_scores = []
    
    latencies = []
    failures = []
    category_metrics = {}
    
    # Filter answerable questions for standard retrieval metrics
    retrieval_cases = [case for case in dataset if case["answerable"]]
    
    for case in retrieval_cases:
        question = case["question"]
        expected_pages = case["expected_pages"]
        category = case["category"]
        
        if category not in category_metrics:
            category_metrics[category] = {"total": 0, "recall1": 0, "recall3": 0}
        category_metrics[category]["total"] += 1
        
        t_start = time.perf_counter()
        results = retrieve_evidence(doc_id, question, top_k=top_k)
        t_duration = time.perf_counter() - t_start
        latencies.append(t_duration)
        
        page_ranks = [r["page_number"] for r in results]
        
        # Recall@1
        r1 = 1 if expected_pages and page_ranks and page_ranks[0] in expected_pages else 0
        recalls_1.append(r1)
        category_metrics[category]["recall1"] += r1
        
        # Recall@3
        r3 = 1 if all(ep in page_ranks[:3] for ep in expected_pages) else 0
        recalls_3.append(r3)
        category_metrics[category]["recall3"] += r3
        
        # Recall@5
        r5 = 1 if all(ep in page_ranks[:5] for ep in expected_pages) else 0
        recalls_5.append(r5)
        
        # Reciprocal Rank (MRR) - target first expected page
        if expected_pages:
            primary_expected = expected_pages[0]
            if primary_expected in page_ranks:
                rank = page_ranks.index(primary_expected) + 1
                rr_scores.append(1.0 / rank)
            else:
                rr_scores.append(0.0)
        
        # Record failure if Recall@1 is 0
        if not r1:
            # Classify failure cause
            # Find scores in results
            retrieved_pages_info = [{"page": r["page_number"], "scores": r["scores"]} for r in results]
            failures.append({
                "question": question,
                "expected": expected_pages,
                "retrieved": page_ranks,
                "category": category,
                "retrieved_info": retrieved_pages_info
            })
            
    # Restore original settings
    settings.RETRIEVAL_TOP_K = orig_top_k
    settings.TEXT_RETRIEVAL_WEIGHT = orig_text_w
    settings.IMAGE_RETRIEVAL_WEIGHT = orig_image_w
    settings.FUSION_METHOD = orig_fusion
    settings.QUERY_AWARE_FUSION = orig_query_aware
    settings.RRF_CONSTANT = orig_rrf_k
    
    return {
        "Recall@1": float(np.mean(recalls_1)) if recalls_1 else 0.0,
        "Recall@3": float(np.mean(recalls_3)) if recalls_3 else 0.0,
        "Recall@5": float(np.mean(recalls_5)) if recalls_5 else 0.0,
        "MRR": float(np.mean(rr_scores)) if rr_scores else 0.0,
        "latencies": latencies,
        "failures": failures,
        "category_metrics": category_metrics
    }

def check_score_compatibility(doc_id):
    """Measures the statistics of text and image score distributions."""
    print("Measuring score compatibility...")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, ocr_text FROM pages WHERE doc_id = ?", (doc_id,))
        rows = cursor.fetchall()
        
    pids = [r["id"] for r in rows]
    
    # Generate some query vectors and look at raw scores
    questions = [
        "account type for Bob Cooper",
        "total balance due",
        "red circle drawing",
        "downward trend chart",
        "payment terms"
    ]
    
    all_text_scores = []
    all_image_scores = []
    
    for q in questions:
        text_vec = VectorStore.search_text_index(VectorStore.validate_vector(VectorStore._text_index.reconstruct(pids[0]), VectorStore._text_dim), len(pids)) # placeholder vector extract
        # Actually generate using embedders
        from backend.embeddings.text_embedder import TextEmbedder
        from backend.embeddings.image_embedder import ImageEmbedder
        
        t_vec = TextEmbedder.embed_text(q)
        i_vec = ImageEmbedder.embed_text(q)
        
        if t_vec is not None:
            t_res = VectorStore.search_text_index(t_vec, len(pids))
            all_text_scores.extend([score for _, score in t_res])
        if i_vec is not None:
            i_res = VectorStore.search_image_index(i_vec, len(pids))
            all_image_scores.extend([score for _, score in i_res])
            
    stats = {
        "text": {
            "min": float(np.min(all_text_scores)) if all_text_scores else 0.0,
            "max": float(np.max(all_text_scores)) if all_text_scores else 0.0,
            "mean": float(np.mean(all_text_scores)) if all_text_scores else 0.0,
            "median": float(np.median(all_text_scores)) if all_text_scores else 0.0,
            "std": float(np.std(all_text_scores)) if all_text_scores else 0.0
        },
        "image": {
            "min": float(np.min(all_image_scores)) if all_image_scores else 0.0,
            "max": float(np.max(all_image_scores)) if all_image_scores else 0.0,
            "mean": float(np.mean(all_image_scores)) if all_image_scores else 0.0,
            "median": float(np.median(all_image_scores)) if all_image_scores else 0.0,
            "std": float(np.std(all_image_scores)) if all_image_scores else 0.0
        }
    }
    print("Score Distributions:")
    print(f"Text raw scores  -> min: {stats['text']['min']:.4f}, max: {stats['text']['max']:.4f}, mean: {stats['text']['mean']:.4f}, std: {stats['text']['std']:.4f}")
    print(f"Image raw scores -> min: {stats['image']['min']:.4f}, max: {stats['image']['max']:.4f}, mean: {stats['image']['mean']:.4f}, std: {stats['image']['std']:.4f}")
    return stats

def main():
    # Load dataset
    dataset = load_benchmark_dataset()
    
    # 1. Build document and get doc_id
    doc_id = build_benchmark_document()
    
    # 2. Check compatibility of score distribution
    score_stats = check_score_compatibility(doc_id)
    
    # 3. RUN EXPERIMENTS
    print("\n" + "=" * 60)
    print("         RUNNING RETRIEVAL EXPERIMENTS")
    print("=" * 60)
    
    # 3.1 Text-only baseline
    text_only = run_evaluation(doc_id, dataset, text_w=1.0, image_w=0.0, fusion_method="weighted")
    print(f"Text-Only Baseline: Recall@1={text_only['Recall@1']:.2%}, Recall@3={text_only['Recall@3']:.2%}, MRR={text_only['MRR']:.4f}")
    
    # 3.2 Image-only baseline
    image_only = run_evaluation(doc_id, dataset, text_w=0.0, image_w=1.0, fusion_method="weighted")
    print(f"Image-Only Baseline: Recall@1={image_only['Recall@1']:.2%}, Recall@3={image_only['Recall@3']:.2%}, MRR={image_only['MRR']:.4f}")
    
    # 3.3 Current Hybrid (Normalized score fusion with 0.65/0.35 weights)
    hybrid_baseline = run_evaluation(doc_id, dataset, text_w=0.65, image_w=0.35, fusion_method="normalized")
    print(f"Hybrid Baseline (0.65/0.35 Normalized): Recall@1={hybrid_baseline['Recall@1']:.2%}, Recall@3={hybrid_baseline['Recall@3']:.2%}, MRR={hybrid_baseline['MRR']:.4f}")
    
    # 3.4 Grid Search for Raw Weights (weighted fusion)
    print("\nGrid Search over Raw Weights (Weighted score fusion):")
    best_weighted = {"Recall@1": 0.0, "weights": (0, 0)}
    for i in range(11):
        tw = round(i * 0.1, 1)
        iw = round(1.0 - tw, 1)
        res = run_evaluation(doc_id, dataset, text_w=tw, image_w=iw, fusion_method="weighted")
        print(f"  Text {tw:.1f} | Image {iw:.1f} -> Recall@1={res['Recall@1']:.2%}, Recall@3={res['Recall@3']:.2%}, MRR={res['MRR']:.4f}")
        if res["Recall@1"] > best_weighted["Recall@1"]:
            best_weighted = {"Recall@1": res["Recall@1"], "Recall@3": res["Recall@3"], "MRR": res["MRR"], "weights": (tw, iw)}
            
    # 3.5 Grid Search for Normalized Weights
    print("\nGrid Search over Normalized Weights:")
    best_norm = {"Recall@1": 0.0, "weights": (0, 0)}
    for i in range(11):
        tw = round(i * 0.1, 1)
        iw = round(1.0 - tw, 1)
        res = run_evaluation(doc_id, dataset, text_w=tw, image_w=iw, fusion_method="normalized")
        print(f"  Text {tw:.1f} | Image {iw:.1f} -> Recall@1={res['Recall@1']:.2%}, Recall@3={res['Recall@3']:.2%}, MRR={res['MRR']:.4f}")
        if res["Recall@1"] > best_norm["Recall@1"]:
            best_norm = {"Recall@1": res["Recall@1"], "Recall@3": res["Recall@3"], "MRR": res["MRR"], "weights": (tw, iw)}

    # 3.6 Reciprocal Rank Fusion (RRF) Experiment
    print("\nReciprocal Rank Fusion (RRF) Experiment:")
    for k in [10, 20, 50, 60, 100]:
        res = run_evaluation(doc_id, dataset, fusion_method="rrf", rrf_constant=k)
        print(f"  RRF constant k={k} -> Recall@1={res['Recall@1']:.2%}, Recall@3={res['Recall@3']:.2%}, MRR={res['MRR']:.4f}")

    # 3.7 Query-Aware Fusion Experiment
    query_aware = run_evaluation(doc_id, dataset, text_w=0.65, image_w=0.35, fusion_method="normalized", query_aware=True)
    print(f"\nQuery-Aware Hybrid: Recall@1={query_aware['Recall@1']:.2%}, Recall@3={query_aware['Recall@3']:.2%}, MRR={query_aware['MRR']:.4f}")

    # 3.8 Top-K Experiments
    print("\nTop-K candidate evaluation (Normalized weight best):")
    for k in [1, 3, 5]:
        res = run_evaluation(doc_id, dataset, top_k=k, text_w=best_norm["weights"][0], image_w=best_norm["weights"][1], fusion_method="normalized")
        print(f"  k={k} -> Recall@1={res['Recall@1']:.2%}, Recall@3={res['Recall@3']:.2%}, MRR={res['MRR']:.4f}")

    # 4. Save optimized metrics reports
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    
    optimized_config_data = {
        "score_statistics": score_stats,
        "text_only": {"Recall@1": text_only["Recall@1"], "Recall@3": text_only["Recall@3"], "MRR": text_only["MRR"]},
        "image_only": {"Recall@1": image_only["Recall@1"], "Recall@3": image_only["Recall@3"], "MRR": image_only["MRR"]},
        "hybrid_baseline": {"Recall@1": hybrid_baseline["Recall@1"], "Recall@3": hybrid_baseline["Recall@3"], "MRR": hybrid_baseline["MRR"]},
        "best_normalized_fusion": {
            "text_weight": best_norm["weights"][0],
            "image_weight": best_norm["weights"][1],
            "Recall@1": best_norm["Recall@1"],
            "Recall@3": best_norm["Recall@3"],
            "MRR": best_norm["MRR"]
        },
        "query_aware_fusion": {"Recall@1": query_aware["Recall@1"], "Recall@3": query_aware["Recall@3"], "MRR": query_aware["MRR"]},
        "failures": query_aware["failures"]
    }
    
    with open(os.path.join(out_dir, "retrieval_optimized.json"), "w") as f:
        json.dump(optimized_config_data, f, indent=2)
    print(f"\nSaved machine-readable results to {out_dir}/retrieval_optimized.json")

if __name__ == "__main__":
    with TestClient(app):
        main()
