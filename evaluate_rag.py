# -*- coding: utf-8 -*-
"""
RAG Evaluation Script — Egyptian Legal Assistant
=================================================
Evaluates the multi-law RAG pipeline using Ragas metrics:
  • faithfulness
  • answer_relevancy
  • context_precision
  • context_recall

KEY FEATURES:
  • Uses ALL 4 Groq API keys simultaneously with round-robin rotation
  • Automatic fallback: if a key returns 429/error, retries with next key
  • Default dataset: ragas_dataset_100.csv (100 questions, 6 legal domains)
  • Per-category scoring breakdown in output
  • Conservative delays tuned for 4-key rotation (~120 RPM combined)

USAGE:
  python evaluate_rag.py                         # uses ragas_dataset_100.csv
  python evaluate_rag.py path/to/questions.csv   # custom CSV
  python evaluate_rag.py path/to/questions.json  # custom JSON
  set QA_FILE_PATH=path/to/file; python evaluate_rag.py  # env var
"""

import os
import sys
import json
import csv
import time
import itertools
import traceback
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# Import the RAG pipeline
from app_final_updated import initialize_rag_pipeline, EMBEDDING_MODEL

# Suppress verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# API KEYS — load ALL available keys for maximum throughput
# ═══════════════════════════════════════════════════════════════════
_KEY_NAMES = ["GROQ_API_KEY", "groq_api", "groq_api_2", "groq_api_3"]
GROQ_API_KEYS: List[str] = []
for name in _KEY_NAMES:
    val = os.getenv(name, "").strip().strip('"').strip("'")
    if val:
        GROQ_API_KEYS.append(val)

if not GROQ_API_KEYS:
    raise RuntimeError(f"No Groq API keys found in .env (checked: {_KEY_NAMES})")

NUM_KEYS = len(GROQ_API_KEYS)
print(f"🔑 Loaded {NUM_KEYS} Groq API key(s)")

# Round-robin key cycle
_key_cycle = itertools.cycle(range(NUM_KEYS))
_current_key_idx = 0


def next_api_key() -> str:
    """Get the next API key via round-robin."""
    global _current_key_idx
    _current_key_idx = next(_key_cycle)
    return GROQ_API_KEYS[_current_key_idx]


def make_evaluator_llm() -> LangchainLLMWrapper:
    """Create an evaluator LLM using the next key in rotation."""
    return LangchainLLMWrapper(ChatGroq(
        groq_api_key=next_api_key(),
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=2048,
        model_kwargs={"top_p": 0.85},
        max_retries=3,
        request_timeout=120,
    ))


def make_evaluator_llm_with_key(key: str) -> LangchainLLMWrapper:
    """Create an evaluator LLM with a specific key (for retry fallback)."""
    return LangchainLLMWrapper(ChatGroq(
        groq_api_key=key,
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=2048,
        model_kwargs={"top_p": 0.85},
        max_retries=2,
        request_timeout=120,
    ))


# ═══════════════════════════════════════════════════════════════════
# DELAY TUNING (4 keys × 30 RPM = 120 RPM combined)
# ═══════════════════════════════════════════════════════════════════
EFFECTIVE_RPM = 30 * NUM_KEYS
REQUEST_DELAY: float = max(2.0, 60.0 / EFFECTIVE_RPM)   # ~0.5s min, floored at 2s
PER_METRIC_DELAY: float = max(4.0, 30.0 / NUM_KEYS)     # 7.5s with 4 keys, floored at 4s
INITIAL_COOLDOWN: float = 3.0
EVALUATION_COOLDOWN: float = max(3.0, 15.0 / NUM_KEYS)

# ═══════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════════

def load_csv_dataset(file_path: str) -> List[Dict[str, str]]:
    """Load questions from a CSV with columns: category, question, ground_truth, ..."""
    items: List[Dict[str, str]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "question" in row and "ground_truth" in row:
                items.append({
                    "question": row["question"].strip(),
                    "ground_truth": row["ground_truth"].strip(),
                    "category": row.get("category", "").strip(),
                    "source_law_name": row.get("source_law_name", "").strip(),
                })
    return items


def load_json_dataset(file_path: str) -> List[Dict[str, str]]:
    """Load questions from a JSON file (list or dict with 'data'/'questions')."""
    with open(file_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "questions"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    raise ValueError("Unsupported JSON format")


def load_dataset(file_path: str) -> List[Dict[str, str]]:
    """Auto-detect CSV vs JSON and load accordingly."""
    if file_path.lower().endswith(".csv"):
        return load_csv_dataset(file_path)
    elif file_path.lower().endswith(".json"):
        return load_json_dataset(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


# ── Resolve dataset path ────────────────────────────────────────
qa_file_path = os.getenv("QA_FILE_PATH", "").strip()
if not qa_file_path and len(sys.argv) > 1:
    qa_file_path = sys.argv[1]

# Default to 100-question CSV
if not qa_file_path:
    default_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ragas_dataset_100.csv")
    default_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dataset_5_questions.json")
    if os.path.exists(default_csv):
        qa_file_path = default_csv
    elif os.path.exists(default_json):
        qa_file_path = default_json

if not qa_file_path or not os.path.exists(qa_file_path):
    print(f"❌ Dataset not found: {qa_file_path or '(none)'}")
    sys.exit(1)

print(f"📂 Loading dataset: {qa_file_path}")
test_questions = load_dataset(qa_file_path)
print(f"✅ Loaded {len(test_questions)} questions")

# ═══════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════

def evaluate_single_question(
    single_data: dict,
    evaluator_embeddings,
    max_retries: int = 3,
) -> Dict[str, float]:
    """Evaluate one question using Ragas, with key-rotation retry on failure."""
    single_dataset = Dataset.from_dict({
        "question": [single_data["question"]],
        "answer": [single_data["answer"]],
        "contexts": [single_data["contexts"]],
        "ground_truth": [single_data["ground_truth"]],
    })

    last_error = None
    for attempt in range(max_retries):
        try:
            # Rotate to a different key on each retry
            key_idx = (_current_key_idx + attempt) % NUM_KEYS
            evaluator_llm = make_evaluator_llm_with_key(GROQ_API_KEYS[key_idx])

            q_result = evaluate(
                single_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                raise_exceptions=False,
            )

            # Extract scores
            if hasattr(q_result, "to_pandas"):
                df = q_result.to_pandas()
                row = df.to_dict("records")[0] if len(df) > 0 else {}
            elif isinstance(q_result, dict):
                row = q_result
            else:
                row = {}

            def _score(v):
                if isinstance(v, list):
                    return v[0] if v else 0.0
                return float(v) if v is not None else 0.0

            return {
                "faithfulness": _score(row.get("faithfulness", 0.0)),
                "answer_relevancy": _score(row.get("answer_relevancy", 0.0)),
                "context_precision": _score(row.get("context_precision", 0.0)),
                "context_recall": _score(row.get("context_recall", 0.0)),
            }

        except Exception as e:
            last_error = e
            print(f"   ⚠️  Attempt {attempt + 1}/{max_retries} failed (key #{key_idx}): {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(3)

    print(f"   ❌ All {max_retries} attempts failed: {last_error}")
    return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}


def run_evaluation():
    """Full evaluation pipeline: generate answers → evaluate → save results."""
    print("=" * 60)
    print("🚀 Starting RAG Evaluation")
    print("=" * 60)

    n = len(test_questions)
    est_mins = n * (REQUEST_DELAY + PER_METRIC_DELAY) / 60.0 + 1
    print(f"\n📊 Questions: {n}")
    print(f"🔑 API keys: {NUM_KEYS} (effective RPM: ~{EFFECTIVE_RPM})")
    print(f"⏱️  Est. time: ~{est_mins:.0f} minutes\n")

    # 1. Initialise pipeline
    print("📥 Loading RAG pipeline…")
    qa_chain = initialize_rag_pipeline()
    print("✅ Pipeline loaded")

    time.sleep(INITIAL_COOLDOWN)

    # 2. Generate answers
    print("\n🤖 Generating answers…\n")
    results: Dict[str, list] = {
        "question": [], "answer": [], "contexts": [], "ground_truth": [],
    }
    categories: List[str] = []

    for idx, item in enumerate(test_questions, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        category = item.get("category", "")

        print(f"[{idx}/{n}] ({idx / n * 100:.0f}%) Q: {question[:70]}…")

        try:
            result = qa_chain.invoke(question)
            answer = result["answer"]
            contexts = [doc.page_content for doc in result["context"]]
            print(f"   ✅ answer={len(answer)} chars, context={len(contexts)} docs")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            answer = "Error generating answer"
            contexts = []

        results["question"].append(question)
        results["answer"].append(answer)
        results["contexts"].append(contexts)
        results["ground_truth"].append(ground_truth)
        categories.append(category)

        if idx < n:
            time.sleep(REQUEST_DELAY)

    # 3. Evaluate with Ragas
    print(f"\n⏳ Cooling down {EVALUATION_COOLDOWN:.0f}s before evaluation…")
    time.sleep(EVALUATION_COOLDOWN)

    print("\n📊 Running Ragas evaluation…")
    print(f"   Using {NUM_KEYS} keys with rotation & retry\n")

    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    all_scores: Dict[str, List[float]] = {
        "faithfulness": [], "answer_relevancy": [], "context_precision": [], "context_recall": [],
    }

    for q_idx in range(n):
        print(f"\n[Eval {q_idx + 1}/{n}] ({(q_idx + 1) / n * 100:.0f}%) {results['question'][q_idx][:60]}…")

        scores = evaluate_single_question(
            {
                "question": results["question"][q_idx],
                "answer": results["answer"][q_idx],
                "contexts": results["contexts"][q_idx],
                "ground_truth": results["ground_truth"][q_idx],
            },
            evaluator_embeddings,
        )

        print(f"   F={scores['faithfulness']:.3f}  AR={scores['answer_relevancy']:.3f}  "
              f"CP={scores['context_precision']:.3f}  CR={scores['context_recall']:.3f}")

        for k in all_scores:
            all_scores[k].append(scores[k])

        if q_idx < n - 1:
            time.sleep(PER_METRIC_DELAY)

    # 4. Calculate averages
    avg_scores = {k: (sum(v) / len(v) if v else 0.0) for k, v in all_scores.items()}
    overall_avg = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0

    # Per-category breakdown
    cat_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for q_idx, cat in enumerate(categories):
        if not cat:
            cat = "غير مصنف"
        for metric in all_scores:
            cat_scores[cat][metric].append(all_scores[metric][q_idx])

    per_category = {}
    for cat, metrics in cat_scores.items():
        per_category[cat] = {
            m: round(sum(v) / len(v), 4) if v else 0.0
            for m, v in metrics.items()
        }
        per_category[cat]["count"] = len(next(iter(metrics.values())))

    # 5. Display results
    print("\n" + "=" * 60)
    print("📈 FINAL AVERAGE RESULTS")
    print("=" * 60)
    for m, s in avg_scores.items():
        print(f"  {m:28s}: {s:.4f}")
    print(f"\n  {'Overall Average':28s}: {overall_avg:.4f}")

    if per_category:
        print("\n📊 PER-CATEGORY BREAKDOWN:")
        for cat, scores in sorted(per_category.items()):
            cat_avg = sum(v for k, v in scores.items() if k != "count") / 4.0
            print(f"  {cat} (n={scores['count']}): avg={cat_avg:.4f}")

    # 6. Save outputs
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # evaluation_results.json — summary
    results_file = os.path.join(base_dir, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {k: round(v, 4) for k, v in avg_scores.items()},
            "overall_average": round(overall_avg, 4),
            "individual_scores": {k: [round(x, 4) for x in v] for k, v in all_scores.items()},
            "per_category": per_category,
            "test_samples": n,
            "config": {
                "num_api_keys": NUM_KEYS,
                "model": "llama-3.3-70b-versatile",
                "embedding": EMBEDDING_MODEL,
                "request_delay": REQUEST_DELAY,
                "per_metric_delay": PER_METRIC_DELAY,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {results_file}")

    # evaluation_breakdown.json — per-question
    breakdown_file = os.path.join(base_dir, "evaluation_breakdown.json")
    breakdown = []
    for q_idx in range(n):
        q_score = sum(all_scores[m][q_idx] for m in all_scores) / 4.0
        breakdown.append({
            "question": results["question"][q_idx],
            "ground_truth": results["ground_truth"][q_idx],
            "actual_answer": results["answer"][q_idx],
            "category": categories[q_idx],
            "score": round(q_score, 4),
            "faithfulness": round(all_scores["faithfulness"][q_idx], 4),
            "answer_relevancy": round(all_scores["answer_relevancy"][q_idx], 4),
            "context_precision": round(all_scores["context_precision"][q_idx], 4),
            "context_recall": round(all_scores["context_recall"][q_idx], 4),
        })
    with open(breakdown_file, "w", encoding="utf-8") as f:
        json.dump({"questions": breakdown, "average_score": round(overall_avg, 4)}, f, ensure_ascii=False, indent=2)
    print(f"💾 {breakdown_file}")

    # evaluation_detailed.json — raw data
    detailed_file = os.path.join(base_dir, "evaluation_detailed.json")
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 {detailed_file}")

    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)
    return avg_scores


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start = datetime.now()
    print(f"\n⏰ Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔑 Keys: {NUM_KEYS} | Dataset: {len(test_questions)} questions\n")

    results = run_evaluation()

    end = datetime.now()
    duration = (end - start).total_seconds()
    print(f"\n⏰ Finished: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duration: {duration / 60:.1f} min")

    if results:
        print("✅ Success — check output files.")
    else:
        print("⚠️  Evaluation failed.")
