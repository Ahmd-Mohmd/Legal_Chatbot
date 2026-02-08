# -*- coding: utf-8 -*-
"""
RAG Evaluation Script using Ragas Metrics
==========================================
Evaluates the Egyptian Legal Assistant (multi-law RAG) using:
- faithfulness
- answer_relevancy
- context_precision
- context_recall

USAGE:
------
1. Command line: python evaluate_rag.py path/to/questions.json
2. Environment variable: set QA_FILE_PATH=path/to/questions.json
3. Default: Place 'test_dataset_5_questions.json' in same directory

JSON FORMAT:
-----------
List format: [{"question": "...", "ground_truth": "..."}, ...]
OR dict format: {"data": [...]} or {"questions": [...]}

RATE LIMITS:
-----------
- Uses DUAL Groq API keys (GROQ_API_KEY + groq_api) with round-robin rotation
- This effectively doubles the rate limit, allowing shorter delays
- Delays are tuned conservatively to avoid 429 errors
"""

import os
import sys
import json
import time
import itertools
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragas.llms import LangchainLLMWrapper as LangchainLLMWrapperType

# Import the RAG pipeline initialization
from app_final_updated import initialize_rag_pipeline

# Suppress verbose API logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)

load_dotenv()
model_name = "Omartificial-Intelligence-Space/GATE-AraBert-v1"

# ==========================================
# 🔑 DUAL API KEY SETUP (round-robin to double rate limits)
# ==========================================
GROQ_API_KEYS = []
_key1 = os.getenv("GROQ_API_KEY")
_key2 = os.getenv("groq_api")
if _key1:
    GROQ_API_KEYS.append(_key1)
if _key2:
    GROQ_API_KEYS.append(_key2)
if not GROQ_API_KEYS:
    raise RuntimeError("No Groq API keys found in .env (need GROQ_API_KEY and/or groq_api)")

print(f"🔑 Loaded {len(GROQ_API_KEYS)} Groq API key(s) — {'dual-key rotation enabled' if len(GROQ_API_KEYS) > 1 else 'single key mode'}")

# Round-robin key iterator
_key_cycle = itertools.cycle(GROQ_API_KEYS)

def next_api_key() -> str:
    """Get the next API key in round-robin rotation."""
    return next(_key_cycle)

def make_evaluator_llm():
    """Create an evaluator LLM using the next API key in rotation.
    Matches app_final_updated.py: llama-3.3-70b-versatile, temp=0.2, top_p=0.85"""
    return LangchainLLMWrapper(ChatGroq(
        groq_api_key=next_api_key(),
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=2048,
        model_kwargs={"top_p": 0.85},
        max_retries=5,
        request_timeout=120,
    ))
# ==========================================
# ⏱️ RATE LIMITING / DELAYS (tuned for DUAL-KEY rotation)
# ==========================================
# With 2 keys we get ~60 RPM combined, so delays can be shorter.
NUM_KEYS = len(GROQ_API_KEYS)
EFFECTIVE_RPM = 30 * NUM_KEYS          # 30 RPM per key

REQUEST_DELAY_SECONDS  = 30.0 / NUM_KEYS   # 15s with 2 keys (was 60s)
EVALUATION_DELAY_SECONDS = 20.0 / NUM_KEYS  # 10s with 2 keys (was 60s)
INITIAL_COOLDOWN       = 5.0                # 5s after loading pipeline (was 10s)
PER_METRIC_DELAY       = 30.0 / NUM_KEYS    # 15s with 2 keys (was 60s)

# ==========================================
# 📝 TEST DATASET
# ==========================================
# Default test questions (used when no file is provided)
DEFAULT_TEST_QUESTIONS = [
    {
        "question": "ما هي شروط الترشح لرئاسة الجمهورية؟",
        "ground_truth": "يجب أن يكون المرشح مصرياً من أبوين مصريين، وألا تكون له جنسية أخرى، وأن يكون متمتعاً بحقوقه المدنية والسياسية، وأن يكون قد أدى الخدمة العسكرية أو أعفي منها قانوناً، وألا تقل سنه يوم فتح باب الترشح عن أربعين سنة ميلادية."
    },
    {
        "question": "ما هي مدة ولاية رئيس الجمهورية؟",
        "ground_truth": "مدة الرئاسة ست سنوات ميلادية، تبدأ من اليوم التالي لانتهاء مدة سلفه، ولا يجوز إعادة انتخابه إلا لمرة واحدة."
    },
    {
        "question": "ما هي حقوق المواطن في الحصول على المعلومات؟",
        "ground_truth": "المعلومات والبيانات والإحصاءات والوثائق الرسمية ملك للشعب، والإفصاح عنها من مصادرها المختلفة حق تكفله الدولة لكل مواطن."
    },
    {
        "question": "ما هو دور مجلس الشيوخ؟",
        "ground_truth": "يختص مجلس الشيوخ بدراسة واقتراح ما يراه كفيلاً بدعم الوحدة الوطنية والسلام الاجتماعي والحفاظ على المقومات الأساسية للمجتمع، ودراسة مشروعات القوانين المكملة للدستور."
    },
    {
        "question": "كيف يتم تعديل الدستور؟",
        "ground_truth": "لرئيس الجمهورية أو لخمس أعضاء مجلس النواب طلب تعديل مادة أو أكثر من الدستور، ويجب الموافقة على التعديل بأغلبية ثلثي أعضاء المجلس، ثم يعرض على الشعب في استفتاء."
    }
]

def load_test_questions(file_path: str):
    """Load test questions from JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
            if "questions" in obj and isinstance(obj["questions"], list):
                return obj["questions"]
        raise ValueError("Unsupported QA JSON format; expected a list or dict with 'data' or 'questions'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ QA file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Invalid JSON format in {file_path}: {e}")
    except Exception as e:
        raise Exception(f"❌ Error loading QA file {file_path}: {e}")


# Load QA file path from environment variable or command line
qa_file_path = os.getenv("QA_FILE_PATH")
if not qa_file_path and len(sys.argv) > 1:
    qa_file_path = sys.argv[1]

# If still not provided, try default file
if not qa_file_path:
    default_path = "test_dataset_5_questions.json"
    if os.path.exists(default_path):
        qa_file_path = default_path
        print(f"📂 Using default dataset: {default_path}")

if qa_file_path and os.path.exists(qa_file_path):
    print(f"📂 Loading questions from: {qa_file_path}")
    try:
        test_questions = load_test_questions(qa_file_path)
        print(f"✅ Loaded {len(test_questions)} questions from file")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        print("📝 Using default inline test questions instead")
        test_questions = DEFAULT_TEST_QUESTIONS
else:
    if qa_file_path:
        print(f"⚠️ File not found: {qa_file_path}")
    print("📝 Using default inline test questions")
    test_questions = DEFAULT_TEST_QUESTIONS

# ==========================================
# 🔄 RUN EVALUATION
# ==========================================

def run_evaluation():
    print("="*60)
    print("🚀 Starting RAG Evaluation with Ragas")
    print("="*60)
    
    print(f"\n📊 Configuration:")
    print(f"   Questions to evaluate: {len(test_questions)}")
    print(f"   Delay per question (generation): {REQUEST_DELAY_SECONDS}s")
    print(f"   Delay per question (evaluation): {PER_METRIC_DELAY}s")
    
    total_gen_time = len(test_questions) * REQUEST_DELAY_SECONDS / 60.0
    total_eval_time = len(test_questions) * PER_METRIC_DELAY / 60.0
    total_time = total_gen_time + total_eval_time + INITIAL_COOLDOWN / 60.0 + EVALUATION_DELAY_SECONDS / 60.0
    
    print(f"\n⏱️ Estimated total time:")
    print(f"   Question generation: ~{total_gen_time:.1f} minutes")
    print(f"   Evaluation phase: ~{total_eval_time:.1f} minutes")
    print(f"   Total: ~{total_time:.1f} minutes ({total_time/60:.1f} hours)\n")
    
    # 1. Initialize RAG Pipeline
    print("\n📥 Loading RAG pipeline...")
    qa_chain = initialize_rag_pipeline()
    print("✅ Pipeline loaded successfully")

    # Let the service cool down before starting requests
    print(f"⏳ Cooling down for {INITIAL_COOLDOWN} seconds...")
    time.sleep(INITIAL_COOLDOWN)
    
    # 2. Generate answers and collect context
    print("\n🤖 Generating answers for test questions...\n")
    
    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for idx, item in enumerate(test_questions, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(test_questions)}] Generating answer ({idx / len(test_questions) * 100:.0f}% complete)")
        print(f"{'='*60}")
        print(f"Q: {question[:80]}...")
        print(f"{'-'*60}")
        
        try:
            # Invoke the chain
            result = qa_chain.invoke(question)
            
            answer = result["answer"]
            context_docs = result["context"]
            
            # Extract context text from documents
            contexts = [doc.page_content for doc in context_docs]
            
            # Store results
            results["question"].append(question)
            results["answer"].append(answer)
            results["contexts"].append(contexts)
            results["ground_truth"].append(ground_truth)
            
            print(f"✅ Generated answer ({len(answer)} chars)")
            print(f"✅ Retrieved {len(contexts)} context documents")

            # Delay between requests to avoid hitting RPM limits
            if idx < len(test_questions):
                print(f"⏳ Waiting {REQUEST_DELAY_SECONDS} seconds before next question...")
                time.sleep(REQUEST_DELAY_SECONDS)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            # Add placeholder to keep dataset aligned
            results["question"].append(question)
            results["answer"].append("Error generating answer")
            results["contexts"].append([])
            results["ground_truth"].append(ground_truth)
    
    # 3. Convert to Ragas Dataset format
    print("\n📊 Creating evaluation dataset...")
    dataset = Dataset.from_dict(results)
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # 4. Run Ragas Evaluation
    print("\n⚙️ Running Ragas evaluation...")
    print("This may take a few minutes...")
    print(f"Using Groq API (llama-3.3-70b-versatile) with {NUM_KEYS} key(s) for evaluation...")

    # Add a delay before evaluation to avoid back-to-back bursts
    print(f"⏳ Waiting {EVALUATION_DELAY_SECONDS:.0f} seconds before evaluation...")
    time.sleep(EVALUATION_DELAY_SECONDS)
    
    # Configure Groq LLM for evaluation (matches app_final_updated.py)
    evaluator_llm = make_evaluator_llm()
    
    # Configure embeddings (same as app_final_updated.py)
    print("Configuring HuggingFace embeddings (same as app_final_updated.py)...")
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=model_name
    ))
    
    try:
        # Evaluate each question separately with delays to avoid rate limits
        print("\n⚠️ Evaluating each question separately with 60-second delays...")
        print(f"⏱️ Estimated time: ~{len(results['question']) * PER_METRIC_DELAY / 60:.1f} minutes\n")
        
        all_scores = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": []
        }
        
        for q_idx in range(len(results["question"])):
            print(f"\n{'='*60}")
            print(f"📋 Question {q_idx + 1}/{len(results['question'])} ({(q_idx + 1) / len(results['question']) * 100:.0f}% complete)")
            print(f"{'='*60}")
            print(f"Q: {results['question'][q_idx][:80]}...")
            print(f"-" * 60)
            
            # Create single-question dataset
            single_q_data = {
                "question": [results["question"][q_idx]],
                "answer": [results["answer"][q_idx]],
                "contexts": [results["contexts"][q_idx]],
                "ground_truth": [results["ground_truth"][q_idx]]
            }
            single_dataset = Dataset.from_dict(single_q_data)
            
            # Rotate to fresh API key for each question evaluation
            evaluator_llm = make_evaluator_llm()
            
            # Evaluate all metrics for this question
            try:
                q_result = evaluate(
                    single_dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                    raise_exceptions=False
                )
                
                # Convert EvaluationResult to dict if needed
                if hasattr(q_result, 'to_pandas'):
                    # Convert to pandas and then to dict
                    result_df = q_result.to_pandas()
                    result_dict = result_df.to_dict('records')[0] if len(result_df) > 0 else {}
                elif isinstance(q_result, dict):
                    result_dict = q_result
                else:
                    # Try to access as attributes
                    result_dict = {
                        'faithfulness': getattr(q_result, 'faithfulness', 0.0),
                        'answer_relevancy': getattr(q_result, 'answer_relevancy', 0.0),
                        'context_precision': getattr(q_result, 'context_precision', 0.0),
                        'context_recall': getattr(q_result, 'context_recall', 0.0)
                    }
                
                # Extract scores (handle if they're lists or single values)
                def get_score(value):
                    if isinstance(value, list):
                        return value[0] if len(value) > 0 else 0.0
                    return float(value) if value is not None else 0.0
                
                f_score = get_score(result_dict.get('faithfulness', 0.0))
                a_score = get_score(result_dict.get('answer_relevancy', 0.0))
                cp_score = get_score(result_dict.get('context_precision', 0.0))
                cr_score = get_score(result_dict.get('context_recall', 0.0))
                
                # Display scores for this question
                print(f"\n📊 Results for Question {q_idx + 1}:")
                print(f"   Faithfulness       : {f_score:.4f}")
                print(f"   Answer Relevancy   : {a_score:.4f}")
                print(f"   Context Precision  : {cp_score:.4f}")
                print(f"   Context Recall     : {cr_score:.4f}")
                
                all_scores["faithfulness"].append(f_score)
                all_scores["answer_relevancy"].append(a_score)
                all_scores["context_precision"].append(cp_score)
                all_scores["context_recall"].append(cr_score)
                
            except Exception as e:
                print(f"\n❌ Error evaluating question {q_idx + 1}: {str(e)}")
                print(f"   Error type: {type(e).__name__}")
                # Print more debug info if verbose
                import traceback
                print(f"   Traceback: {traceback.format_exc()[:200]}...")
                all_scores["faithfulness"].append(0.0)
                all_scores["answer_relevancy"].append(0.0)
                all_scores["context_precision"].append(0.0)
                all_scores["context_recall"].append(0.0)
            
            # Wait between questions to avoid rate limits
            if q_idx < len(results["question"]) - 1:
                print(f"\n⏳ Waiting {PER_METRIC_DELAY} seconds before next question...")
                time.sleep(PER_METRIC_DELAY)
        
        # Calculate average scores
        print("\n" + "="*60)
        print("📊 CALCULATING AVERAGE SCORES")
        print("="*60)
        
        evaluation_results = {
            "faithfulness": sum(all_scores["faithfulness"]) / len(all_scores["faithfulness"]) if all_scores["faithfulness"] else 0.0,
            "answer_relevancy": sum(all_scores["answer_relevancy"]) / len(all_scores["answer_relevancy"]) if all_scores["answer_relevancy"] else 0.0,
            "context_precision": sum(all_scores["context_precision"]) / len(all_scores["context_precision"]) if all_scores["context_precision"] else 0.0,
            "context_recall": sum(all_scores["context_recall"]) / len(all_scores["context_recall"]) if all_scores["context_recall"] else 0.0
        }
        
        print("\n" + "="*60)
        print("📈 FINAL AVERAGE RESULTS")
        print("="*60)
        
        # Display average results
        for metric_name, score in evaluation_results.items():
            if isinstance(score, (int, float)):
                print(f"  {metric_name:28s}: {score:.4f}")
        
        overall_avg = sum(evaluation_results.values()) / len(evaluation_results)
        print(f"\n  {'Overall Average':28s}: {overall_avg:.4f}")
        
        # Save results to JSON
        results_file = "evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            results_dict = {
                "metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) 
                           for k, v in evaluation_results.items()},
                "individual_scores": all_scores,
                "test_samples": len(dataset),
                "overall_average": overall_avg,
                "evaluation_details": {
                    "delay_per_question": f"{REQUEST_DELAY_SECONDS:.1f}s",
                    "delay_per_metric": f"{PER_METRIC_DELAY:.1f}s",
                    "num_api_keys": NUM_KEYS,
                    "model": "llama-3.3-70b-versatile",
                    "temperature": 0.2,
                    "top_p": 0.85,
                    "embeddings": model_name
                }
            }
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Save individual question breakdown
        breakdown_file = "evaluation_breakdown.json"
        breakdown_data = []
        for q_idx in range(len(results["question"])):
            # Calculate average score for this question across all metrics
            question_score = (
                all_scores["faithfulness"][q_idx] +
                all_scores["answer_relevancy"][q_idx] +
                all_scores["context_precision"][q_idx] +
                all_scores["context_recall"][q_idx]
            ) / 4.0
            
            breakdown_data.append({
                "question": results["question"][q_idx],
                "ground_truth": results["ground_truth"][q_idx],
                "actual_answer": results["answer"][q_idx],
                "score": round(question_score, 4)
            })
        
        # Calculate average score of all questions
        total_avg_score = sum(item["score"] for item in breakdown_data) / len(breakdown_data) if breakdown_data else 0.0
        
        # Create simplified results structure
        simplified_results = {
            "questions": breakdown_data,
            "average_score": round(total_avg_score, 4)
        }
        
        with open(breakdown_file, "w", encoding="utf-8") as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Question breakdown saved to: {breakdown_file}")
        print(f"📊 Average score across all questions: {total_avg_score:.4f}")
        
        # Save detailed results
        detailed_file = "evaluation_detailed.json"
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Detailed results saved to: {detailed_file}")
        
        print("\n" + "="*60)
        print("✅ Evaluation Complete!")
        print("="*60)
        
        return evaluation_results
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print("\n⚠️ Troubleshooting:")
        print("   1. Check GROQ_API_KEY is set in .env file")
        print("   2. Verify you have valid Groq API credits")
        print("   3. Ensure internet connection is stable")
        print("   4. Try increasing PER_METRIC_DELAY in the script")
        print("   5. Reduce the number of test questions")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# 📊 METRIC EXPLANATIONS
# ==========================================

def print_metric_explanations():
    """Print what each metric measures"""
    print("\n" + "="*60)
    print("📖 RAGAS METRICS EXPLANATION")
    print("="*60)
    
    explanations = {
        "faithfulness": "Is the answer grounded in the context? (0-1, higher is better)\n"
                       "Measures if the answer contains only information from the retrieved context.",
        
        "answer_relevancy": "Does the answer relate to the question? (0-1, higher is better)\n"
                           "Measures how well the answer addresses the question asked.",
        
        "context_precision": "How much retrieved context was relevant? (0-1, higher is better)\n"
                            "Measures the signal-to-noise ratio in retrieved documents.",
        
        "context_recall": "Did we retrieve all needed information? (0-1, higher is better)\n"
                         "Measures if all ground truth information is in the context.",
        
        "context_relevancy": "Overall relevance of context to question (0-1, higher is better)\n"
                            "Measures how relevant the retrieved context is to the question."
    }
    
    for metric, explanation in explanations.items():
        print(f"\n{metric.upper()}:")
        print(f"  {explanation}")
    
    print("\n" + "="*60)

# ==========================================
# 🎯 MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    from datetime import datetime
    
    start_time = datetime.now()
    
    print("\n" + "="*60)
    print("🎯 RAG EVALUATION SYSTEM")
    print("   Egyptian Legal Assistant — Multi-Law RAG")
    print(f"   API Keys: {len(GROQ_API_KEYS)} (dual-key rotation {'ON' if len(GROQ_API_KEYS) > 1 else 'OFF'})")
    print("="*60)
    print(f"\n⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print what metrics mean
    print_metric_explanations()
    
    # Run evaluation
    input("\nPress ENTER to start evaluation...")
    
    results = run_evaluation()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"⏰ Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Duration: {duration.total_seconds() / 60:.1f} minutes")
    print(f"📝 Questions evaluated: {len(test_questions)}")
    
    if results:
        print(f"\n✅ Evaluation completed successfully!")
        print(f"\n📂 Output files:")
        print(f"   - evaluation_results.json (average metrics & config)")
        print(f"   - evaluation_breakdown.json (per-question scores)")
        print(f"   - evaluation_detailed.json (full Q&A data)")
    else:
        print(f"\n⚠️ Evaluation could not be completed.")
        print(f"   Check the error messages above for troubleshooting.")
    
    print("\n" + "="*60)
