"""
beam_ingest.py — BEAM Benchmark Ingest + Evaluation
=====================================================

Uses /build_ingest_data/ for ingestion, /multihop/ for chat retrieval,
and /compose/search_slots/ for QA ground truth.

Run:
    python beam_ingest.py --chat-dir test_chats/100K/1 --chat-id 100K_1
    python beam_ingest.py --chat-dir test_chats/100K/1 --chat-id 100K_1 --wipe
    python beam_ingest.py --chat-dir test_chats/100K/1 --chat-id 100K_1 --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_URL = os.getenv("HB_SERVER_URL", os.getenv("SERVER_URL", "http://localhost:8000"))
API_KEY    = os.getenv("HB_API_KEY",    os.getenv("API_KEY", ""))
DB_NAME    = os.getenv("HB_DB_NAME",    "beam_db")

DIM = 1024  # Supports phase_dim=512

REQUEST_DELAY = 0.5
TIMEOUT = 300
MAX_RETRIES = 3

# ── Schemas ────────────────────────────────────────────────────────────────────

CHAT_SCHEMA = {
    "molecule": "Row",
    "primary_key": {"name": "turn_id", "encoding": "exact"},
    "fields": {
        "turn_id": {"name": "turn_id", "encoding": "exact"},
        "chat_id": {"name": "chat_id", "encoding": "exact"},
        "role": {"name": "role", "encoding": "exact"},
        "content": {"name": "content", "encoding": "semantic"},
        "time_anchor": {"name": "time_anchor", "encoding": "temporal"},
    },
    "field_order": ["turn_id", "chat_id", "role", "content", "time_anchor"],
}

QA_SCHEMA = {
    "molecule": "Row",
    "primary_key": {"name": "qa_id", "encoding": "exact"},
    "fields": {
        "qa_id": {"name": "qa_id", "encoding": "exact"},
        "chat_id": {"name": "chat_id", "encoding": "exact"},
        "category": {"name": "category", "encoding": "exact"},
        "difficulty": {"name": "difficulty", "encoding": "exact"},
        "plan_ref": {"name": "plan_ref", "encoding": "exact"},
        "question": {"name": "question", "encoding": "semantic"},
        "ideal_answer": {"name": "ideal_answer", "encoding": "exact"},
        "rubric": {"name": "rubric", "encoding": "exact"},
    },
    "field_order": [
        "qa_id", "chat_id", "category", "difficulty",
        "plan_ref", "question", "ideal_answer", "rubric"
    ],
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def chat_namespace(chat_id: str) -> str:
    return f"beam_chat_{chat_id.replace('/', '_')}"

def qa_namespace(chat_id: str) -> str:
    return f"beam_qa_{chat_id.replace('/', '_')}"

def retry_request(func):
    """Decorator to retry HTTP requests."""
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 10 * (attempt + 1)
                    print(f"      ⚠ Retry {attempt+1}/{MAX_RETRIES} in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise
    return wrapper

# ── Load turns from chat.json ────────────────────────────────────────────────

def load_turns(chat_file: Path) -> list[dict]:
    """Extract all turns with 'role' and 'content' from chat.json."""
    with open(chat_file, encoding="utf-8") as f:
        data = json.load(f)
    
    turns = []
    
    def extract(obj):
        if isinstance(obj, dict):
            if "role" in obj and "content" in obj:
                turns.append(obj)
            else:
                for value in obj.values():
                    extract(value)
        elif isinstance(obj, list):
            for item in obj:
                extract(item)
    
    extract(data)
    print(f"  ✅ Loaded {len(turns)} turns")
    return turns

def prepare_chat_rows(chat_id: str, turns: list[dict]) -> list[dict]:
    """Prepare turns as rows for ingestion."""
    rows = []
    for turn in turns:
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "").strip()
        tid = turn.get("id", "?")
        time_anchor = turn.get("time_anchor", "")
        
        if not content:
            continue
        
        rows.append({
            "turn_id": f"{chat_id}_turn_{tid}",
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "time_anchor": time_anchor,
        })
    
    return rows

# ── Ingest chat ───────────────────────────────────────────────────────────────

@retry_request
def ingest_chat(chat_id: str, turns: list[dict]) -> Dict[str, Any]:
    """Ingest chat turns into Hyperbinder using /build_ingest_data/."""
    namespace = chat_namespace(chat_id)
    rows = prepare_chat_rows(chat_id, turns)
    
    if not rows:
        raise ValueError("No rows to ingest")
    
    df = pd.DataFrame(rows)
    url = f"{SERVER_URL}/build_ingest_data/"
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    
    print(f"  📤 Uploading {len(df)} chat turns → namespace: {namespace}")
    print(f"  📊 Sample first row: {df.iloc[0].to_dict()}")
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        df.to_csv(tmp, index=False)
        tmp_path = tmp.name
    
    try:
        with open(tmp_path, "rb") as f:
            response = requests.post(
                url,
                headers=headers,
                files={"file": (f"chat_{chat_id}.csv", f, "text/csv")},
                data={
                    "dim": DIM,
                    "seed": 42,
                    "depth": 3,
                    "db_name": DB_NAME,
                    "namespace": namespace,
                    "template_schema": json.dumps(CHAT_SCHEMA),
                    "on_conflict": "update",
                },
                timeout=TIMEOUT,
            )
        response.raise_for_status()
        result = response.json()
        
        rows_added = result.get('rows_added', 0)
        print(f"  ✅ Chat ingested: {rows_added} rows → {namespace}")
        
        # Verify the upload
        if rows_added > 0:
            verify_resp = requests.get(
                f"{SERVER_URL}/namespace/{DB_NAME}/{namespace}/count",
                headers=headers,
                timeout=10,
            )
            if verify_resp.ok:
                count = verify_resp.json().get("count", 0)
                print(f"  📊 Verified: {count} rows in namespace")
        
        return result
    except Exception as e:
        print(f"  ❌ Chat ingest failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"     Response: {e.response.text[:500]}")
        raise
    finally:
        os.unlink(tmp_path)

# ── Ingest QA questions ───────────────────────────────────────────────────────

_IDEAL_ANSWER_KEYS = ("answer", "ideal_answer", "ideal_response", "ideal_summary", "response")

@retry_request
def ingest_qa(chat_id: str, probing_questions: dict) -> Dict[str, Any]:
    """Ingest probing questions into Hyperbinder using /build_ingest_data/."""
    namespace = qa_namespace(chat_id)
    rows = []

    for category, questions in probing_questions.items():
        if not isinstance(questions, list):
            continue
        for i, q in enumerate(questions):
            question = q.get("question", "").strip()
            if not question:
                continue
            
            ideal_answer = ""
            for key in _IDEAL_ANSWER_KEYS:
                val = q.get(key, "")
                if val and str(val).strip():
                    ideal_answer = str(val).strip()
                    break
            
            rubric = q.get("rubric", [])
            rubric_str = " | ".join(rubric) if isinstance(rubric, list) else str(rubric)
            
            rows.append({
                "qa_id": f"{chat_id}_{category}_{i}",
                "chat_id": chat_id,
                "category": category,
                "difficulty": q.get("difficulty", ""),
                "plan_ref": q.get("plan_reference", ""),
                "question": question,
                "ideal_answer": ideal_answer,
                "rubric": rubric_str,
            })
    
    if not rows:
        raise ValueError("No QA rows to ingest")
    
    df = pd.DataFrame(rows)
    url = f"{SERVER_URL}/build_ingest_data/"
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    
    print(f"  📤 Uploading {len(df)} QA rows → namespace: {namespace}")
    print(f"  📊 Sample first row: {df.iloc[0].to_dict()}")
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        df.to_csv(tmp, index=False)
        tmp_path = tmp.name
    
    try:
        with open(tmp_path, "rb") as f:
            response = requests.post(
                url,
                headers=headers,
                files={"file": ("beam_qa.csv", f, "text/csv")},
                data={
                    "dim": DIM,
                    "seed": 42,
                    "depth": 3,
                    "db_name": DB_NAME,
                    "namespace": namespace,
                    "template_schema": json.dumps(QA_SCHEMA),
                    "on_conflict": "update",
                },
                timeout=TIMEOUT,
            )
        response.raise_for_status()
        result = response.json()
        
        rows_added = result.get('rows_added', 0)
        print(f"  ✅ QA ingested: {rows_added} rows → {namespace}")
        
        # Verify
        if rows_added > 0:
            verify_resp = requests.get(
                f"{SERVER_URL}/namespace/{DB_NAME}/{namespace}/count",
                headers=headers,
                timeout=10,
            )
            if verify_resp.ok:
                count = verify_resp.json().get("count", 0)
                print(f"  📊 Verified: {count} rows in namespace")
        
        return result
    except Exception as e:
        print(f"  ❌ QA ingest failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"     Response: {e.response.text[:500]}")
        raise
    finally:
        os.unlink(tmp_path)

# ── Search chat using /multihop/ (REST) ──────────────────────────────────────

def search_chat_multihop(question: str, chat_id: str, top_k: int = 10) -> List[str]:
    """
    Search chat using the REST /multihop/ endpoint.
    
    The REST /multihop/ endpoint expects:
        - query: dict with field → value mapping (simple)
        - path: list of (field, value) tuples for expansion
    """
    namespace = chat_namespace(chat_id)
    url = f"{SERVER_URL}/multihop/"
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    
    try:
        # ✅ CORRECT FORMAT for REST /multihop/
        response = requests.post(
            url,
            headers=headers,
            json={
                "query": {"content": question},  # ← Simple dict, not nested!
                "path": [],  # Empty path for simple search
                "db_name": DB_NAME,
                "namespace": namespace,
                "top_k": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        # Extract passages from results
        passages = []
        for r in results:
            if isinstance(r, dict):
                # Try different possible field names
                text = r.get("content") or r.get("text") or r.get("value") or r.get("data", {}).get("content", "")
                if text:
                    passages.append(text)
            elif isinstance(r, str):
                passages.append(r)
        
        return passages
        
    except Exception as e:
        print(f"      ✗ Multihop search failed: {e}")
        return []


def search_chat_direct(question: str, chat_id: str, top_k: int = 10) -> List[str]:
    """
    Fallback: Direct slot search using /compose/search_slots/
    """
    namespace = chat_namespace(chat_id)
    url = f"{SERVER_URL}/compose/search_slots/{DB_NAME}/{namespace}"
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "slot_queries": {
                    "content": {"query": question, "weight": 1.0},
                },
                "top_k": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return [
            r.get("data", {}).get("content", "")
            for r in results
            if r.get("data", {}).get("content")
        ]
    except Exception as e:
        print(f"      ✗ Direct search failed: {e}")
        return []


# ── Search chat (main entry point) ──────────────────────────────────────────

def search_chat(question: str, chat_id: str, top_k: int = 10) -> List[str]:
    """
    Main search function. Tries multihop first, falls back to direct search.
    """
    # Only use multihop for questions without ground truth
    # But first, let's check if this question has a stored ideal_answer
    gt = get_ground_truth(question, chat_id)
    if gt and gt.get("ideal_answer"):
        # Question has ground truth, use direct search
        return search_chat_direct(question, chat_id, top_k)
    else:
        # No ground truth, use multihop
        return search_chat_multihop(question, chat_id, top_k)


# ── Search QA ground truth ────────────────────────────────────────────────────

def get_ground_truth(question: str, chat_id: str) -> Optional[dict]:
    """Get ideal_answer and rubric from QA namespace."""
    namespace = qa_namespace(chat_id)
    url = f"{SERVER_URL}/compose/search_slots/{DB_NAME}/{namespace}"
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "slot_queries": {
                    "question": {"query": question, "weight": 1.0},
                    "chat_id": {
                        "query": chat_id,
                        "mode": "filter",
                        "encoding": "exact",
                        "threshold": 0.95,
                    },
                },
                "top_k": 1,
            },
            timeout=30,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            return results[0].get("data", {})
        return None
    except Exception as e:
        print(f"      ✗ Ground truth failed: {e}")
        return None

# ── Generate answer with DeepSeek ────────────────────────────────────────────

def generate_answer(question: str, passages: list[str], rubric: str = "") -> str:
    """Generate answer using DeepSeek."""
    if not passages:
        return "Based on the provided chat, there is no relevant information to answer this question."
    
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return passages[0] if passages else "No passages available."
    
    context = "\n\n".join(f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages))
    
    system = (
        "You are a helpful assistant answering questions about a long conversation. "
        "Answer based only on the provided passages. Be specific and accurate."
    )
    
    if rubric:
        system += f"\n\nYour answer should satisfy: {rubric}"
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", timeout=60)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {question}\n\nRelevant passages:\n{context}"}
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content or passages[0]
    except Exception as e:
        print(f"      ✗ Generation failed: {e}")
        return passages[0]

# ── Score answer ─────────────────────────────────────────────────────────────

def score_answer(generated: str, ideal_answer: str, rubric: str) -> bool:
    """Simple scoring - checks if generated answer contains rubric requirements."""
    if not rubric:
        return generated.strip() == ideal_answer.strip() if ideal_answer else True
    
    requirements = [r.strip() for r in rubric.split("|") if r.strip()]
    if not requirements:
        return True
    
    # Extract key phrases from rubric
    key_phrases = []
    for req in requirements:
        match = re.search(r'should contain:\s*(.+)', req, re.IGNORECASE)
        if match:
            key_phrases.append(match.group(1).lower())
        elif ":" in req:
            key_phrases.append(req.split(":", 1)[-1].strip().lower())
        else:
            key_phrases.append(req.lower())
    
    # Check if key phrases appear in generated answer
    generated_lower = generated.lower()
    found = sum(1 for phrase in key_phrases if phrase in generated_lower)
    return found >= len(key_phrases) * 0.6

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_chat(chat_id: str, probing_questions: dict) -> dict:
    """Run evaluation on the ingested data."""
    results = {}
    
    for category, questions in probing_questions.items():
        if not isinstance(questions, list):
            continue
        
        print(f"\n  [{category}] {len(questions)} questions")
        cat_results = []
        
        for q in questions:
            question = q.get("question", "").strip()
            if not question:
                continue
            
            # Get ground truth
            gt = get_ground_truth(question, chat_id)
            ideal_answer = (gt or {}).get("ideal_answer", "")
            rubric = (gt or {}).get("rubric", "")
            
            # Search chat using the appropriate method
            if ideal_answer:
                # Has ground truth - use direct search
                passages = search_chat_direct(question, chat_id, top_k=10)
                print(f"      🔍 Direct search for: {question[:40]}...")
            else:
                # No ground truth - use multihop
                passages = search_chat_multihop(question, chat_id, top_k=10)
                print(f"      🔍 Multihop for: {question[:40]}...")
            
            time.sleep(REQUEST_DELAY)
            
            # Generate answer
            if passages:
                generated = generate_answer(question, passages, rubric)
            elif ideal_answer:
                generated = ideal_answer
            else:
                generated = "No information available."
            
            # Score
            passed = score_answer(generated, ideal_answer, rubric)
            
            status = "✓" if passed else "✗"
            print(f"    {status} {question[:60]}... (passages: {len(passages)})")
            
            cat_results.append({
                "question": question,
                "ideal_answer": ideal_answer,
                "generated": generated,
                "passages_found": len(passages),
                "passed": passed,
            })
        
        passed = sum(1 for r in cat_results if r["passed"])
        total = len(cat_results)
        results[category] = {
            "questions": cat_results,
            "passed": passed,
            "total": total,
            "accuracy": passed / total if total > 0 else 0.0,
        }
        print(f"    → {passed}/{total} = {passed/total:.1%}")
    
    return results

# ── Wipe namespace ────────────────────────────────────────────────────────────

def wipe_namespace(chat_id: str):
    """Delete existing namespaces."""
    for ns in [chat_namespace(chat_id), qa_namespace(chat_id)]:
        try:
            response = requests.delete(
                f"{SERVER_URL}/db/{DB_NAME}/namespace/{ns}",
                headers={"X-API-Key": API_KEY} if API_KEY else {},
                timeout=30,
            )
            if response.status_code in (200, 404):
                print(f"  ✅ Deleted: {ns}")
            else:
                print(f"  ⚠️ Could not delete {ns}: {response.status_code}")
        except Exception as e:
            print(f"  ⚠️ Could not delete {ns}: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run_chat(chat_dir: Path, chat_id: str, wipe: bool = False, eval_only: bool = False):
    print(f"\n{'=' * 60}")
    print(f"  BEAM: {chat_id}")
    print(f"  DB: {DB_NAME} | Dim: {DIM}")
    print(f"{'=' * 60}")
    
    chat_file = chat_dir / "chat.json"
    if not chat_file.exists():
        print(f"  ❌ chat.json not found")
        return
    
    pq_file = chat_dir / "probing_questions" / "probing_questions.json"
    if not pq_file.exists():
        print(f"  ❌ probing_questions.json not found")
        return
    
    # Load data
    turns = load_turns(chat_file)
    with open(pq_file, encoding="utf-8") as f:
        probing_questions = json.load(f)
    
    total_q = sum(len(v) for v in probing_questions.values() if isinstance(v, list))
    print(f"  Questions: {total_q} across {len(probing_questions)} categories")
    
    # Wipe if requested
    if wipe:
        print(f"\n  🗑️ Wiping namespaces...")
        wipe_namespace(chat_id)
    
    # Ingest if not eval-only
    if not eval_only:
        print(f"\n  [1/2] Ingesting chat...")
        ingest_chat(chat_id, turns)
        
        print(f"\n  [2/2] Ingesting QA...")
        ingest_qa(chat_id, probing_questions)
    
    # Evaluate
    print(f"\n  🔍 Evaluating...")
    results = evaluate_chat(chat_id, probing_questions)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"beam_results_{chat_id}_{timestamp}.json"
    
    total_passed = sum(r["passed"] for r in results.values())
    total_questions = sum(r["total"] for r in results.values())
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "chat_id": chat_id,
            "db_name": DB_NAME,
            "dim": DIM,
            "timestamp": timestamp,
            "results": results,
            "overall": {
                "passed": total_passed,
                "total": total_questions,
                "accuracy": total_passed / total_questions if total_questions > 0 else 0.0,
            }
        }, f, indent=2)
    
    print(f"\n  ✅ Results saved to: {out_file}")
    
    # Summary
    print(f"\n{'─' * 56}")
    print(f"  RESULTS — {chat_id}")
    print(f"{'─' * 56}")
    for cat, r in results.items():
        print(f"  {cat:<30} {r['passed']:>3}/{r['total']:<3}  {r['accuracy']:.1%}")
    print(f"{'─' * 56}")
    overall = total_passed / total_questions if total_questions > 0 else 0.0
    print(f"  {'OVERALL':<30} {total_passed:>3}/{total_questions:<3}  {overall:.1%}")
    print(f"{'─' * 56}\n")

# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEAM Benchmark Ingest + Evaluation")
    parser.add_argument("--chat-dir", type=str, required=True)
    parser.add_argument("--chat-id", type=str, required=True)
    parser.add_argument("--wipe", action="store_true", help="Delete existing namespaces")
    parser.add_argument("--eval-only", action="store_true", help="Skip ingestion, only evaluate")
    args = parser.parse_args()
    
    print(f"\nServer: {SERVER_URL}")
    print(f"API Key: {'✅ Set' if API_KEY else '❌ Missing'}")
    print(f"DB: {DB_NAME}")
    
    run_chat(
        Path(args.chat_dir),
        args.chat_id,
        wipe=args.wipe,
        eval_only=args.eval_only,
    )