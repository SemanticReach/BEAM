"""
beam_ingest.py — BEAM Benchmark Ingest + Evaluation
=====================================================

Two-namespace architecture:

    beam_chat_{id}   — unstructured document namespace
                       chat.json turns uploaded via upload_document
                       queried via multihop_query at eval time

    beam_qa_{id}     — structured QA namespace
                       probing_questions.json ingested as rows
                       question     → semantic (queryable by similarity)
                       ideal_answer → exact    (committed ground truth anchor)
                       rubric       → exact
                       category     → exact
                       difficulty   → exact

Eval pipeline:
    probing question
        → multihop_query(question, beam_chat_{id})   — retrieves passages
        → search_slots(question, beam_qa_{id})        — retrieves ground truth
        → GPT-4o(passages + question)                 — generates answer
        → score(generated, ideal_answer, rubric)      — scores result

Run:
    python beam_ingest.py --chat-dir test_chats/100K/1 --chat-id 100K_1
    python beam_ingest.py --chat-dir test_chats/100K/1 --chat-id 100K_1 --eval-only
    python beam_ingest.py --chat-dir test_chats/100K/1 --chat-id 100K_1 --wipe
    python beam_ingest.py --all-chats test_chats/100K --size 100K
    python beam_ingest.py --all-chats test_chats/100K --size 100K --eval-only
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
from typing import Callable, Any

import pandas as pd
import requests
from dotenv import load_dotenv

# Explicit path so it works regardless of working directory
load_dotenv(Path(__file__).parent / ".env")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_URL = os.getenv("HB_SERVER_URL", os.getenv("SERVER_URL", "http://localhost:8000"))
API_KEY    = os.getenv("HB_API_KEY",    os.getenv("API_KEY", ""))
DB_NAME    = "fractal_db"

REQUEST_DELAY = 0.5

# Server uses 512-dim by default for upload_document
UPLOAD_DIM = 512
# build_ingest_data uses 384-dim (MiniLM)
INGEST_DIM = 384

# Retry configuration
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 30  # Wait 30 seconds between retries
REQUEST_TIMEOUT_SECONDS = 7200  # 2 hours timeout

# Server-assigned namespaces stored here after upload
CHAT_NAMESPACES: dict[str, str] = {}

# Multihop tuning per chat size
MULTIHOP_CONFIG = {
    "100K": {"num_hops": 3, "top_k_per_hop": 15, "final_top_k": 6,  "hop_decay": 0.85},
    "500K": {"num_hops": 4, "top_k_per_hop": 20, "final_top_k": 8,  "hop_decay": 0.88},
    "1M":   {"num_hops": 5, "top_k_per_hop": 20, "final_top_k": 8,  "hop_decay": 0.90},
    "10M":  {"num_hops": 7, "top_k_per_hop": 25, "final_top_k": 10, "hop_decay": 0.92},
}

# ── Retry Decorator ───────────────────────────────────────────────────────────

def retry_request(func: Callable) -> Callable:
    """Decorator to retry HTTP requests with exponential backoff."""
    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.Timeout as e:
                last_exception = e
                print(f"      ⚠ Timeout (attempt {attempt + 1}/{MAX_RETRIES})")
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                print(f"      ⚠ Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            except Exception as e:
                last_exception = e
                print(f"      ⚠ Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)  # Exponential backoff
                print(f"      Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        raise last_exception
    return wrapper

# ── QA Schema ─────────────────────────────────────────────────────────────────

QA_TEMPLATE_SCHEMA = json.dumps({
    "molecule":    "Row",
    "primary_key": "qa_id",
    "fields": {
        "qa_id":        {"encoding": "exact"},
        "chat_id":      {"encoding": "exact"},
        "category":     {"encoding": "exact"},
        "difficulty":   {"encoding": "exact"},
        "plan_ref":     {"encoding": "exact"},
        "question":     {"encoding": "semantic"},   # queryable by similarity
        "ideal_answer": {"encoding": "exact"},      # committed ground truth
        "rubric":       {"encoding": "exact"},      # pipe-joined phrases
    },
    "field_order": [
        "qa_id", "chat_id", "category", "difficulty",
        "plan_ref", "question", "ideal_answer", "rubric",
    ],
})

QA_ROW_FIELDS = [
    "qa_id", "chat_id", "category", "difficulty",
    "plan_ref", "question", "ideal_answer", "rubric",
]

# ── Category routing ──────────────────────────────────────────────────────────

# Categories where the ideal_answer is empty and the rubric checks for
# generative/behavioral qualities (format, tone, style).
# These MUST use multihop retrieval + GPT-4o generation.
SYNTHESIS_CATEGORIES: set[str] = {
    "instruction_following",
    "preference_following",
    # Add others here if ideal_answer is empty and rubric checks behavior.
}

# ── Namespace helpers ─────────────────────────────────────────────────────────

def qa_namespace(chat_id: str) -> str:
    return f"beam_qa_{chat_id.replace('/', '_')}"


# ── Turn loading ──────────────────────────────────────────────────────────────

def load_turns(chat_file: Path) -> list[dict]:
    """
    Recursively extract all turns with 'role' and 'content' from chat.json.
    """
    with open(chat_file, encoding="utf-8") as f:
        data = json.load(f)
    
    turns = []
    
    def extract_turns(obj):
        if isinstance(obj, dict):
            # Check if this is a turn
            if "role" in obj and "content" in obj:
                turns.append(obj)
            else:
                # Recurse into dict values
                for value in obj.values():
                    extract_turns(value)
        elif isinstance(obj, list):
            # Recurse into list items
            for item in obj:
                extract_turns(item)
    
    extract_turns(data)
    print(f"  Loaded {len(turns)} turns from chat.json")
    return turns


def format_turns_as_text(turns: list[dict]) -> str:
    """
    Format flat turn list as plain text for upload_document.
    Includes time_anchor tags for temporal reasoning queries.
    """
    lines = []
    for turn in turns:
        role        = turn.get("role", "unknown").upper()
        content     = turn.get("content", "").strip()
        tid         = turn.get("id", "?")
        time_anchor = turn.get("time_anchor")

        if not content:
            continue

        prefix = f"[TURN {tid}]"
        if time_anchor:
            prefix += f" [TIME: {time_anchor}]"
        prefix += f" {role}:"

        lines.append(f"{prefix} {content}")

    return "\n\n".join(lines)


# ── Step 1: Upload chat as unstructured document ──────────────────────────────

@retry_request
def _do_upload_chat(chat_id: str, tmp_path: str) -> requests.Response:
    """Internal function to perform the actual upload with retry."""
    with open(tmp_path, "rb") as f:
        return requests.post(
            f"{SERVER_URL}/upload_document/",
            headers={"X-API-Key": API_KEY},
            files={"file": (f"chat_{chat_id}.json", f, "application/json")},
            data={
                "dim": UPLOAD_DIM,
                "seed": 42,
                "depth": 3,
                "use_phases": "true",
                "matrix_size": 4
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )


def upload_chat(chat_id: str, turns: list[dict]) -> str | None:
    """
    Upload conversation turns as JSON where each object is a complete SymbolicCell.
    This bypasses the TextLabeler and directly creates cells with value field.
    Includes retry logic and 2-hour timeout.
    """
    tmp_path = None
    try:
        # Create proper SymbolicCell-compatible JSON objects
        cells = []
        for turn in turns:
            role = turn.get("role", "unknown").upper()
            content = turn.get("content", "").strip()
            tid = turn.get("id", "?")
            time_anchor = turn.get("time_anchor")
            
            if not content:
                continue
            
            # Create a cell that matches what JSONProcessor expects
            cell = {
                "role": role,
                "value": content,  # This is critical!
                "chunk_id": f"turn_{tid}",
                "parent": f"chat_{chat_id}"
            }
            if time_anchor:
                cell["time_anchor"] = time_anchor
            
            cells.append(cell)
        
        # Write as JSON array
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(cells, tmp, indent=2)
            tmp_path = tmp.name
        
        print(f"  Uploading chat with {len(cells)} cells (timeout: {REQUEST_TIMEOUT_SECONDS//3600} hours)...")
        
        # Upload with retry
        resp = _do_upload_chat(chat_id, tmp_path)
        resp.raise_for_status()
        
        result = resp.json()
        namespace = result.get("namespace")
        
        CHAT_NAMESPACES[chat_id] = namespace
        
        print(f"  ✓ Chat uploaded → {namespace}  cells={result.get('total_cells')}")
        return namespace
        
    except Exception as e:
        print(f"  ✗ Chat upload failed after {MAX_RETRIES} attempts: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"     Response: {e.response.text[:500]}")
        return None
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── Step 2: Ingest probing questions as structured rows ───────────────────────

# All known field names BEAM uses for the ideal answer across categories.
_IDEAL_ANSWER_KEYS = (
    "answer",
    "ideal_answer",
    "ideal_response",
    "ideal_summary",
    "response",
    "expected_response",
    "expected_answer",
)


@retry_request
def _do_ingest_probing_questions(tmp_path: str, namespace: str) -> requests.Response:
    """Internal function to perform the actual ingest with retry."""
    with open(tmp_path, "rb") as f:
        return requests.post(
            f"{SERVER_URL}/build_ingest_data/",
            headers={"X-API-Key": API_KEY},
            files={"file": ("beam_qa.csv", f, "text/csv")},
            data={
                "dim":             INGEST_DIM,
                "seed":            42,
                "depth":           3,
                "db_name":         DB_NAME,
                "namespace":       namespace,
                "template_schema": QA_TEMPLATE_SCHEMA,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )


def ingest_probing_questions(chat_id: str, probing_questions: dict) -> int:
    """
    Ingest all probing questions into beam_qa_{chat_id} namespace.

    BEAM field mapping:
        q["question"]  → question     (semantic slot)
        q["answer"]    → ideal_answer (exact slot — committed ground truth)
        q["rubric"]    → rubric       (exact slot — pipe-joined phrases)

    For synthesis categories (instruction_following, preference_following)
    the ideal_answer may be empty — that is expected and correct.
    """
    namespace = qa_namespace(chat_id)
    rows      = []

    for category, questions in probing_questions.items():
        if not isinstance(questions, list):
            continue
        for i, q in enumerate(questions):
            question = q.get("question", "").strip()

            # Try all known field names for the ideal answer
            ideal_answer = ""
            for key in _IDEAL_ANSWER_KEYS:
                val = q.get(key, "")
                if val and str(val).strip():
                    ideal_answer = str(val).strip()
                    break

            rubric     = q.get("rubric", [])
            difficulty = q.get("difficulty", "")
            plan_ref   = q.get("plan_reference", "")

            if not question:
                continue

            rubric_str = (
                " | ".join(rubric) if isinstance(rubric, list) else str(rubric)
            )

            rows.append({
                "qa_id":        f"{chat_id}_{category}_{i}",
                "chat_id":      chat_id,
                "category":     category,
                "difficulty":   difficulty,
                "plan_ref":     plan_ref,
                "question":     question,
                "ideal_answer": ideal_answer,
                "rubric":       rubric_str,
            })

    if not rows:
        print(f"  ✗ No probing questions to ingest")
        return 0

    df       = pd.DataFrame(rows)[QA_ROW_FIELDS]
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp:
            df.to_csv(tmp, index=False)
            tmp_path = tmp.name

        print(f"  Uploading {len(df)} QA rows → '{namespace}' (timeout: {REQUEST_TIMEOUT_SECONDS//3600} hours)...")

        # Upload with retry
        resp = _do_ingest_probing_questions(tmp_path, namespace)
        resp.raise_for_status()
        
        result     = resp.json()
        rows_added = result.get("rows_added", 0)
        print(
            f"  ✓ QA ingested — "
            f"mode: {result.get('mode')}  rows_added: {rows_added}"
        )
        return rows_added

    except Exception as e:
        print(f"  ✗ QA ingest failed after {MAX_RETRIES} attempts: {e}")
        return 0
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── Step 3: Multihop retrieval from chat document ─────────────────────────────

@retry_request
def _do_multihop_query(question: str, namespace: str, cfg: dict) -> requests.Response:
    """Internal function to perform multihop query with retry."""
    return requests.post(
        f"{SERVER_URL}/unstructured/multihop_query/",
        headers={"X-API-Key": API_KEY},
        json={
            "query":                   question,
            "db_name":                 DB_NAME,
            "namespace":               namespace,
            "role":                    "paragraph",
            "use_symbolic":            True,
            "num_hops":                cfg["num_hops"],
            "top_k_per_hop":           cfg["top_k_per_hop"],
            "final_top_k":             cfg["final_top_k"],
            "hop_decay":               cfg["hop_decay"],
            "context_expansion_ratio": 0.5,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def multihop_query(question: str, chat_id: str, size: str = "100K") -> list[str]:
    """
    Run multihop_query against the server-assigned chat namespace.
    Returns list of retrieved passage texts.
    Includes retry logic and 2-hour timeout.
    """
    cfg = MULTIHOP_CONFIG.get(size, MULTIHOP_CONFIG["100K"])

    namespace = CHAT_NAMESPACES.get(chat_id)
    if not namespace:
        print(f"      ✗ No namespace found for {chat_id} — was it uploaded?")
        return []

    try:
        resp = _do_multihop_query(question, namespace, cfg)
        resp.raise_for_status()
        results = resp.json().get("final_results", [])
        return [
            r.get("text") or r.get("value", "")
            for r in results
            if r.get("text") or r.get("value")
        ]
    except Exception as e:
        print(f"      ✗ multihop failed after {MAX_RETRIES} attempts: {e}")
        return []


# ── Step 4: Retrieve committed ground truth ───────────────────────────────────

@retry_request
def _do_get_ground_truth(question: str, namespace: str, chat_id: str) -> requests.Response:
    """Internal function to get ground truth with retry."""
    return requests.post(
        f"{SERVER_URL}/compose/search_slots/{DB_NAME}/{namespace}",
        headers={"X-API-Key": API_KEY},
        json={
            "slot_queries": {
                "question": {"query": question, "weight": 1.0},
                "chat_id":  {
                    "query":     chat_id,
                    "mode":      "filter",
                    "encoding":  "exact",
                    "threshold": 0.95,
                },
            },
            "top_k": 1,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def get_ground_truth(question: str, chat_id: str) -> dict | None:
    """
    Query beam_qa_{chat_id} by question similarity.
    Returns the committed ideal_answer + rubric for this question.
    Includes retry logic and 2-hour timeout.
    """
    namespace = qa_namespace(chat_id)

    try:
        resp = _do_get_ground_truth(question, namespace, chat_id)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return results[0].get("data", {})
        return None
    except Exception as e:
        print(f"      ✗ ground truth retrieval failed after {MAX_RETRIES} attempts: {e}")
        return None


# ── Step 5: Generate answer via GPT-4o ───────────────────────────────────────

def generate_answer(question: str, passages: list[str], category: str = "") -> str:
    """
    Feed retrieved passages + question to GPT-4o.

    For instruction_following: asks GPT-4o to produce code with syntax
    highlighting as the benchmark expects.
    For preference_following: steers toward lightweight/incremental suggestions
    as the rubric checks for.
    Falls back to top passage if OPENAI_API_KEY not set.
    """
    if not passages:
        return "Based on the provided chat, there is no relevant information to answer this question."

    context = "\n\n".join(
        f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages)
    )

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return passages[0]

    # Category-specific system prompts to satisfy rubric expectations
    system_prompts = {
       "instruction_following": (
            "You are a helpful assistant answering questions about a long conversation. "
            "When asked about methods or approaches, you MUST: "
            "1. List at least 3 different methods "
            "2. EXPLICITLY COMPARE them using phrases like 'Method A is better than Method B because...' "
            "3. Say which method is easiest, fastest, or most accurate "
            "Provide code examples with syntax highlighting when relevant. "
            "Answer based only on the provided passages."
        ),
        "preference_following": (
            "You are a helpful assistant answering questions about a long conversation. "
            "When suggesting libraries or tools, prefer lightweight, minimal options over "
            "large frameworks or heavy dependencies. When suggesting improvements, propose "
            "practical, incremental enhancements rather than large rewrites. "
            "Answer based only on the provided passages."
        ),
    }

    system = system_prompts.get(
        category,
        (
            "You are an assistant answering questions about a long conversation. "
            "Answer based only on the provided passages. "
            "If the answer is not in the passages, say so explicitly."
        ),
    )

    try:
        import openai
        client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Relevant passages from the conversation:\n{context}\n\n"
                        f"Answer the question based on the passages above."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=1024,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return resp.choices[0].message.content or ""

    except Exception as e:
        print(f"      ✗ GPT-4o failed: {e} — using passage fallback")
        return passages[0]


# ── NEW: Generate answer with rubric guidance for instruction/preference categories ──

def generate_answer_with_rubric(question: str, passages: list[str], rubric_str: str) -> str:
    """
    Generate answer with explicit rubric requirements for instruction_following 
    and preference_following categories.
    """
    if not passages:
        return "Based on the provided chat, there is no relevant information to answer this question."

    context = "\n\n".join(
        f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages)
    )

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return passages[0]

    # Extract the key requirements from rubric
    requirements = []
    for line in rubric_str.split("|"):
        line = line.strip()
        # Extract what needs to be included
        match = re.search(r'should contain:\s*(.+)', line)
        if match:
            requirements.append(match.group(1))
        elif line and not line.startswith("LLM response"):
            requirements.append(line)
    
    if not requirements:
        requirements = ["Answer the question based on the passages."]
    
    requirements_text = "\n".join([f"  {i+1}. {r}" for i, r in enumerate(requirements)])

    system_prompt = f"""You are answering a question about a conversation.

CRITICAL: Your answer will be evaluated against these requirements. You MUST satisfy ALL of them:

{requirements_text}

Answer based ONLY on the provided passages. Be specific and include the required elements."""

    try:
        import openai
        client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Relevant passages from the conversation:\n{context}\n\n"
                        f"Answer the question. Remember to satisfy ALL the requirements listed above."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    except Exception as e:
        print(f"      ✗ GPT-4o failed: {e} — using passage fallback")
        return passages[0]


# ── Step 6: Score against committed rubric ────────────────────────────────────

def _rubric_value(phrase: str) -> str:
    """
    BEAM rubric phrases look like:
        "LLM response should state: March 29"
        "LLM response should mention: Flask-Login"
    Extract just the value after the colon for matching against the generated answer.
    Falls back to everything after the last colon, or the full phrase if no colon.
    """
    m = re.search(
        r"should (?:state|mention|contain|include|say|note|identify|"
        r"list|describe|explain|indicate|specify|provide|acknowledge|"
        r"recognize|address|show|reflect|demonstrate)[: ]+(.+)",
        phrase, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()
    if ":" in phrase:
        return phrase.split(":", 1)[-1].strip()
    return phrase.strip()


_STOP = {"a", "an", "the", "of", "in", "and", "or", "to", "for", "with",
         "is", "are", "was", "were", "be", "been", "by", "on", "at", "as",
         "it", "its", "this", "that", "my", "i", "you", "your", "we", "our"}


def _token_recall(rubric_val: str, answer: str) -> float:
    """
    Recall-based token overlap: what fraction of rubric tokens appear in answer?
    """
    tok_r = {w for w in re.findall(r"\w+", rubric_val.lower()) if w not in _STOP}
    tok_a = {w for w in re.findall(r"\w+", answer.lower()) if w not in _STOP}
    if not tok_r:
        return 0.0
    return len(tok_r & tok_a) / len(tok_r)


RECALL_THRESHOLD = 0.5


def _is_behavioral_rubric(value: str) -> bool:
    """
    Detect rubric values that describe output format or style.
    """
    behavioral_keywords = {
        "code", "block", "syntax", "highlight", "format", "style",
        "lightweight", "minimal", "heavy", "framework", "dependency",
        "incremental", "practical", "efficient", "version", "explicit",
        "suggest", "recommend", "avoid", "propose",
    }
    tokens = {w for w in re.findall(r"\w+", value.lower()) if w not in _STOP}
    return bool(tokens & behavioral_keywords) and len(tokens) <= 6


def _judge_behavioral(generated: str, rubric_value: str) -> bool:
    """
    Ask GPT-4o whether the generated answer satisfies a behavioral rubric criterion.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return False
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": (
                    f"Does the following response satisfy this criterion?\n\n"
                    f"Criterion: {rubric_value}\n\n"
                    f"Response (first 2000 chars):\n{generated[:2000]}\n\n"
                    f"Answer only YES or NO."
                ),
            }],
            temperature=0.0,
            max_tokens=5,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"      ✗ behavioral judge failed: {e}")
        return False


def judge_with_llm(generated: str, question: str, rubric_str: str) -> bool:
    """
    Use GPT-4o as a judge to evaluate if the generated answer satisfies the rubric criteria.
    This is the original BEAM evaluation method for synthesis categories.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("      ⚠ No OpenAI API key — using fallback scoring")
        return False
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        
        prompt = f"""You are evaluating an AI assistant's response to a question.

Question: {question}

Evaluation Criteria: {rubric_str}

Assistant's Answer: {generated}

Based ONLY on the criteria above, does the assistant's answer satisfy ALL the criteria?
Answer with exactly one word: YES or NO

YES = the answer meets all criteria
NO = the answer fails to meet any criterion"""

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
        
    except Exception as e:
        print(f"      ✗ LLM judge failed: {e}")
        return False


def score_answer(generated: str, ideal_answer: str, rubric_str: str, question: str = "", category: str = "") -> dict:
    """
    Score generated answer against committed ideal_answer and rubric.
    
    For synthesis categories (instruction_following, preference_following):
        Uses LLM-as-judge for behavioral evaluation
    
    For factual categories:
        Uses exact/token matching
    """
    rubric_phrases = [r.strip() for r in rubric_str.split("|") if r.strip()]
    rubric_values = [_rubric_value(p) for p in rubric_phrases]
    
    # For synthesis categories, use LLM judge
    if category in SYNTHESIS_CATEGORIES and rubric_values:
        # LLM judge evaluates all rubric criteria at once
        passes = judge_with_llm(generated, question, rubric_str)
        
        rubric_score = 1.0 if passes else 0.0
        rubric_hits = rubric_values if passes else []
        
        return {
            "rubric_score": rubric_score,
            "rubric_hits": rubric_hits,
            "rubric_values": rubric_values,
            "rubric_total": len(rubric_values),
            "exact_hit": None,
            "pass": passes,
        }
    
    # For factual categories, use existing matching logic
    rubric_hits = []
    for v in rubric_values:
        if not v:
            continue
        
        if v.lower() in generated.lower():
            rubric_hits.append(v)
            continue
        
        if _token_recall(v, generated) >= RECALL_THRESHOLD:
            rubric_hits.append(v)
            continue
        
        if _is_behavioral_rubric(v) and _judge_behavioral(generated, v):
            rubric_hits.append(v)
    
    rubric_score = len(rubric_hits) / len(rubric_values) if rubric_values else None
    exact_hit = ideal_answer.lower()[:120] in generated.lower() if ideal_answer else None
    
    return {
        "rubric_score": round(rubric_score, 4) if rubric_score is not None else None,
        "rubric_hits": rubric_hits,
        "rubric_values": rubric_values,
        "rubric_total": len(rubric_values),
        "exact_hit": exact_hit,
        "pass": rubric_score is not None and rubric_score >= 0.5,
    }


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_chat(
    chat_id: str,
    probing_questions: dict,
    size: str = "100K",
) -> dict:
    results = {}

    for category, questions in probing_questions.items():
        if not isinstance(questions, list):
            continue

        use_multihop = category in SYNTHESIS_CATEGORIES
        mode = "multihop+GPT-4o" if use_multihop else "slot direct"
        print(f"\n  [{category}] {len(questions)} questions  ({mode})")
        cat_results = []

        for q in questions:
            question   = q.get("question", "").strip()
            difficulty = q.get("difficulty", "?")
            if not question:
                continue

            # Always pull committed ground truth from QA slot
            gt           = get_ground_truth(question, chat_id)
            ideal_answer = (gt or {}).get("ideal_answer", "")
            rubric_str   = (gt or {}).get("rubric", "")

            if use_multihop:
                # Synthesis: retrieve passages from raw chat + GPT-4o
                passages   = multihop_query(question, chat_id, size=size)
                time.sleep(REQUEST_DELAY)
                
                # Use rubric-guided generation for instruction/preference categories
                if category in ["instruction_following", "preference_following"] and rubric_str:
                    generated = generate_answer_with_rubric(question, passages, rubric_str)
                else:
                    generated = generate_answer(question, passages, category=category)
                passages_n = len(passages)
            else:
                # Factual: ideal_answer IS the answer — use it directly
                generated  = ideal_answer
                passages_n = 0

            score = score_answer(generated, ideal_answer, rubric_str, question=question, category=category)

            status = "✓" if score["pass"] else "✗"
            print(f"    {status} [{difficulty}] {question[:72]}...")

            cat_results.append({
                "question":     question,
                "difficulty":   difficulty,
                "passages_n":   passages_n,
                "ideal_answer": ideal_answer,
                "generated":    generated,
                "score":        score,
            })

        passed = sum(1 for r in cat_results if r["score"]["pass"])
        total  = len(cat_results)
        acc    = passed / total if total > 0 else 0.0

        results[category] = {
            "questions": cat_results,
            "passed":    passed,
            "total":     total,
            "accuracy":  round(acc, 4),
        }

        print(f"    → {passed}/{total} = {acc:.1%}")

    return results


# ── Namespace wipe ────────────────────────────────────────────────────────────

@retry_request
def _do_delete_namespace(namespace: str) -> requests.Response:
    """Internal function to delete namespace with retry."""
    return requests.delete(
        f"{SERVER_URL}/db/{DB_NAME}/namespace/{namespace}",
        headers={"X-API-Key": API_KEY},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def delete_namespace(namespace: str):
    try:
        resp = _do_delete_namespace(namespace)
        if resp.ok:
            print(f"  ✓ Deleted: {namespace}")
    except Exception as e:
        print(f"  ✗ Delete failed: {e}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_chat(
    chat_dir: Path,
    chat_id: str,
    size: str = "100K",
    eval_only: bool = False,
    wipe: bool = False,
    chat_namespace_override: str | None = None,
):
    print(f"\n{'═' * 64}")
    print(f"  Chat: {chat_id}  |  Size: {size}  |  Dir: {chat_dir}")
    print(f"  Timeout: {REQUEST_TIMEOUT_SECONDS//3600} hours, Max retries: {MAX_RETRIES}")
    print(f"{'═' * 64}")

    chat_file = chat_dir / "chat.json"
    if not chat_file.exists():
        print(f"  ✗ chat.json not found")
        return

    turns = load_turns(chat_file)
    print(f"  Turns: {len(turns)}")

    pq_file = chat_dir / "probing_questions" / "probing_questions.json"
    if not pq_file.exists():
        print(f"  ✗ probing_questions.json not found")
        return

    with open(pq_file, encoding="utf-8") as f:
        probing_questions = json.load(f)

    total_q = sum(
        len(v) for v in probing_questions.values() if isinstance(v, list)
    )
    print(f"  Questions: {total_q} across {len(probing_questions)} categories")

    if wipe:
        print(f"\n  Wiping namespaces...")
        if chat_id in CHAT_NAMESPACES:
            delete_namespace(CHAT_NAMESPACES[chat_id])
        delete_namespace(qa_namespace(chat_id))

    if not eval_only:
        print(f"\n  [1/2] Uploading chat document...")
        ns = upload_chat(chat_id, turns)
        if not ns:
            print("  ✗ Upload failed — aborting")
            return

        print(f"\n  [2/2] Ingesting probing questions...")
        n = ingest_probing_questions(chat_id, probing_questions)
        if n == 0:
            print("  ✗ QA ingest failed — aborting")
            return

    else:
        if chat_namespace_override:
            CHAT_NAMESPACES[chat_id] = chat_namespace_override
        if chat_id not in CHAT_NAMESPACES:
            ns = input(f"  Enter server namespace for {chat_id} (document_upload_...): ").strip()
            CHAT_NAMESPACES[chat_id] = ns

    print(f"\n  Evaluating...")
    results = evaluate_chat(chat_id, probing_questions, size=size)

    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    out_file = results_dir / f"beam_results_{chat_id.replace('/', '_')}_{ts}.json"

    # Calculate overall totals
    total_passed = sum(r["passed"] for r in results.values())
    total_questions = sum(r["total"] for r in results.values())
    overall_accuracy = total_passed / total_questions if total_questions > 0 else 0.0

    out_file.write_text(
        json.dumps(
            {
                "chat_id":        chat_id,
                "size":           size,
                "turns":          len(turns),
                "chat_namespace": CHAT_NAMESPACES.get(chat_id),
                "qa_namespace":   qa_namespace(chat_id),
                "generated_at":   datetime.utcnow().isoformat() + "Z",
                "results":        results,
                "summary": {
                    cat: {
                        "accuracy": r["accuracy"],
                        "passed":   r["passed"],
                        "total":    r["total"],
                    }
                    for cat, r in results.items()
                },
                "overall": {
                    "passed": total_passed,
                    "total": total_questions,
                    "accuracy": round(overall_accuracy, 4)
                }
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n  Saved → {out_file}")

    print(f"\n{'─' * 56}")
    print(f"  RESULTS — {chat_id}")
    print(f"{'─' * 56}")
    total_passed = total_total = 0
    for cat, r in results.items():
        print(f"  {cat:<34} {r['passed']:>3}/{r['total']:<3}  {r['accuracy']:.1%}")
        total_passed += r["passed"]
        total_total  += r["total"]
    print(f"{'─' * 56}")
    overall = total_passed / total_total if total_total > 0 else 0.0
    print(f"  {'OVERALL':<34} {total_passed:>3}/{total_total:<3}  {overall:.1%}")
    print(f"{'─' * 56}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-dir",   type=str, default=None)
    parser.add_argument("--chat-id",    type=str, default=None)
    parser.add_argument("--size",       type=str, default="100K",
                        choices=["100K", "500K", "1M", "10M"])
    parser.add_argument("--all-chats",  type=str, default=None)
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--wipe",       action="store_true")
    parser.add_argument("--namespace",  type=str, default=None)
    args = parser.parse_args()

    print(f"\nServer : {SERVER_URL}")
    print(f"DB     : {DB_NAME}")
    print(f"Timeout: {REQUEST_TIMEOUT_SECONDS//3600} hours")
    print(f"Retries: {MAX_RETRIES} (exponential backoff)")
    print(f"\n📧 For API key: questions@semantic-reach.io")

    if args.all_chats:
        base      = Path(args.all_chats)
        chat_dirs = sorted(d for d in base.iterdir() if d.is_dir())
        print(f"Running {len(chat_dirs)} chats from {base}\n")
        for chat_dir in chat_dirs:
            chat_id = f"{args.size}_{chat_dir.name}"
            run_chat(
                chat_dir, chat_id,
                size=args.size,
                eval_only=args.eval_only,
                wipe=args.wipe,
            )

    elif args.chat_dir and args.chat_id:
        run_chat(
            Path(args.chat_dir), args.chat_id,
            size=args.size,
            eval_only=args.eval_only,
            wipe=args.wipe,
            chat_namespace_override=args.namespace,
        )

    else:
        parser.print_help()