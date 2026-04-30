"""
Merged Architecture: smoking_rag_app.py + SHS Chatbot (Pinecone)
================================================================
Upgraded 14-Step Pipeline:

  [STEP 1 ] Age Gate         — verify & tier user age (no behavioral analysis)
  [STEP 2 ] RAG Retrieval    — Pinecone semantic search (Mukesh)
  [STEP 3 ] Confidence Score — LOW / MEDIUM / HIGH from chunk scores
  [STEP 4 ] Low-conf guard   — block tier-2 users if LOW confidence
  [STEP 5 ] LLM Generation   — Ollama (temperature=0.1, medical prefix)
  [STEP 6 ] Encoding Guard   — discard & regenerate if non-Latin chars found
  [STEP 7 ] Hallucination    — known-bad-fact + contradiction checks
  [STEP 8 ] Fact-Check       — second Ollama call verifies accuracy (temp=0.1)
  [STEP 9 ] Accuracy Rewrite — third Ollama call if fact-check FAILs
  [STEP 10] Safety Filter    — Ollama call for users aged ≤17 (temp=0.1)
  [STEP 11] Safety Rewrite   — Ollama rewrites if UNSAFE
  [STEP 12] Display          — response + confidence badge + labels
  [STEP 13] Feedback         —  / 👎 with optional comment
  [STEP 14] Logging          — questions_log.csv, feedback_log.csv, error logs

Changes from previous version:
  - REMOVED BehavioralAgeAnalyzer (age deception via vocabulary — discriminatory)
  - REMOVED silent age override based on language patterns
  - REPLACED with honest age gate at app startup
  - ADDED hallucination guard (encoding, known-bad-facts, contradictions)
  - ADDED fact-checking Ollama layer (Step 8/9)
  - ADDED safety filter Ollama layer for teens (Step 10/11)
  - ADDED response confidence scoring (Step 3)
  - ADDED feedback buttons and CSV logging (Step 13)
  - ADDED admin stats panel (sidebar)
  - ALL Ollama calls now use temperature=0.1
  - ALL prompts prefixed with medical accuracy instruction

Admin panel for knowledge-base uploads is at the bottom.
"""

import streamlit as st
import requests
import json
import logging
import re
import sys
import os
import csv
import unicodedata
import hashlib
import threading
import concurrent.futures
from dotenv import load_dotenv, dotenv_values
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from openai import OpenAI

# ── SSL cert fix ───────────────────────────────────────────────────────────────
# SSL_CERT_FILE may point to a stale venv path.  Replace with active certifi.
try:
    import certifi as _certifi
    _ssl_cert = os.environ.get("SSL_CERT_FILE", "")
    if _ssl_cert and not os.path.isfile(_ssl_cert):
        os.environ["SSL_CERT_FILE"] = _certifi.where()
except Exception:
    pass

# ── optional heavy deps ────────────────────────────────────────────────────────
try:
    import numpy as np
    import statistics
    _numpy_ok = True
except ImportError:
    _numpy_ok = False

# ── Pinecone / embedding imports (Mukesh's stack) ─────────────────────────────
_pinecone_ok = False
_torch_ok = False

try:
    from pinecone import Pinecone as _PineconeClient
    from transformers import AutoTokenizer, AutoModel
    import torch
    _pinecone_ok = True
    _torch_ok = True
except Exception:
    pass

# ── conversation + orchestration imports (Mukesh's stack) ─────────────────────
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SHS_chatbot_pinecone-main"))
    from conversation_manager import conversation_manager as _conv_manager, ConversationState  # pyright: ignore[reportMissingImports]
except Exception:
    _conv_manager = None
    ConversationState = None

try:
    from chatbot_orchestrator import chatbot_orchestrator as _orchestrator  # pyright: ignore[reportMissingImports]
except Exception:
    _orchestrator = None

try:
    from prompt_engine import prompt_engine as _prompt_engine  # pyright: ignore[reportMissingImports]
except Exception:
    _prompt_engine = None

try:
    from ollama_test import ask_ollama as _ask_ollama  # pyright: ignore[reportMissingImports]
except Exception:
    _ask_ollama = None

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ╔╗
# ║                    CONSTANTS & LOG FILE SETUP                              ║
# ╚

# All log files live in a 'logs/' subdirectory next to this script.
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

QUESTIONS_LOG    = LOG_DIR / "questions_log.csv"
FEEDBACK_LOG     = LOG_DIR / "feedback_log.csv"
HALLUCINATION_LOG = LOG_DIR / "hallucination_errors.log"
FACT_CHECK_LOG   = LOG_DIR / "fact_check_failures.log"
SAFETY_LOG       = LOG_DIR / "safety_rewrites.log"
PIPELINE_ERR_LOG = LOG_DIR / "pipeline_errors.log"

# CRITICAL FIX: All Ollama prompts must start with this prefix to ground
# the model in evidence-based medicine and prevent hallucinated terminology.
MEDICAL_PREFIX = (
    "You are a medically accurate health educator specialising in secondhand smoke (SHS) "
    "and its effects on children's health. You help parents understand SHS risks and "
    "support them in quitting smoking to protect their child. "
    "Only state facts supported by CDC, NHS, or WHO guidelines. "
    "Do not invent terminology. Respond in English only.\n\n"
)

MODEL = "gpt-4o-mini"
_ENV_PATH = Path(__file__).parent / ".env"
_ENV_TXT_PATH = Path(__file__).parent / ".env.txt"

if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
elif _ENV_TXT_PATH.exists():
    load_dotenv(dotenv_path=_ENV_TXT_PATH, override=True)

_dotenv_map = {}
if _ENV_PATH.exists():
    _dotenv_map.update(dotenv_values(_ENV_PATH))
if _ENV_TXT_PATH.exists():
    _dotenv_map.update(dotenv_values(_ENV_TXT_PATH))
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY", "")
    or os.getenv("OPENAI_KEY", "")
    or (_dotenv_map.get("OPENAI_API_KEY") or "")
    or (_dotenv_map.get("OPENAI_KEY") or "")
)
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ── CSV initialisation helper ─────────────────────────────────────────────────

def _ensure_csv(path: Path, headers: List[str]) -> None:
    """Create a CSV file with the given headers if it does not yet exist."""
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)


def _init_log_files() -> None:
    """Ensure all log CSV files exist with correct column headers."""
    _ensure_csv(QUESTIONS_LOG, [
        "timestamp", "user_age_tier", "question", "confidence_level",
        "safety_filter_triggered", "fact_check_result",
        "hallucination_detected", "response_discarded",
    ])
    _ensure_csv(FEEDBACK_LOG, [
        "timestamp", "user_age_tier", "question", "response_snippet",
        "rating", "user_comment", "safety_filter_triggered", "fact_check_result",
    ])


_init_log_files()


# ╔╗
# ║                         LOGGING HELPERS                                    ║
# ╚

def log_question(
    user_age_tier: int,
    question: str,
    confidence_level: str,
    safety_filter_triggered: bool,
    fact_check_result: str,
    hallucination_detected: bool,
    response_discarded: bool,
) -> None:
    """Append one row to questions_log.csv (Upgrade 6)."""
    try:
        with open(QUESTIONS_LOG, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                user_age_tier,
                question[:300],
                confidence_level,
                "yes" if safety_filter_triggered else "no",
                fact_check_result,
                "yes" if hallucination_detected else "no",
                "yes" if response_discarded else "no",
            ])
    except Exception as e:
        logger.warning("log_question failed: %s", e)


def log_feedback(
    user_age_tier: int,
    question: str,
    response_snippet: str,
    rating: str,
    user_comment: str,
    safety_filter_triggered: bool,
    fact_check_result: str,
) -> None:
    """Append one row to feedback_log.csv (Upgrade 5)."""
    try:
        with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                user_age_tier,
                question[:300],
                response_snippet[:100],
                rating,
                user_comment[:200],
                "yes" if safety_filter_triggered else "no",
                fact_check_result,
            ])
    except Exception as e:
        logger.warning("log_feedback failed: %s", e)


def log_hallucination_error(detail: str, broken_response: str) -> None:
    """Append hallucination / encoding error to hallucination_errors.log."""
    try:
        with open(HALLUCINATION_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n[{datetime.now().isoformat()}] {detail}\n"
                f"--- BROKEN RESPONSE ---\n{broken_response[:500]}\n"
                f"--- END ---\n"
            )
    except Exception as e:
        logger.warning("log_hallucination_error failed: %s", e)


def log_fact_check_failure(
    question: str,
    original_response: str,
    flagged_claims: List[str],
) -> None:
    """Append fact-check FAIL details to fact_check_failures.log (Upgrade 2)."""
    try:
        with open(FACT_CHECK_LOG, "a", encoding="utf-8") as f:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question[:300],
                "original_response": original_response[:500],
                "flagged_claims": flagged_claims,
            }
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("log_fact_check_failure failed: %s", e)


def log_safety_rewrite(
    user_age: int,
    original: str,
    rewritten: str,
    reason: str,
) -> None:
    """Append safety rewrite details to safety_rewrites.log (Upgrade 3)."""
    try:
        with open(SAFETY_LOG, "a", encoding="utf-8") as f:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "user_age": user_age,
                "original_response": original[:500],
                "rewritten_response": rewritten[:500],
                "reason": reason,
            }
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("log_safety_rewrite failed: %s", e)


def log_pipeline_error(detail: str) -> None:
    """Append pipeline error to pipeline_errors.log."""
    try:
        with open(PIPELINE_ERR_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {detail}\n")
    except Exception as e:
        logger.warning("log_pipeline_error failed: %s", e)


# ╔╗
# ║               SHARED OLLAMA CALL HELPER                                    ║
# ║  All LLM calls route through here so temperature=0.1 is enforced.         ║
# ╚

def call_ollama(
    prompt: str,
    model: str = None,
    temperature: float = 0.1,
    num_predict: int = 300,
    num_ctx: int = 1024,
) -> str:
    """
    Send a prompt to the local Ollama REST API and return the text response.

    # MODEL SWITCH: defaults to MODEL constant (qwen2.5:3b)
    # Change 3: num_ctx=1024 controls memory; num_predict per call type:
    #   main=250, fact-check=60, safety=20, rewrite=200, loop-detect=10
    # Change 3: repeat_penalty=1.3 prevents looping/repetition
    # Change 4 (Fallback chain):
    #   1. Try with num_ctx as supplied
    #   2. Retry with num_ctx=512 if empty/error
    #   3. If retry also fails, return "" — caller handles gracefully
    - Errors are written to pipeline_errors.log.
    """
    if model is None:
        model = MODEL
    if _openai_client is None:
        log_pipeline_error("OPENAI_API_KEY is missing; returning empty response")
        return ""

    try:
        response = _openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MEDICAL_PREFIX.strip()},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=num_predict,
            timeout=120,
        )
        content = response.choices[0].message.content if response.choices else ""
        return (content or "").strip()
    except Exception as e:
        log_pipeline_error(f"call_openai exception: {e}")
        return ""


# ╔╗
# ║         CHANGE 5 — PYTHON RESPONSE QUALITY CHECKS                         ║
# ║  SPEED FIX 1: These run FIRST before any downstream LLM calls.            ║
# ║  No network I/O — immediate rejection of obviously bad responses.          ║
# ╚

# SPEED IMPROVEMENTS EXPECTED:
# Fix 1 (Python checks first):  saves ~2-3 sec — avoids LLM calls for bad responses
# Fix 2 (parallel calls 18+):   saves ~3-5 sec — fact-check runs in thread pool
# Fix 3 (session cache):        saves ~8-10 sec — skips entire pipeline for repeated questions
# Fix 4 (streaming display):    reduces perceived latency by ~3 sec
# Fix 5 (strict token limits):  saves ~1-2 sec per LLM call
# Fix 6 (progress bar):         reduces perceived latency (user sees progress)
# Fix 7 (lazy load RAG):        saves ~5-10 sec at startup
# Total real saving: 5-10 sec per query, 10+ sec at startup

def check_response_quality(text: str) -> List[str]:
    """
    Python-level quality checks run BEFORE any downstream LLM calls.

    Returns a list of issue strings (empty list = no issues found).

    Check 1: Loop detection — duplicate sentences (> 30% duplicated)
    Check 2: Length check — must be 30–600 words
    Check 3: Language check — non-Latin characters (ord > 1000) banned
    Check 4: Stage model terms — transtheoretical model terms banned from health ed
    Check 5: Contradiction check — heart rate cannot both increase AND decrease
    """
    issues: List[str] = []
    if not text or not text.strip():
        issues.append("Empty response")
        return issues

    # Check 1 — Duplicate sentences (loop detection)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]
    if len(sentences) > 3:
        unique_count = len(set(s.lower() for s in sentences))
        if unique_count < len(sentences) * 0.7:
            issues.append(
                f"Loop detected: {len(sentences) - unique_count} duplicate sentences found"
            )

    # Check 2 — Length (30–600 words)
    word_count = len(text.split())
    if word_count < 30:
        issues.append(f"Too short: {word_count} words (minimum 30)")
    elif word_count > 600:
        issues.append(f"Too long: {word_count} words (maximum 600)")

    # Check 3 — Non-Latin characters (encoding / language error)
    non_latin = [c for c in text if ord(c) > 1000]
    if len(non_latin) > 3:
        issues.append(
            f"Non-Latin characters found: {len(non_latin)} chars (possible encoding error)"
        )

    # Check 4 — Stage model terms (transtheoretical model banned)
    STAGE_MODEL_TERMS = [
        "precontemplation", "contemplation", "preparation stage",
        "action stage", "maintenance stage", "transtheoretical",
        "stages of change", "prochaska",
    ]
    text_lower = text.lower()
    for term in STAGE_MODEL_TERMS:
        if term in text_lower:
            issues.append(
                f"Stage model term found: '{term}' — banned from health education responses"
            )
            break

    # Check 5 — Contradiction check (nicotine and heart rate)
    if (
        re.search(r"increases?\s+(?:the\s+)?heart\s+rate", text_lower)
        and re.search(r"decreases?\s+(?:the\s+)?heart\s+rate", text_lower)
    ):
        issues.append(
            "Contradiction: response claims nicotine both increases and decreases heart rate"
        )

    return issues


# ╔╗
# ║                UPGRADE 1 — AGE GATE                                        ║
# ║  Replaces BehavioralAgeAnalyzer.  Age is accepted as stated; no           ║
# ║  vocabulary analysis, no silent override.                                  ║
# ╚

def get_age_tier(age: int) -> int:
    """
    Map a stated age to a content tier.

    Tier 1 (≤13)  — block app entirely
    Tier 2 (14-17) — strict / teen mode (safety filter active)
    Tier 3 (18+)   — full medical detail allowed
    """
    if age <= 13:
        return 1
    if age <= 17:
        return 2
    return 3


def tier_to_category(tier: int) -> str:
    """Map age tier to the legacy age_category string used by SmokingContentFilter."""
    return {1: "child", 2: "teen", 3: "adult"}.get(tier, "adult")


def render_age_gate() -> None:
    """
    Display the age verification gate.

    Sets st.session_state.user_age when the user submits a valid age.
    Blocks further rendering until age is provided.
    No behavioral analysis is performed — the stated age is accepted as truth.
    """
    st.title(" Protecting Your Child from Secondhand Smoke")
    st.markdown(
        "### A support tool for parents and caregivers who want to quit smoking "
        "to protect their child's health."
    )

    st.info(
        "💡 **Why this tool exists:** Secondhand smoke (SHS) is one of the leading "
        "causes of preventable illness in children — linked to asthma, ear infections, "
        "SIDS, and respiratory disease. This assistant provides evidence-based information "
        "to help you understand those risks and take steps to quit."
    )

    st.markdown("---")
    st.markdown("### Before we begin, please confirm your age.")
    st.caption(
        "Your age helps us show you age-appropriate health information. "
        "We do not store or share this information."
    )

    age_input = st.text_input("Enter your age:", key="age_input_field", max_chars=3)

    if st.button("Continue", type="primary"):
        raw = age_input.strip()
        if not raw.isdigit():
            st.error("Please enter a valid age to continue.")
            return
        age = int(raw)
        if age < 10 or age > 100:
            st.error("Please enter a valid age to continue.")
            return
        # Accept the stated age as truth — no behavioral override
        st.session_state.user_age = age
        st.rerun()


# ╔╗
# ║         CONTENT FILTER — Query Filtering + Prompt Construction             ║
# ║  BehavioralAgeAnalyzer removed.  Age accepted from age gate only.         ║
# ╚

class SmokingContentFilter:
    """Age-based content filtering for smoking-related queries."""

    def __init__(self):
        # NOTE: BehavioralAgeAnalyzer removed — age is now provided by the
        # age gate and accepted as stated.  No vocabulary-based override.

        self.age_policies = {
            "child": {
                "age_range": "6-12",
                "allowed_topics": ["health effects", "saying no", "why smoking is bad"],
                "forbidden_topics": [
                    "smoking methods", "tobacco products", "purchasing", "how to smoke",
                    "tobacco types", "cigarette brands", "hookah", "shisha", "vaping methods",
                    "rolling tobacco", "smoking techniques", "best tobacco", "where to buy",
                ],
                "response_style": "simple, protective, educational",
                "key_messages": [
                    "Smoking is very harmful",
                    "It's okay to say no",
                    "Ask trusted adults for help",
                ],
            },
            "teen": {
                "age_range": "13-17",
                "allowed_topics": ["health risks", "peer pressure", "addiction science", "prevention"],
                "forbidden_topics": [
                    "purchasing information", "how to smoke", "smoking methods", "where to buy",
                    "tobacco types", "best tobacco", "cigarette brands", "hookah methods",
                    "how to use hookah", "vaping techniques", "rolling tobacco", "smoking techniques",
                    "how to inhale", "best cigarettes", "shisha flavors",
                ],
                "response_style": "informative, respectful, health-focused",
                "key_messages": [
                    "Understand serious health risks",
                    "Resist peer pressure",
                    "Seek support if needed",
                ],
            },
            "adult": {
                "age_range": "18+",
                "allowed_topics": [
                    "all smoking information", "tobacco types", "cessation methods",
                    "harm reduction", "comprehensive health data", "hookah",
                    "smoking methods", "product information",
                ],
                "forbidden_topics": [],
                "response_style": "comprehensive, factual, supportive",
                "key_messages": [
                    "Full information access",
                    "Cessation support available",
                    "Make informed decisions",
                ],
            },
        }

        # Keywords that signal a parent needs help setting respectful boundaries
        # with family members, elders, or guests who smoke around children.
        self._SOCIAL_BOUNDARY_KEYWORDS = [
            "elder", "elders", "guest", "guests", "family", "relative", "relatives",
            "visitor", "visitors", "in-laws", "grandparent", "grandparents",
            "uncomfortable asking", "ask people not to smoke", "ask them not to smoke",
            "causing conflict", "without causing conflict", "without conflict",
            "avoid conflict", "how do i explain", "how do i tell", "how do i ask",
            "around my child", "around my baby", "around my kids", "around children",
            "smoke in the house", "smoke indoors", "smoke near", "smoke around",
        ]

        self.smoking_keywords = {
            "high_risk": [
                "how to smoke", "where to buy", "cigarette brands", "smoking techniques",
                "light up", "inhale", "best tobacco", "tobacco types", "what tobacco",
                "which tobacco", "hookah", "shisha", "how to use", "vaping methods",
                "rolling tobacco", "pack a bowl", "best cigarettes",
                "how do i smoke", "teach me to smoke", "learn to smoke", "start smoking",
            ],
            "medium_risk": ["tobacco", "nicotine", "cigarettes", "smoking", "vaping", "e-cigarettes"],
            "educational": [
                "health effects", "lung cancer", "quit smoking", "cessation",
                "prevention", "peer pressure",
            ],
        }

        # ResponseValidator is defined below; instantiated here after class def.
        self.response_validator = ResponseValidator(self)

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_age_category(self, age: int) -> str:
        if age < 13:
            return "child"
        if age < 18:
            return "teen"
        return "adult"

    def detect_slang_evasion(self, query: str) -> List[str]:
        slang_terms = {
            "cigarettes": ["cigs", "smokes", "cancer sticks", "coffin nails", "darts"],
            "smoking": ["lighting up", "taking a drag", "puffing", "burning one"],
            "tobacco": ["baccy", "leaf", "chew"],
            "vaping": ["clouds", "juice", "mod", "pod"],
        }
        detected = []
        query_lower = query.lower()
        for category, terms in slang_terms.items():
            for term in terms:
                if term in query_lower:
                    detected.append(f"{term} → {category}")
        return detected

    def classify_query_risk(self, query: str) -> Tuple[str, List[str]]:
        query_lower = query.lower()
        detected_keywords: List[str] = []
        risk_level = "low"
        for level, keywords in self.smoking_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected_keywords.append(keyword)
                    if level == "high_risk":
                        risk_level = "high"
                    elif level == "medium_risk" and risk_level != "high":
                        risk_level = "medium"
        return risk_level, detected_keywords

    # Keyword clusters for purchase/acquisition intent detection
    _PURCHASE_KEYWORDS = [
        "buy", "purchase", "get", "obtain", "order", "find", "where", "shop",
        "store", "retailer", "seller", "dealer", "market",
    ]
    _PRODUCT_KEYWORDS = [
        "cigarette", "tobacco", "cig", "smoke", "vape", "juul", "hookah",
        "shisha", "nicotine", "e-cigarette", "ecig", "joint", "weed",
    ]

    def _matches_purchase_intent(self, query_lower: str) -> bool:
        """Detect purchase/acquisition intent even when phrased indirectly."""
        has_purchase = any(kw in query_lower for kw in self._PURCHASE_KEYWORDS)
        has_product = any(kw in query_lower for kw in self._PRODUCT_KEYWORDS)
        # Also catch exact forbidden multi-word phrases
        explicit_phrases = [
            "where to buy", "where to get", "how to get", "how to buy",
            "where can i buy", "where can i get", "how do i get", "how do i buy",
        ]
        has_explicit = any(p in query_lower for p in explicit_phrases)
        return (has_purchase and has_product) or has_explicit

    def should_block_query(self, age_category: str, query: str) -> Tuple[bool, str]:
        """Block queries that request forbidden content for the user's age tier."""
        risk_level, _ = self.classify_query_risk(query)
        query_lower = query.lower()

        if age_category in ["child", "teen"]:
            # Check purchase/acquisition intent (handles paraphrasing)
            if self._matches_purchase_intent(query_lower):
                return True, (
                    "Access denied: Information about obtaining tobacco products is not "
                    "available for users under 18. This aligns with CDC/WHO youth protection guidelines."
                )
            # Check exact forbidden topic phrases (single-word topics still work)
            for forbidden in self.age_policies[age_category]["forbidden_topics"]:
                # For single-word topics, use word-boundary matching
                words = forbidden.lower().split()
                if len(words) == 1:
                    if re.search(r'\b' + re.escape(forbidden.lower()) + r'\b', query_lower):
                        return True, (
                            f"Access denied: Users under 18 cannot access information about '{forbidden}'. "
                            "This restriction is based on CDC/WHO tobacco prevention guidelines."
                        )
                else:
                    if forbidden.lower() in query_lower:
                        return True, (
                            f"Access denied: Users under 18 cannot access information about '{forbidden}'. "
                            "This restriction is based on CDC/WHO tobacco prevention guidelines."
                        )
            if risk_level == "high":
                return True, (
                    "Access denied: This type of information is restricted for users under 18. "
                    "Per CDC/WHO guidelines, minors should only access tobacco prevention and "
                    "health education content."
                )
        return False, ""

    # ── Step 5 — Prompt Construction ─────────────────────────────────────────

    # Keywords that indicate a parent asking about their child's SHS exposure
    _SHS_CHILD_KEYWORDS = [
        "my child", "my kid", "my baby", "my infant", "my son", "my daughter",
        "my toddler", "my newborn", "secondhand", "second hand", "second-hand",
        "passive smoke", "shs", "ets", "thirdhand", "third hand", "third-hand",
        "smoke around", "smoking near", "smoking indoors", "smok", "baby",
        "asthma", "ear infection", "sids", "cot death", "wheez",
    ]

    # Keywords that signal a parent asking for cessation/quitting help
    _CESSATION_KEYWORDS = [
        "quit", "quitting", "stop smoking", "give up", "cessation", "nrt",
        "nicotine patch", "nicotine gum", "nicotine replacement", "varenicline",
        "champix", "chantix", "bupropion", "zyban", "cold turkey", "withdrawal",
        "cravings", "urge to smoke", "how do i stop", "how to stop",
        "help me quit", "want to quit", "trying to quit",
    ]

    def _classify_parent_intent(self, query: str) -> str:
        """
        Classify a query as 'shs_child', 'cessation', 'social_boundary', or 'general'.

        Used to select the most targeted prompt variant for adult users.
        """
        q = query.lower()
        social_score = sum(1 for kw in self._SOCIAL_BOUNDARY_KEYWORDS if kw in q)
        shs_score = sum(1 for kw in self._SHS_CHILD_KEYWORDS if kw in q)
        ces_score = sum(1 for kw in self._CESSATION_KEYWORDS if kw in q)
        if social_score > 0:
            return "social_boundary"
        if shs_score > 0 and ces_score == 0:
            return "shs_child"
        if ces_score > 0:
            return "cessation"
        return "general"

    def create_age_appropriate_prompt(
        self,
        age_category: str,
        query: str,
        rag_context: str = "",
    ) -> str:
        """
        Build a short, single-task prompt optimised for qwen2.5:3b.

                Adult users (parents/caregivers) get one of four targeted variants:
                    - shs_child: focused on child health impacts of SHS
                    - cessation: focused on quit strategies for parents
                    - social_boundary: respectful scripts for family / guest smoking conflicts
                    - general: broad smoking/health question

                Teen/child users get the existing strict-safety prompt.
        """
        rag_block = (
            f"\n\nUse the reference facts below only for health claims. "
            f"Do not copy author citations, incomplete sentence fragments, or raw context wording.\n"
            f"CONTEXT:\n{rag_context}"
            if rag_context else ""
        )
        response_rules = (
            "Answer the user's question directly. "
            "Do not ask the user follow-up questions. "
            "Do not repeat generic warnings unless directly needed."
        )

        if age_category == "adult":
            intent = self._classify_parent_intent(query)

            if intent == "shs_child":
                return (
                    MEDICAL_PREFIX
                    + "A parent or caregiver is asking about secondhand smoke and their child's health. "
                    "Answer using only CDC, NHS, or WHO facts about SHS risks to children. "
                    "Be clear, empathetic, and action-oriented — help them understand what they "
                    "can do to protect their child. Maximum 180 words. Plain English. "
                    + response_rules + " "
                    + "End with the free quit line: 1-800-QUIT-NOW (1-800-784-8669)."
                    + f"{rag_block}\n\n"
                    + f"QUESTION: {query}\n\nANSWER:"
                )

            if intent == "cessation":
                return (
                    MEDICAL_PREFIX
                    + "A parent who smokes is asking for help quitting to protect their child from "
                    "secondhand smoke. Answer using only CDC, NHS, or WHO guidelines on cessation. "
                    "Include at least one evidence-based cessation method (e.g. NRT, varenicline, "
                    "behavioural support). Be encouraging, practical, and specific. "
                    "Maximum 180 words. Plain English. "
                    + response_rules + " "
                    "End with: Free confidential support — 1-800-QUIT-NOW (1-800-784-8669)."
                    f"{rag_block}\n\n"
                    f"QUESTION: {query}\n\nANSWER:"
                )

            if intent == "social_boundary":
                return (
                    MEDICAL_PREFIX
                    + "A parent or caregiver needs help asking elders, relatives, or guests not to smoke "
                    "around children without creating conflict. "
                    "Answer every part of the user's question. Give respectful, practical scripts they can "
                    "say out loud. Use the context only for the health-risk facts, but write the advice in "
                    "your own words. Do not begin with a citation, author name, or sentence fragment. "
                    "Short bullets are allowed when they make the answer clearer. Maximum 220 words. Plain English."
                    + " " + response_rules
                    + f"{rag_block}\n\n"
                    + f"QUESTION: {query}\n\nANSWER:"
                )

            # General adult question about smoking/health
            return (
                MEDICAL_PREFIX
                + "Answer this question about smoking and health for an adult. "
                "Use only CDC, NHS, or WHO facts. "
                "If the user asks multiple related questions, answer each part clearly. "
                "Do not begin with a citation, author name, or sentence fragment. "
                + response_rules + " "
                + "Maximum 180 words. Plain English."
                + f"{rag_block}\n\n"
                + f"QUESTION: {query}\n\nANSWER:"
            )

        # teen / child — strict safety mode (unchanged)
        age_label = {
            "teen": "teen aged 14–17",
            "child": "child aged 6–13",
        }.get(age_category, "teen")

        return (
            MEDICAL_PREFIX
            + f"Answer this health question for a {age_label}. "
            f"Only discuss health risks and prevention. "
            f"Never mention alcohol, cannabis, e-cigarettes, or drugs as alternatives. "
            f"Never give instructions on how to smoke. "
            f"Maximum 120 words. Plain English. "
            f"End with: For support call 1-800-QUIT-NOW (free and confidential)."
            f"{rag_block}\n\n"
            f"QUESTION: {query}\n\nANSWER:"
        )

    # ── Output validation hook ────────────────────────────────────────────────

    def validate_output(self, response: str, age_category: str, query: str) -> Dict:
        validation = self.response_validator.validate_response(response, age_category)
        if not validation["is_valid"]:
            return {
                "action": "blocked",
                "original_response": response,
                "final_response": self.response_validator.get_safe_fallback(age_category, query),
                "violations": validation["violations"],
                "adjustments": [],
                "reason": f"Response contained {validation['violation_count']} forbidden terms",
            }
        if validation.get("soft_adjust_needed", False):
            adjustment = self.response_validator.soft_adjust_response(response, age_category)
            return {
                "action": "adjusted",
                "original_response": response,
                "final_response": adjustment["adjusted_response"],
                "violations": [],
                "adjustments": adjustment["adjustments_made"],
                "reason": f"{adjustment['adjustment_count']} modifications for {age_category}",
            }
        if age_category in ["child", "teen"]:
            query_lower = query.lower()
            if any(w in query_lower for w in ["addict", "why", "how does", "what makes"]):
                if "brain" not in response.lower() or "developing" not in response.lower():
                    adjustment = self.response_validator.soft_adjust_response(response, age_category)
                    if adjustment["adjustment_count"] > 0:
                        return {
                            "action": "adjusted",
                            "original_response": response,
                            "final_response": adjustment["adjusted_response"],
                            "violations": [],
                            "adjustments": adjustment["adjustments_made"],
                            "reason": "Added age-specific health warning",
                        }
        return {
            "action": "passed",
            "original_response": response,
            "final_response": response,
            "violations": [],
            "adjustments": [],
            "reason": "Response passed validation",
        }


# ╔╗
# ║                       RESPONSE VALIDATOR                                   ║
# ╚

class ResponseValidator:
    """Validates LLM output against age-based content policies."""

    def __init__(self, content_filter: SmokingContentFilter):
        self.content_filter = content_filter

        self.forbidden_output_terms = {
            "child": [
                "hookah", "shisha", "waterpipe", "cigarette brand", "marlboro", "camel", "newport",
                "vape pen", "juul", "e-cigarette", "smokeless tobacco", "chewing tobacco",
                "rolling tobacco", "pack of cigarettes", "nicotine pouch", "how to smoke",
                "smoking technique", "inhale", "exhale smoke", "light a cigarette",
                "take a drag", "take a puff", "tobacco shop", "smoke shop", "where to buy",
                "buy cigarettes", "try vaping", "use vaping", "switch to vaping",
                "nicotine fix", "nicotine replacement", "get your nicotine",
                "alternative to smoking", "instead of cigarettes",
                "if you smoke", "when you smoke", "if you decide to smoke",
                "if you do smoke", "continue smoking", "keep smoking",
                # SAFETY: Substances as coping strategies
                "cannabis", "marijuana", "weed", "smoke weed", "use weed",
                "alcohol to cope", "drink to cope", "use alcohol", "using alcohol",
                "drugs like cannabis", "drugs like alcohol", "cannabis or alcohol",
                "alcohol or cannabis", "alcohol or marijuana",
            ],
            "teen": [
                "hookah", "shisha", "waterpipe", "cigarette brand", "marlboro", "camel", "newport",
                "vape pen", "juul", "e-cigarette", "smokeless tobacco", "chewing tobacco",
                "rolling tobacco", "nicotine pouch", "hookah method", "shisha setup",
                "how to use hookah", "cigarette brand comparison", "best tobacco",
                "vaping technique", "how to vape", "how to inhale", "how to smoke",
                "smoking technique", "inhale deeply", "light a cigarette",
                "take a drag", "take a puff", "smoke shop location", "where to buy",
                "smoking tips", "tobacco shop", "buy cigarettes", "purchase tobacco",
                "try vaping", "use vaping", "switch to vaping", "nicotine fix",
                "get your nicotine", "nicotine replacement",
                "alternative to smoking", "instead of cigarettes", "smokeless alternative",
                "vaping as alternative", "if you smoke", "when you smoke",
                "if you decide to smoke", "if you do smoke", "continue smoking",
                "keep smoking", "smoke again", "start smoking", "try smoking",
                "reduce your smoking", "smoke less", "cut down on cigarettes",
                "safer way to smoke", "less harmful",
                # SAFETY: Substance coping recommendations
                "cannabis to cope", "marijuana to cope", "weed to cope",
                "use cannabis", "use marijuana", "try cannabis", "try marijuana",
                "smoke weed", "drink alcohol to cope", "alcohol to cope",
                "using cannabis", "using marijuana", "drugs like cannabis",
                "drugs like alcohol", "cannabis or alcohol", "alcohol or cannabis",
                "alcohol or marijuana",
            ],
            "adult": [],
        }

        self.harmful_patterns = {
            "child": [
                "alternative method", "other way to get nicotine",
                "satisfy your craving", "nicotine craving",
            ],
            "teen": [
                "alternative method", "other way to get nicotine",
                "satisfy your craving", "nicotine craving",
                "manage your addiction", "feed your addiction",
            ],
            "adult": [],
        }

        self.soft_remove_phrases = {
            "child": [
                "pleasurable sensations", "pleasurable feelings", "pleasurable nature",
                "feeling of pleasure", "sense of pleasure", "pleasant sensation",
                "relaxation", "relaxing effect", "calming effect", "stress relief",
                "rewarding experience", "rewarding feeling", "feels good",
                "enjoyable", "enjoyment", "satisfying feeling", "satisfaction",
            ],
            "teen": [
                "pleasurable sensations", "pleasurable feelings", "pleasurable nature",
                "feeling of pleasure", "sense of pleasure", "pleasant sensation",
                "periods of rest and relaxation", "relaxation", "relaxing effect",
                "calming effect", "stress relief", "helps you relax",
                "rewarding experience", "rewarding experiences", "rewarding feeling",
                "feels good", "good feeling", "enjoyable", "enjoyment",
                "satisfying feeling", "satisfaction", "pleasant experience",
            ],
            "adult": [],
        }

        self.age_specific_warnings = {
            "child": (
                "\n\n**Important for young people:** Smoking is very dangerous and you should "
                "never try it. Talk to a parent, teacher, or doctor if you have questions. "
                "Adults who want help quitting can call 1-800-QUIT-NOW for free support."
            ),
            "teen": (
                "\n\n**Important for teens:** Your brain is still developing until age 25, "
                "making you MORE vulnerable to nicotine addiction than adults. "
                "Teens who try smoking become addicted faster and more intensely. "
                "90% of adult smokers started before age 18. The best choice is to never start.\n"
                "If you or someone you know needs support, talk to a trusted adult, parent, "
                "doctor, or school counselor. Free confidential help is available 24/7: "
                "**1-800-QUIT-NOW** (1-800-784-8669)."
            ),
            "adult": "",
        }

    def validate_response(self, response: str, age_category: str) -> Dict:
        response_lower = response.lower()
        violations: List[Dict] = []
        soft_adjust_needed = False

        for term in self.forbidden_output_terms.get(age_category, []):
            if term in response_lower:
                # SAFETY: substance-related terms are always high severity
                severity = "high" if any(
                    x in term for x in [
                        "how to", "where to buy", "technique", "try vaping",
                        "switch to", "if you smoke", "decide to smoke",
                        "nicotine fix", "alternative", "smokeless",
                        "cannabis", "marijuana", "weed", "alcohol to cope",
                        "drink to cope", "drugs like",
                    ]
                ) else "medium"
                violations.append({"term": term, "severity": severity, "category": "forbidden_term"})

        for pattern in self.harmful_patterns.get(age_category, []):
            if pattern in response_lower:
                violations.append({"term": pattern, "severity": "high", "category": "harmful_pattern"})

        # SAFETY: dedicated substance-safety contextual scanner
        substance_violations = self.scan_substance_safety(response, age_category)
        violations.extend(substance_violations)

        for phrase in self.soft_remove_phrases.get(age_category, []):
            if phrase in response_lower:
                soft_adjust_needed = True
                break

        is_valid = len(violations) == 0
        return {
            "is_valid": is_valid,
            "violations": violations,
            "violation_count": len(violations),
            "severity": (
                "high" if any(v["severity"] == "high" for v in violations)
                else "medium" if violations else "none"
            ),
            "soft_adjust_needed": soft_adjust_needed,
        }

    def scan_substance_safety(self, response: str, age_category: str) -> List[Dict]:
        """
        SAFETY: Dedicated contextual scanner for dangerous substance recommendations.

        Catches cannabis/alcohol/marijuana recommended as coping tools even when
        phrased in ways that bypass the simple substring forbidden_output_terms list.
        Uses a ±120-char window around each substance mention.
        Adults are not subject to this scan.
        """
        if age_category == "adult":
            return []

        violations: List[Dict] = []
        response_lower = response.lower()

        broad_forbidden = [
            "drugs like cannabis", "drugs like alcohol", "drugs like marijuana",
            "use cannabis", "use marijuana", "try cannabis", "try marijuana",
            "smoke weed", "drink alcohol to cope", "alcohol to cope",
            "cannabis to cope", "marijuana to cope", "weed to cope",
            "using cannabis", "using marijuana",
            "cannabis or alcohol", "alcohol or cannabis", "alcohol or marijuana",
        ]
        for phrase in broad_forbidden:
            if phrase in response_lower:
                violations.append({"term": phrase, "severity": "high", "category": "substance_safety"})

        substances = ["cannabis", "marijuana", "weed", "alcohol", "beer", "wine", "liquor"]
        coping_keywords = [
            "cope", "coping", "withdrawal", "craving", "relief", "help",
            "alternative", "instead", "substitute", "manage", "use",
            "try", "recommend", "suggest", "option", "consider",
        ]

        for substance in substances:
            idx = 0
            while True:
                pos = response_lower.find(substance, idx)
                if pos == -1:
                    break
                window = response_lower[max(0, pos - 120): pos + len(substance) + 120]
                for keyword in coping_keywords:
                    if keyword in window:
                        vkey = f"{substance} near '{keyword}' (contextual)"
                        if not any(v["term"] == vkey for v in violations):
                            violations.append({
                                "term": vkey, "severity": "high", "category": "substance_safety",
                            })
                        break
                idx = pos + 1

        return violations

    def soft_adjust_response(self, response: str, age_category: str) -> Dict:
        adjusted_response = response
        adjustments_made: List[str] = []
        phrases = sorted(self.soft_remove_phrases.get(age_category, []), key=len, reverse=True)
        replaced_positions: set = set()

        for phrase in phrases:
            phrase_lower = phrase.lower()
            response_lower = adjusted_response.lower()
            start = 0
            while True:
                pos = response_lower.find(phrase_lower, start)
                if pos == -1:
                    break
                overlap = any(
                    pos < end and pos + len(phrase) > begin
                    for begin, end in replaced_positions
                )
                if not overlap:
                    replacement = self._get_neutral_replacement(phrase)
                    if replacement:
                        if adjusted_response[pos].isupper():
                            replacement = replacement[0].upper() + replacement[1:]
                        adjusted_response = (
                            adjusted_response[:pos]
                            + replacement
                            + adjusted_response[pos + len(phrase):]
                        )
                        replaced_positions.add((pos, pos + len(replacement)))
                        msg = f"Replaced '{phrase}' with '{replacement}'"
                        if msg not in adjustments_made:
                            adjustments_made.append(msg)
                        response_lower = adjusted_response.lower()
                start = pos + 1

        warning = self.age_specific_warnings.get(age_category, "")
        if warning and "important for" not in adjusted_response.lower():
            adjusted_response = adjusted_response.rstrip() + warning
            adjustments_made.append("Added age-specific health warning")

        return {
            "adjusted_response": adjusted_response,
            "adjustments_made": adjustments_made,
            "adjustment_count": len(adjustments_made),
        }

    def _get_neutral_replacement(self, phrase: str) -> str:
        replacements = {
            "pleasurable sensations": "dopamine responses",
            "pleasurable feelings": "chemical effects",
            "pleasurable nature": "addictive nature",
            "feeling of pleasure": "dopamine release",
            "sense of pleasure": "chemical response",
            "pleasant sensation": "neurological response",
            "feels good": "triggers addiction",
            "good feeling": "chemical dependence",
            "enjoyable": "addictive",
            "enjoyment": "chemical dependence",
            "satisfying feeling": "craving relief",
            "satisfaction": "temporary craving relief",
            "pleasant experience": "addictive process",
            "relaxation": "temporary stress masking",
            "relaxing effect": "short-term relief",
            "calming effect": "brief anxiety suppression",
            "stress relief": "temporary craving satisfaction",
            "helps you relax": "temporarily masks stress",
            "periods of rest and relaxation": "addiction-reinforcing breaks",
            "rewarding experience": "dopamine-driven cycle",
            "rewarding experiences": "addiction reinforcement",
            "rewarding feeling": "diminishing dopamine response",
        }
        return replacements.get(phrase.lower(), "")

    def get_safe_fallback(self, age_category: str, query: str) -> str:
        query_lower = query.lower()
        if age_category == "child":
            if "addict" in query_lower:
                return (
                    "Smoking is very addictive — once someone starts, it's very hard to stop. "
                    "The chemicals in cigarettes trick your brain into wanting more. "
                    "That's why the best choice is to never start smoking at all. "
                    "If you have questions, talk to a parent, teacher, or doctor. "
                    "Adults who want help quitting can call 1-800-QUIT-NOW (1-800-784-8669)."
                )
            return (
                "Smoking is very harmful to your body. It can hurt your lungs, heart, "
                "and make it hard to breathe. The best choice is to never start smoking. "
                "If you have questions, please talk to a parent, teacher, or doctor. "
                "Adults who want help quitting can call 1-800-QUIT-NOW (1-800-784-8669)."
            )
        if age_category == "teen":
            if "addict" in query_lower:
                return (
                    "Nicotine affects the brain's reward system. When nicotine enters the body, "
                    "it triggers dopamine release. Over time the brain becomes dependent on "
                    "nicotine to feel normal, leading to addiction.\n\n"
                    "Key facts:\n"
                    "- 90% of adult smokers started before age 18\n"
                    "- Teen brains are MORE vulnerable to addiction than adult brains\n"
                    "- Most people who try cigarettes become addicted\n\n"
                    "The best protection is to never start. For support, talk to a trusted "
                    "adult, doctor, or school counselor, or call 1-800-QUIT-NOW (1-800-784-8669)."
                )
            return (
                "Smoking and tobacco products are highly addictive and cause serious diseases "
                "including lung cancer, heart disease, and respiratory problems. "
                "All tobacco and nicotine products (including vaping) are harmful and illegal "
                "for people under 18. If you need help resisting peer pressure or quitting, "
                "talk to a trusted adult, parent, doctor, or school counselor. "
                "Free confidential support 24/7: 1-800-QUIT-NOW (1-800-784-8669)."
            )
        return "Please ask an age-appropriate health education question."


# ╔╗
# ║         CRITICAL FIX — HALLUCINATION GUARD (Steps 6 & 7)                  ║
# ║  Checks every generated response for:                                      ║
# ║    • Non-Latin / garbled characters (Step 6 — encoding guard)             ║
# ║    • Known physiologically incorrect claims (Step 7)                       ║
# ║    • Self-contradictory physiological statements (Step 7)                  ║
# ╚

class HallucinationGuard:
    """
    Detects hallucinations and encoding errors in LLM-generated responses.

    This is a rule-based first pass BEFORE the Ollama fact-checker call.
    Fast, zero-latency, catches the most dangerous failure modes:
      - Non-English / garbled output (model went off-rails)
      - Known incorrect medical claims (e.g. nicotine decreases heart rate)
      - Self-contradictions within the same response
    """

    # Known physiologically incorrect claims about nicotine/tobacco.
    # Each tuple is (regex_pattern, human-readable description).
    KNOWN_INCORRECT_FACTS: List[Tuple[str, str]] = [
        (
            r"nicotine\s+(?:decreases?|lowers?|reduces?|slows?)\s+(?:your\s+)?heart\s+rate",
            "Incorrect: nicotine INCREASES heart rate (not decreases it)",
        ),
        (
            r"nicotine\s+(?:decreases?|lowers?|reduces?)\s+blood\s+pressure",
            "Incorrect: nicotine raises blood pressure short-term (not lowers it)",
        ),
        (
            r"(?:absorbed?|enters?|passes?)\s+(?:through|via|by)\s+(?:the\s+)?pineal\s+gland",
            "Incorrect anatomy: nicotine is NOT absorbed via the pineal gland "
            "(it enters via lungs, skin, or oral mucosa)",
        ),
        (
            r"smoking\s+(?:improves?|enhances?|boosts?|increases?)\s+"
            r"(?:lung|respiratory)\s+(?:function|capacity|health|performance)",
            "Incorrect: smoking DAMAGES lung function, does not improve it",
        ),
        (
            r"carbon\s+monoxide\s+(?:helps?|aids?|improves?|enhances?|boosts?)\s+"
            r"(?:oxygen|O2)\s+(?:transport|absorption|delivery|uptake)",
            "Incorrect: CO BLOCKS oxygen transport via carboxyhemoglobin formation",
        ),
        (
            r"tar\s+is\s+(?:absorbed?|processed|digested)\s+"
            r"(?:through|via|by|in)\s+(?:the\s+)?stomach",
            "Incorrect anatomy: tobacco tar is deposited in the lungs, not the stomach",
        ),
        (
            r"nicotine\s+(?:cures?|treats?|prevents?|protects?\s+against)\s+"
            r"(?:cancer|lung\s+cancer|tumou?rs?|disease)",
            "Incorrect: nicotine does not cure or prevent cancer",
        ),
        (
            r"second\s*hand\s+smoke\s+(?:is\s+)?(?:harmless|safe|not\s+harmful|not\s+dangerous|benign)",
            "Incorrect: secondhand smoke is a known carcinogen per WHO/CDC/NHS",
        ),
        # ── SHS-specific incorrect facts ─────────────────────────────────────
        (
            r"second\s*hand\s+smoke\s+(?:is\s+)?(?:only\s+)?dangerous\s+(?:indoors|inside)",
            "Incorrect: SHS is harmful both indoors AND outdoors — there is no safe level "
            "of exposure (CDC/WHO); outdoor exposure near children also poses risk.",
        ),
        (
            r"(?:opening\s+(?:a\s+)?window|ventilation|air\s+(?:filter|purifier)|fan)\s+"
            r"(?:removes?|eliminates?|clears?|gets?\s+rid\s+of|protects?\s+(?:against|from))\s+"
            r"(?:all\s+)?second\s*hand\s+smoke",
            "Incorrect: ventilation and air filters do NOT eliminate SHS — only smoke-free "
            "environments fully protect children (CDC/Surgeon General).",
        ),
        (
            r"smoking\s+(?:outside|outdoors|in\s+(?:the\s+)?garden|on\s+(?:the\s+)?balcony)\s+"
            r"(?:is\s+)?(?:completely\s+)?(?:safe|harmless|fine|ok)\s+(?:for|around)\s+"
            r"(?:children|kids|babies|infants|child)",
            "Incorrect: thirdhand smoke residue on clothing and surfaces still exposes "
            "children even when smoking is done outdoors (CDC).",
        ),
        (
            r"(?:e\-?cigarettes?|vap(?:e|ing)|electronic\s+cigarettes?)\s+"
            r"(?:produce[sd]?|emit[sd]?|release[sd]?|creat(?:e[sd]?|ing))\s+"
            r"(?:only\s+)?(?:harmless\s+(?:water\s+)?vapou?r|water\s+vapou?r|clean\s+air)",
            "Incorrect: e-cigarette aerosol contains nicotine, ultrafine particles, and "
            "toxic chemicals — it is NOT simply water vapour (CDC/NHS).",
        ),
        (
            r"(?:children|kids|infants|babies)\s+(?:are\s+)?(?:not\s+)?(?:less|more)\s+"
            r"(?:affected|harmed|impacted|susceptible|vulnerable)\s+(?:by|to)\s+"
            r"second\s*hand\s+smoke\s+than\s+adults",
            "Incorrect: children are MORE vulnerable to SHS than adults — their lungs and "
            "immune systems are still developing (WHO/CDC).",
        ),
        (
            r"(?:asthma|wheez(?:e|ing)|respiratory\s+illness)\s+(?:in\s+children\s+)?(?:is\s+)?"
            r"(?:not\s+)?(?:caused|triggered|worsened|exacerbated)\s+by\s+"
            r"second\s*hand\s+smoke",
            "Incorrect: SHS is a well-established trigger for asthma attacks and respiratory "
            "illness in children (CDC/NHS/WHO).",
        ),
    ]

    # Pairs of contradictory claims.  If BOTH appear in the same response, it is flagged.
    # Each tuple is (pattern_a, pattern_b, description).
    CONTRADICTION_PAIRS: List[Tuple[str, str, str]] = [
        (
            r"increases?\s+(?:the\s+)?heart\s+rate",
            r"decreases?\s+(?:the\s+)?heart\s+rate",
            "Contradictory heart-rate claims in same response",
        ),
        (
            r"raises?\s+(?:blood\s+pressure|bp)",
            r"lowers?\s+(?:blood\s+pressure|bp)",
            "Contradictory blood-pressure claims in same response",
        ),
        (
            r"(?:causes?|increases?\s+(?:the\s+)?risk\s+of)\s+(?:cancer|lung\s+cancer)",
            r"(?:prevents?|reduces?\s+(?:the\s+)?risk\s+of)\s+(?:cancer|lung\s+cancer)",
            "Contradictory cancer-risk claims in same response",
        ),
        (
            r"(?:harmful|dangerous|toxic)\s+(?:to|for)\s+(?:the\s+)?(?:lungs|body|health)",
            r"(?:beneficial|helpful|good)\s+(?:for|to)\s+(?:the\s+)?(?:lungs|body|health)",
            "Contradictory health-effect claims in same response",
        ),
        (
            r"damages?\s+(?:the\s+)?(?:lungs|airways|respiratory\s+system)",
            r"improves?\s+(?:the\s+)?(?:lungs|airways|respiratory\s+system)",
            "Contradictory respiratory claims in same response",
        ),
    ]

    def check_encoding(self, text: str) -> Tuple[bool, str]:
        """
        Step 6 — Language/Encoding Guard.

        English medical text should only contain ASCII + extended Latin characters.
        Arabic, CJK, Cyrillic, etc. appearing in the response indicates the model
        has gone off-rails and the response must be discarded and regenerated.

        Returns (is_clean: bool, detail: str).
        """
        non_latin_chars: List[str] = []
        for char in text:
            cp = ord(char)
            # Allow basic ASCII (0-127) and Latin Extended A/B (0x80–0x024F)
            if cp <= 0x024F:
                continue
            # Allow common typographic characters (em-dash, curly quotes, bullet, degree)
            if cp in (
                0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D,
                0x2026, 0x2022, 0x00B0, 0x00B1, 0x00B2, 0x00B3,
            ):
                continue
            # Flag any non-Latin letter (Arabic, CJK, Cyrillic, etc.)
            cat = unicodedata.category(char)
            if cat.startswith("L"):
                non_latin_chars.append(char)

        if non_latin_chars:
            unique = list(dict.fromkeys(non_latin_chars))[:10]
            return False, f"Non-Latin characters detected: {''.join(unique)}"
        return True, ""

    def check_known_incorrect_facts(self, text: str) -> List[str]:
        """
        Step 7 — Hallucination Guard (known-bad-fact check).

        Scans for regex patterns that match physiologically incorrect claims.
        Returns a list of human-readable error descriptions.
        """
        text_lower = text.lower()
        errors: List[str] = []
        for pattern, description in self.KNOWN_INCORRECT_FACTS:
            if re.search(pattern, text_lower):
                errors.append(description)
        return errors

    def check_contradictions(self, text: str) -> List[str]:
        """
        Step 7 — Hallucination Guard (self-contradiction check).

        Detects responses that make logically contradictory physiological claims
        about the same effect in the same response.
        """
        text_lower = text.lower()
        found: List[str] = []
        for pat_a, pat_b, description in self.CONTRADICTION_PAIRS:
            if re.search(pat_a, text_lower) and re.search(pat_b, text_lower):
                found.append(description)
        return found

    def run(self, text: str) -> Dict:
        """
        Run all hallucination checks and return a combined result dict.
        Encoding check is separated so Step 6 and Step 7 can log independently.
        """
        encoding_ok, encoding_detail = self.check_encoding(text)
        incorrect_facts = self.check_known_incorrect_facts(text)
        contradictions = self.check_contradictions(text)

        all_issues = (
            ([f"Encoding: {encoding_detail}"] if not encoding_ok else [])
            + incorrect_facts
            + contradictions
        )
        return {
            "has_issues": len(all_issues) > 0,
            "issues": all_issues,
            "encoding_ok": encoding_ok,
            "encoding_detail": encoding_detail,
            "incorrect_facts": incorrect_facts,
            "contradictions": contradictions,
        }


# ╔╗
# ║         UPGRADE 2 — FACT-CHECKING LAYER (Steps 8 & 9)                     ║
# ╚

class FactChecker:
    """
    Second Ollama LLM call that acts as an independent medical fact-checker.

    - Sends the generated response to a second Ollama call with a reviewer prompt.
    - Parses PASS / FAIL from the response.
    - If FAIL: triggers a third Ollama call to rewrite with accurate information.
    - Logs all failures to fact_check_failures.log.
    - On any Ollama error: defaults to PASS (fail-open) and logs to pipeline_errors.log.
    """

    # MODEL SWITCH: short single-task prompts for qwen2.5:3b
    CHECKER_PROMPT_TEMPLATE = (
        MEDICAL_PREFIX
        + "Review this health response about secondhand smoke or smoking cessation. "
        "Does it contain any of these: "
        "invented medical terms, contradictory statements, non-English text, "
        "repeated paragraphs, or behavior change stages used as biology? "
        "Reply with PASS or FAIL and one sentence reason only.\n\n"
        "RESPONSE:\n{response}"
    )

    # Audience-aware rewrite template — {audience} is substituted per call
    REWRITE_PROMPT_TEMPLATE = (
        MEDICAL_PREFIX
        + "Rewrite this in plain English for {audience}. "
        "Maximum 150 words. Only include medically accurate facts. "
        "End with: For support call 1-800-QUIT-NOW (free and confidential).\n\n"
        "ISSUES TO FIX:\n{issues}\n\n"
        "ORIGINAL:\n{response}\n\n"
        "REWRITE:"
    )

    # Audience labels per age tier
    _AUDIENCE_LABELS: Dict[int, str] = {
        1: "a young person",
        2: "a teen aged 14–17",
        3: "a parent or adult caregiver concerned about their child's health",
    }

    def check(self, question: str, response: str, model: str) -> Dict:
        """
        Run the fact-checking Ollama call.

        # MODEL SWITCH: num_predict=60, num_ctx=512 (short answer expected: PASS/FAIL + 1 sentence)
        Returns dict with keys:
          result   — "PASS" or "FAIL"
          issues   — list of flagged claim strings (empty if PASS)
        """
        prompt = self.CHECKER_PROMPT_TEMPLATE.format(response=response)
        try:
            # Change 5 — Speed Fix 5: strict token limit for fact-check call
            fc_text = call_ollama(prompt, model, temperature=0.1, num_predict=60, num_ctx=512)
            if not fc_text:
                # Fallback: default to PASS if Ollama unavailable
                log_pipeline_error("FactChecker: empty response from Ollama — defaulting to PASS")
                return {"result": "PASS", "issues": []}

            if re.match(r"^\s*pass\b", fc_text, re.IGNORECASE):
                return {"result": "PASS", "issues": []}

            if re.search(r"\bfail\b", fc_text, re.IGNORECASE):
                issues = re.findall(r"\d+\.\s+(.+)", fc_text)
                if not issues:
                    # FAIL stated but no numbered list — use whole response as one issue
                    issues = [fc_text.strip()]
                return {"result": "FAIL", "issues": issues}

            # Ambiguous response — default to PASS
            log_pipeline_error(
                f"FactChecker: ambiguous response '{fc_text[:80]}' — defaulting to PASS"
            )
            return {"result": "PASS", "issues": []}

        except Exception as e:
            log_pipeline_error(f"FactChecker.check exception: {e}")
            return {"result": "PASS", "issues": []}

    def rewrite(self, response: str, issues: List[str], model: str, age_tier: int = 3) -> str:
        """
        Rewrite a FAIL response using a third Ollama call.

        age_tier controls which audience label is injected into the rewrite prompt:
          tier 2 → "a teen aged 14–17"
          tier 3 → "a parent or adult caregiver concerned about their child's health"

        # MODEL SWITCH: num_predict=200, num_ctx=512
        If the rewrite call fails, returns the original response with a safety note.
        """
        audience = self._AUDIENCE_LABELS.get(age_tier, "an adult")
        issues_text = "\n".join(f"- {i}" for i in issues)
        prompt = self.REWRITE_PROMPT_TEMPLATE.format(
            audience=audience,
            issues=issues_text,
            response=response,
        )
        try:
            # Change 5 — Speed Fix 5: strict token limit for rewrite call
            rewritten = call_ollama(prompt, model, temperature=0.1, num_predict=350, num_ctx=768)
            if rewritten:
                # Strip any echoed prompt artifacts (model sometimes includes the
                # ISSUES TO FIX / ORIGINAL / REWRITE: header text in its output)
                for marker in ("REWRITE:", "REWRITE :"):
                    idx = rewritten.upper().rfind(marker.upper())
                    if idx != -1:
                        candidate = rewritten[idx + len(marker):].strip()
                        if candidate:
                            rewritten = candidate
                            break
                # Also strip any leading "ISSUES TO FIX" or "ORIGINAL:" lines
                lines = rewritten.splitlines()
                clean_lines = []
                skip_prefixes = ("issues to fix", "original:", "original response")
                for line in lines:
                    if any(line.strip().lower().startswith(p) for p in skip_prefixes):
                        continue
                    clean_lines.append(line)
                rewritten = "\n".join(clean_lines).strip()
                if rewritten:
                    return rewritten
            log_pipeline_error("FactChecker.rewrite: empty response — returning original with note")
        except Exception as e:
            log_pipeline_error(f"FactChecker.rewrite exception: {e}")

        # Fallback: return original with disclaimer
        return (
            response
            + "\n\n*Note: This information should be verified with a healthcare provider or "
            "by calling 1-800-QUIT-NOW.*"
        )


# ╔╗
# ║         UPGRADE 3 — SAFETY FILTER (Steps 10 & 11, ages ≤17 only)         ║
# ╚

class SafetyFilter:
    """
    Third Ollama LLM call — child safety review for users aged 17 and under.

    - Skipped entirely for users 18+ (tier 3).
    - Sends the response to Ollama with a safety reviewer prompt.
    - If SAFE: response passes through unchanged.
    - If UNSAFE: Ollama rewrites the response in age-appropriate English and
      appends the helpline.
    - Logs all rewrites to safety_rewrites.log.
    - On any Ollama error: defaults to SAFE (fail-open) and logs to pipeline_errors.log.
    """

    # MODEL SWITCH: short single-task safety prompt for qwen2.5:3b
    # Only asks SAFE or UNSAFE — no rewrite requested (rewrite done separately if needed)
    SAFETY_PROMPT_TEMPLATE = (
        MEDICAL_PREFIX
        + "Review this response for a {user_age} year old. "
        "Does it mention alcohol, drugs, cannabis, e-cigarettes, or graphic medical content? "
        "Reply SAFE or UNSAFE only.\n\n"
        "RESPONSE:\n{response}"
    )

    def check(self, response: str, user_age: int, model: str) -> Dict:
        """
        Run the safety filter Ollama call.

        # MODEL SWITCH: num_predict=20, num_ctx=512 (expects SAFE or UNSAFE only)
        Returns dict with keys:
          result    — "SAFE" or "UNSAFE"
          rewritten — rewritten text if UNSAFE, None if SAFE
        """
        prompt = self.SAFETY_PROMPT_TEMPLATE.format(
            user_age=user_age,
            response=response,
        )
        try:
            # Change 5 — Speed Fix 5: strict token limit (SAFE/UNSAFE is 1 word)
            sf_text = call_ollama(prompt, model, temperature=0.1, num_predict=20, num_ctx=512)
            if not sf_text:
                log_pipeline_error("SafetyFilter: empty response from Ollama — defaulting to SAFE")
                return {"result": "SAFE", "rewritten": None}

            if re.match(r"^\s*safe\b", sf_text, re.IGNORECASE):
                return {"result": "SAFE", "rewritten": None}

            # UNSAFE — request a rewrite via a separate call
            rewrite_prompt = (
                MEDICAL_PREFIX
                + f"Rewrite this response for a {user_age} year old. "
                "Remove all mentions of alcohol, drugs, cannabis, e-cigarettes, and graphic content. "
                "Maximum 100 words. Plain English. "
                "End with: For support call 1-800-QUIT-NOW (free and confidential).\n\n"
                f"ORIGINAL:\n{response}\n\nREWRITE:"
            )
            rewritten = call_ollama(
                rewrite_prompt, model, temperature=0.1, num_predict=200, num_ctx=512
            )
            # Strip any leading "UNSAFE:" preamble the LLM might add
            rewritten = re.sub(r"(?i)^unsafe[\s:\-–—]*", "", (rewritten or "")).strip()
            if not rewritten:
                rewritten = response  # fallback: keep original if rewrite fails
            return {"result": "UNSAFE", "rewritten": rewritten}

        except Exception as e:
            log_pipeline_error(f"SafetyFilter.check exception: {e}")
            return {"result": "SAFE", "rewritten": None}


# ╔╗
# ║         UPGRADE 4 — CONFIDENCE CALCULATOR (Step 3)                        ║
# ╚

class ConfidenceCalculator:
    """
    Calculates a confidence level from the Pinecone RAG retrieval results.

    Rules:
      - 0 chunks or avg score < 0.5           → LOW    (🔴)
      - ≥2 chunks and 0.5 ≤ avg score ≤ 0.75 → MEDIUM (🟡)
      - ≥2 chunks and avg score > 0.75        → HIGH   (🟢)
    """

    def calculate(self, rag_matches: List[Dict]) -> Dict:
        n = len(rag_matches)
        if n == 0:
            return self._result("LOW", 0.0, 0)

        scores = [m.get("score", 0.0) for m in rag_matches]
        avg = sum(scores) / n

        if n < 2 or avg < 0.5:
            level = "LOW"
        elif avg <= 0.75:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return self._result(level, avg, n)

    @staticmethod
    def _result(level: str, avg_score: float, chunks: int) -> Dict:
        badges = {
            "HIGH":   "🟢 **High confidence** — based on verified documents",
            "MEDIUM": "🟡 **Medium confidence** — answer may be partial",
            "LOW":    "🔴 **Low confidence** — We're not fully sure about this. "
                      "Please verify with a doctor or call 1-800-QUIT-NOW",
        }
        return {
            "level": level,
            "avg_score": round(avg_score, 3),
            "chunks": chunks,
            "badge": badges[level],
        }


# ╔╗
# ║         STEPS 2 & 4 — PINECONE SEARCH + RAG GENERATION (Mukesh)           ║
# ╚

class RAGKnowledgeLayer:
    """
    Wraps Mukesh's SemanticSearch + Ollama pipeline.
    Falls back gracefully when Pinecone / torch are unavailable.

    CRITICAL FIX: generate_answer() now accepts a temperature parameter
    defaulting to 0.1 (instead of 0.7) to reduce hallucination risk on
    medical content.
    """

    PINECONE_API_KEY = os.environ.get(
        "PINECONE_API_KEY",
        "pcsk_5U97iG_CqwvK14qpUy8qQze1WJg53KGQ5TMozchuw6dqZNds1fEPpQwxPpZcfKPAXfK4CV",
    )
    INDEX_NAME = "chatbot"
    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self):
        self._index = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._ready = False
        self._error: Optional[str] = None
        self._init()

    def _init(self):
        if not (_pinecone_ok and _torch_ok):
            self._error = (
                "Pinecone / PyTorch not installed — RAG layer disabled; "
                "LLM will use parametric memory only."
            )
            return
        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModel.from_pretrained(self.MODEL_NAME).to(self._device)
            self._model.eval()
            pc = _PineconeClient(api_key=self.PINECONE_API_KEY)
            self._index = pc.Index(self.INDEX_NAME)
            self._ready = True
            logger.info("RAG knowledge layer initialised (Pinecone + %s).", self.MODEL_NAME)
        except Exception as e:
            self._error = str(e)
            logger.warning("RAG layer init failed: %s", e)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error_message(self) -> Optional[str]:
        return self._error

    # ── Step 2 — Semantic Search ──────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5, age_category: str = "adult") -> List[Dict]:
        if not self._ready:
            return []
        try:
            clean = self._clean_query(query)
            embedding = self._get_embedding(clean)
            meta_filter = {"age_relevance": {"$in": ["all", age_category]}}
            results = self._index.query(
                vector=embedding,
                top_k=top_k,
                filter=meta_filter,
                include_metadata=True,
            )
            matches = results.get("matches", [])
            # Fallback: if strict age filter returns nothing, widen to all ages.
            if not matches:
                results = self._index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                )
                matches = results.get("matches", [])
            source_boost = {"CDC": 0.05, "NIH": 0.05, "WHO": 0.03, "PubMed": 0.02}
            for m in matches:
                src = m.get("metadata", {}).get("verified_source", "")
                m["score"] = m.get("score", 0) + source_boost.get(src, 0)
            matches = sorted(matches, key=lambda x: x.get("score", 0), reverse=True)
            formatted = []
            for m in matches:
                if m.get("score", 0) >= 0.35:
                    meta = m.get("metadata", {})
                    formatted.append({
                        "id": m.get("id"),
                        "score": m.get("score"),
                        "source": meta.get("source", "Unknown"),
                        "text": meta.get("text", ""),
                        "verified_source": meta.get("verified_source", ""),
                        "topic": meta.get("topic", ""),
                        "age_relevance": meta.get("age_relevance", "all"),
                    })
            return formatted[:3]
        except Exception as e:
            logger.warning("Pinecone search failed: %s", e)
            return []

    def _get_embedding(self, text: str) -> List[float]:
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        embedding = outputs.last_hidden_state[:, 0].squeeze().cpu().numpy().tolist()
        if not isinstance(embedding, list):
            embedding = [float(x) for x in embedding]
        if len(embedding) > 1024:
            embedding = embedding[:1024]
        return [float(f"{x:.5f}") for x in embedding]

    def _clean_query(self, query: str) -> str:
        query = re.sub(r"[?]", "", query.lower().strip())
        expansions = {
            "shs": "secondhand smoke",
            "ets": "environmental tobacco smoke",
            "second hand smoke": "secondhand smoke",
            "passive smoke": "secondhand smoke",
            "passive smoking": "secondhand smoke",
            "thirdhand smoke": "thirdhand smoke residue",
            "third hand smoke": "thirdhand smoke residue",
            "third-hand smoke": "thirdhand smoke residue",
            "sids": "sudden infant death syndrome",
            "cot death": "sudden infant death syndrome",
            "nrt": "nicotine replacement therapy",
            "cold turkey": "abrupt cessation",
            "vaping": "e-cigarette aerosol",
            "e-cig": "e-cigarette",
            "ecig": "e-cigarette",
        }
        for abbr, exp in expansions.items():
            query = query.replace(abbr, exp)
        return query

    def build_rag_context(self, matches: List[Dict]) -> str:
        if not matches:
            return ""
        lines = []
        for i, m in enumerate(matches, 1):
            src = m.get("source", "Unknown")
            score = m.get("score", 0)
            text = self._sanitize_context_snippet((m.get("text") or "").strip())[:600]
            lines.append(f"[{i}] Source: {src} (relevance: {score:.3f})\n{text}")
        return "\n\n".join(lines)

    def _sanitize_context_snippet(self, text: str) -> str:
        """Trim retrieval snippets so the model does not continue a broken fragment."""
        if not text:
            return ""

        cleaned = re.sub(r"\s+", " ", text).strip()
        if re.match(r"^(and|or|but|because|which|that)\b", cleaned, re.IGNORECASE):
            sentence_break = re.search(r"[.!?]\s+", cleaned)
            if sentence_break:
                cleaned = cleaned[sentence_break.end():].strip()

        cleaned = re.sub(r"\(([^)]*et al\.,?\s*(?:19|20)\d{2}[^)]*)\)", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -:;,")
        return cleaned

    # ── Step 5 — RAG Answer Generation ───────────────────────────────────────

    def generate_answer(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.1,
        stream_placeholder=None,
        num_predict: int = 250,
    ) -> str:
        """
        Send the age-aware prompt to Ollama.

        # MODEL SWITCH: defaults to MODEL constant (qwen2.5:3b)
        # Change 3: num_ctx=1024, num_predict=250, repeat_penalty=1.3
        # Speed Fix 4: streaming display via stream_placeholder (st.empty() widget)
        # Change 4: retry with num_ctx=512 if first call returns empty
        temperature defaults to 0.1 (medical content — low hallucination risk).
        Caller may pass temperature=0.05 for low-confidence queries.
        """
        if model is None:
            model = MODEL
        if _openai_client is None:
            return "Error: OPENAI_API_KEY is not configured."

        try:
            use_stream = stream_placeholder is not None
            if use_stream:
                stream = _openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": MEDICAL_PREFIX.strip()},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=num_predict,
                    stream=True,
                    timeout=120,
                )
                full_text = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        full_text += delta
                        if stream_placeholder is not None:
                            stream_placeholder.markdown(full_text + " ")
                if stream_placeholder is not None:
                    stream_placeholder.markdown(full_text.strip())
                return full_text.strip()

            response = _openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MEDICAL_PREFIX.strip()},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=num_predict,
                timeout=120,
            )
            content = response.choices[0].message.content if response.choices else ""
            if content:
                return content.strip()
            return f"Error: No response received from {model}."
        except Exception as e:
            log_pipeline_error(f"generate_answer exception: {e}")
            return f"Error connecting to OpenAI: {e}"

    @staticmethod
    def list_ollama_models() -> List[str]:
        if not OPENAI_API_KEY:
            return []
        return [MODEL]


# ╔╗
# ║                     LEGACY RESEARCH LOGGING                                ║
# ╚

def log_interaction(
    age: int,
    query: str,
    response: str,
    blocked: bool,
    action: str = "passed",
    reason: str = "",
    rag_sources: Optional[List[Dict]] = None,
    retrieval_scores: Optional[List[float]] = None,
) -> None:
    """Legacy JSON logger — kept for compatibility with existing research tooling."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "age": age,
        "query": query,
        "response_preview": response[:200],
        "blocked": blocked,
        "action": action,
        "reason": reason,
        "rag_sources": [s.get("source", "") for s in (rag_sources or [])],
        "retrieval_scores": retrieval_scores or [],
    }
    logger.info("RESEARCH_LOG: %s", json.dumps(entry))


def build_recent_conversation_context(
    history: List[Dict[str, str]],
    max_messages: int = 6,
) -> str:
    """Return recent chat turns to help the model answer follow-up questions."""
    if not history:
        return ""

    recent_messages = history[-max_messages:]
    lines = []
    for msg in recent_messages:
        role = "User" if msg.get("role") == "user" else "Assistant"
        text = (msg.get("text") or "").strip()
        if text:
            lines.append(f"{role}: {text[:400]}")
    return "\n".join(lines)


def is_follow_up_query(query: str) -> bool:
    """Return True when a query appears to depend on previous conversation context."""
    q = (query or "").strip().lower()
    if not q:
        return False

    follow_up_patterns = [
        r"^what about",
        r"^and what about",
        r"^can you explain",
        r"^tell me more",
        r"^more details",
        r"^why\?*$",
        r"^how\?*$",
        r"^then what",
        r"\bthat\b",
        r"\bthis\b",
        r"\bit\b",
        r"\bthose\b",
        r"\bthem\b",
    ]
    return any(re.search(pat, q) for pat in follow_up_patterns)


def strip_response_headings(text: str) -> str:
    """Remove markdown heading lines so responses render without large titles."""
    if not text:
        return text

    cleaned_lines: List[str] = []
    for line in text.splitlines():
        # Remove markdown ATX heading markers (e.g., '# Title', '## Title').
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            cleaned_lines.append(re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip())
        else:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def strip_role_labels(text: str) -> str:
    """Remove echoed chat role labels from model output."""
    if not text:
        return text

    lines = []
    for line in text.splitlines():
        # Drop standalone role label lines.
        if re.match(r"^\s*(user|assistant|system)\s*:\s*$", line, re.IGNORECASE):
            continue
        # Strip leading role labels from content lines.
        cleaned_line = re.sub(
            r"^\s*(user|assistant|system)\s*:\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        lines.append(cleaned_line)

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def clean_generated_response_artifacts(text: str) -> str:
    """Remove obvious retrieval/citation fragments that leak into model output."""
    if not text:
        return text

    def _is_leading_fragment(line: str) -> bool:
        compact = line.strip()
        if not compact:
            return False
        if re.match(r"^(and|or|but|because|which|that)\b", compact, re.IGNORECASE) and re.search(
            r"et al\.,?\s*(?:19|20)\d{2}|\([^)]*(?:19|20)\d{2}[^)]*\)",
            compact,
            re.IGNORECASE,
        ):
            return True
        if re.match(r"^[\"'“‘’(]*[A-Z][A-Za-z-]+ et al\.,?\s*(?:19|20)\d{2}", compact):
            return True
        return False

    def _is_trailing_fragment(line: str) -> bool:
        compact = line.strip()
        if not compact:
            return False
        if compact in {"By", "For example", "For instance", "Such as"}:
            return True
        if len(compact.split()) <= 2 and not re.search(r"[.!?]$", compact):
            return True
        return False

    lines = [line.rstrip() for line in text.splitlines()]
    while lines and _is_leading_fragment(lines[0]):
        lines.pop(0)
    while lines and _is_trailing_fragment(lines[-1]):
        lines.pop()

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def strip_leading_followup_questions(text: str) -> str:
    """Remove leading interviewer-style question lines from model output."""
    if not text:
        return text

    lines = text.splitlines()
    kept = []
    skipping = True
    for line in lines:
        s = line.strip()
        if skipping and s:
            if re.match(r"^(what|how|has|have|do|does|did|can|could|would|is|are)\b.*\?$", s, re.IGNORECASE):
                continue
            skipping = False
        kept.append(line)

    cleaned = "\n".join(kept).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def build_rag_evidence_block(matches: List[Dict], max_items: int = 5) -> str:
    """Create compact evidence lines from retrieved chunks."""
    if not matches:
        return ""
    lines: List[str] = []
    for i, m in enumerate(matches[:max_items], 1):
        source = (m.get("verified_source") or m.get("source") or "Unknown").strip()
        score = float(m.get("score", 0.0))
        snippet = re.sub(r"\s+", " ", (m.get("text") or "").strip())[:420]
        if not snippet:
            continue
        lines.append(f"[{i}] {source} | relevance={score:.3f} | {snippet}")
    return "\n".join(lines)


def build_shs_structured_prompt(
    query: str,
    evidence_block: str,
    age_tier: int,
    conversation_context: str = "",
) -> str:
    """Prompt for RAG-first, empathy-first natural response flow."""
    audience = "adult parent/caregiver" if age_tier == 3 else "teen user"
    emotional_signal = bool(
        re.search(
            r"\b(guilty|ashamed|scared|afraid|worried|anxious|stressed|overwhelmed|upset|hopeless|failed)\b",
            (query or "").lower(),
        )
    )
    empathy_instruction = (
        "The user shows emotional distress. Start with one validating sentence "
        "that acknowledges feelings without blame."
        if emotional_signal
        else "Use a warm, supportive tone without sounding clinical or robotic."
    )
    convo = (
        f"\nConversation context (continuity only):\n{conversation_context}\n"
        if conversation_context else ""
    )
    return (
        "Answer the SHS question using ONLY the EVIDENCE BLOCK facts.\n"
        "Style and behavior rules:\n"
        f"- {empathy_instruction}\n"
        "- Non-judgmental language only. Do not lecture or blame.\n"
        "- Avoid rigid section headers and avoid repeating the same point twice.\n"
        "- Use natural 2-4 short paragraphs.\n"
        "- Keep content ratio about 30% health explanation and 70% actionable guidance.\n"
        "- Give specific micro-actions where relevant (example: smoke only outside, no smoking in car, wash hands/change outer layer after smoking, delay first cigarette, contact quitline 1-800-QUIT-NOW).\n"
        "- If evidence is missing for part of the question, say this softly in one sentence without mentioning system or knowledge-base limitations.\n"
        "- No citations, author names, raw retrieval fragments, or internal system wording.\n"
        "- No new medical facts beyond evidence.\n\n"
        f"Audience: {audience}\n"
        f"User question: {query}\n"
        f"{convo}\n"
        f"EVIDENCE BLOCK:\n{evidence_block}\n\n"
        "Write the final response now."
    )


def normalize_query_for_cache(query: str) -> str:
    """Normalize query text to make cache validation strict and deterministic."""
    q = (query or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def looks_incomplete_response(text: str) -> bool:
    """Heuristic check for responses that end abruptly."""
    if not text:
        return True

    trimmed = text.strip()
    if len(trimmed) < 40:
        return True

    # Common signs of truncation.
    if re.search(r"\b(to|for|with|about|and|or|because|including|such as)\s*$", trimmed, re.IGNORECASE):
        return True
    if trimmed.endswith(":"):
        return True
    if not re.search(r"[.!?]\s*$", trimmed):
        return True

    return False


def should_render_confidence_badge(confidence_level: str) -> bool:
    """Hide all confidence badges."""
    return False


def is_simple_formatting_query(query: str) -> bool:
    """
    Detect simple formatting/style requests that don't need deep fact-checking.
    Examples: paraphrase, bullet points, summaries, rewrites, explanations.
    """
    query_lower = query.lower()
    simple_patterns = [
        r"\bparaphrase\b",
        r"\bparameter\b",
        r"\brewrite\b",
        r"\bsummari[sze]",
        r"\bsummary\b",
        r"\bbullet\s*points?\b",
        r"\bbullets?\b",
        r"\bpoint\s*form\b",
        r"\bparagraph",
        r"\bsimpl(ify|ified)\b",
        r"\bexplain\s+simply\b",
        r"\blist\b",
        r"\boutline\b",
        r"\breformat\b",
        r"\bstyle\b",
        r"\brephrase\b",
        r"\bwrite\s+in\b",
        r"\be[xs]plain\b",
    ]
    for pattern in simple_patterns:
        if re.search(pattern, query_lower):
            return True
    return False


def detect_medication_mention(query: str) -> tuple[bool, str]:
    """
    Detect if query mentions any medications.
    Returns: (is_medication_related, medication_name_or_list)
    """
    query_lower = query.lower()
    
    # Common medication-related keywords
    medication_keywords = [
        r"\bmedication(s)?\b",
        r"\bmedicine(s)?\b",
        r"\bdrug(s)?\b",
        r"\bpill(s)?\b",
        r"\btablet(s)?\b",
        r"\bprescription\b",
        r"\bover.the.counter\b",
        r"\botc\b",
        r"\btreatment\b",
        r"\bremedy\b",
    ]
    
    # Common medication names
    common_medications = [
        "ibuprofen", "aspirin", "acetaminophen", "paracetamol", "tylenol",
        "advil", "bayer", "aleve", "naproxen",
        "antacid", "antacids", "tums", "rolaids",
        "antihistamine", "antihistamines",
        "decongestant", "decongestants",
        "cough syrup", "cough medicine",
        "antibiotic", "antibiotics",
        "steroid", "steroids",
        "insulin", "metformin",
        "statins", "blood pressure",
        "antidepressant", "antidepressants",
    ]
    
    # Check keywords
    for pattern in medication_keywords:
        if re.search(pattern, query_lower):
            return True, "medication-related treatment"
    
    # Check specific medication names
    mentioned_meds = []
    for med in common_medications:
        if med.lower() in query_lower:
            mentioned_meds.append(med.title())
    
    if mentioned_meds:
        med_list = ", ".join(mentioned_meds)
        return True, med_list
    
    return False, ""


def render_parent_action_box(
    age_tier: int,
    query: str,
    filter_system: "SmokingContentFilter",
) -> None:
    """Supplemental parent action panel disabled by request."""
    return


# ╔╗
# ║         UPGRADE 6 — ADMIN STATS VIEW                                       ║
# ╚

@st.cache_resource
def get_rag_layer() -> "RAGKnowledgeLayer":
    """
    Speed Fix 7: Lazy-load RAG layer once per server process (cached across all sessions).
    Only called when the first question is submitted — NOT at startup.
    The @st.cache_resource decorator ensures the heavy model (BAAI/bge-large-en-v1.5)
    and Pinecone connection are initialised exactly once per server restart.
    """
    return RAGKnowledgeLayer()


def show_admin_stats() -> None:
    """
    Display admin statistics panel (Upgrade 6).

    Shows session-level counters (from st.session_state) and historical
    data loaded from questions_log.csv and feedback_log.csv.
    No personally identifiable information is shown.
    """
    st.subheader("📊 Admin Statistics")
    stats = st.session_state.get("session_stats", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Questions (session)", stats.get("total_questions", 0))
        st.metric("Safety Filter Triggers", stats.get("safety_triggers", 0))
        st.metric("Fact-Check Failures", stats.get("fact_check_failures", 0))
    with col2:
        st.metric("Hallucinations Detected", stats.get("hallucinations_detected", 0))
        st.metric("Encoding Errors / Regen", stats.get("encoding_errors", 0))
        st.metric("Responses Regenerated", stats.get("regenerations", 0))
    with col3:
        st.metric(" Positive Feedback", stats.get("positive_feedback", 0))
        st.metric("👎 Negative Feedback", stats.get("negative_feedback", 0))

    st.markdown("---")

    # Historical data from CSV
    if QUESTIONS_LOG.exists():
        try:
            rows: List[List[str]] = []
            with open(QUESTIONS_LOG, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            st.markdown(f"**Total questions logged (all sessions):** {len(rows)}")

            if rows:
                # Top 10 keywords (simple word-frequency, stopwords excluded)
                stop_words = {
                    "the", "a", "an", "is", "it", "in", "on", "of", "to", "and",
                    "or", "for", "with", "what", "how", "why", "does", "do", "can",
                    "i", "my", "me", "are", "was", "be", "this", "that", "about",
                    "from", "have", "has", "if", "will", "would", "should",
                }
                all_words: List[str] = []
                for row in rows:
                    q = row.get("question", "").lower()
                    all_words.extend(
                        w for w in re.findall(r"\b[a-z]{3,}\b", q) if w not in stop_words
                    )
                if all_words:
                    top_words = Counter(all_words).most_common(10)
                    st.markdown("**Top 10 question keywords (all sessions):**")
                    for word, count in top_words:
                        st.write(f"  • `{word}` — {count} occurrences")
        except Exception as e:
            st.warning(f"Could not load questions_log.csv: {e}")
    else:
        st.info("No questions_log.csv found yet.")

    if FEEDBACK_LOG.exists():
        try:
            pos = neg = 0
            with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("rating") == "positive":
                        pos += 1
                    elif row.get("rating") == "negative":
                        neg += 1
            total_fb = pos + neg
            if total_fb:
                pct = round(pos / total_fb * 100)
                st.markdown(
                    f"**Feedback (all sessions):** {pos}  / {neg} 👎 "
                    f"({pct}% positive)"
                )
        except Exception as e:
            st.warning(f"Could not load feedback_log.csv: {e}")


# ╔╗
# ║                    STREAMLIT MAIN APP — 14-STEP PIPELINE                   ║
# ╚

def main() -> None:
    st.set_page_config(
        page_title="SHS & Child Health — Parent Support",
        page_icon="",
        layout="wide",
    )

    # ── STEP 1 — Age Gate ─────────────────────────────────────────────────────
    if "user_age" not in st.session_state:
        render_age_gate()
        return

    user_age: int = st.session_state.user_age
    age_tier: int = get_age_tier(user_age)
    age_category: str = tier_to_category(age_tier)

    # Tier 1 (≤13) — block the app entirely; redirect to trusted adult
    if age_tier == 1:
        st.error(
            "🚫 This app is designed for parents, caregivers, and teens 14 and older. "
            "If you have questions about smoking or health, please talk to a trusted adult, "
            "parent, teacher, or school nurse."
        )
        if st.button(" Go back and re-enter age"):
            del st.session_state["user_age"]
            st.rerun()
        return

    # ── Initialise non-RAG singletons (RAG is lazy-loaded on first question) ──
    if "content_filter" not in st.session_state:
        st.session_state.content_filter = SmokingContentFilter()
    if "hallucination_guard" not in st.session_state:
        st.session_state.hallucination_guard = HallucinationGuard()
    if "fact_checker" not in st.session_state:
        st.session_state.fact_checker = FactChecker()
    if "safety_filter" not in st.session_state:
        st.session_state.safety_filter = SafetyFilter()
    if "confidence_calc" not in st.session_state:
        st.session_state.confidence_calc = ConfidenceCalculator()
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {
            "total_questions": 0,
            "safety_triggers": 0,
            "fact_check_failures": 0,
            "hallucinations_detected": 0,
            "encoding_errors": 0,
            "regenerations": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
        }
    # Speed Fix 3: Session response cache (max 20 entries, LRU-style)
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}
    if "queued_query" not in st.session_state:
        st.session_state.queued_query = ""
    # Check OpenAI availability once per session (cached)
    if "model_available" not in st.session_state:
        available_models = RAGKnowledgeLayer.list_ollama_models()
        st.session_state.model_available = MODEL in available_models

    filter_system: SmokingContentFilter = st.session_state.content_filter
    h_guard: HallucinationGuard = st.session_state.hallucination_guard
    fc: FactChecker = st.session_state.fact_checker
    sf: SafetyFilter = st.session_state.safety_filter
    conf_calc: ConfidenceCalculator = st.session_state.confidence_calc

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙ Configuration")

        tier_labels = {2: "Teen (14–17) — Strict Mode", 3: "Adult (18+) — Full Mode"}
        st.info(f"**Age Tier:** {tier_labels.get(age_tier, str(age_tier))}")

        if st.button(" Change age"):
            del st.session_state["user_age"]
            st.rerun()

        st.markdown("---")
        # fixed model indicator with availability status
        if st.session_state.model_available:
            st.success(f"🟢 Model: `{MODEL}`")
        else:
            st.error("🔴 OPENAI_API_KEY is missing or invalid.")

        st.markdown("---")
        # RAG status — shown after first question (Speed Fix 7: lazy load)
        rag_status_placeholder = st.empty()
        if "rag_is_ready" in st.session_state:
            if st.session_state.rag_is_ready:
                rag_status_placeholder.success("✅ Pinecone RAG: Connected")
            else:
                rag_status_placeholder.warning(
                    f"⚠ RAG: {st.session_state.get('rag_error', 'Unavailable')}"
                )
        else:
            rag_status_placeholder.info("💡 RAG: Will connect on first question")

        policy = filter_system.age_policies[age_category]
        st.markdown(f"**Response Style:** {policy['response_style']}")

        with st.expander("Content Policies"):
            st.write("**Allowed:**")
            for t in policy["allowed_topics"]:
                st.write(f"✅ {t}")
            if policy["forbidden_topics"]:
                st.write("**Forbidden:**")
                for t in policy["forbidden_topics"]:
                    st.write(f" {t}")

        st.markdown("---")
        # Upgrade 6 — Admin stats toggle (no PII shown)
        show_admin = st.checkbox("📊 Show Admin Stats")

        st.markdown("---")
        if st.button("🧹 Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.pop("pipeline_result", None)
            st.session_state.pop("feedback_state", None)
            st.session_state.pop("response_cache", None)
            st.session_state.pop("queued_query", None)
            st.session_state.pop("last_confidence_level", None)
            st.session_state.pop("last_parent_action_intent", None)
            st.rerun()

        # Suggested questions for parent users
        if age_tier == 3:
            st.markdown("---")
            st.markdown("**💡 Try asking:**")
            suggested = [
                "How does secondhand smoke affect my child's asthma?",
                "Is smoking outside safe enough to protect my baby?",
                "What NRT options are safe while breastfeeding?",
                "How quickly does my child's health improve after I quit?",
                "Does opening a window protect my child from smoke?",
            ]
            for s in suggested:
                if st.button(s, key=f"sugg_{hash(s)}", use_container_width=True):
                    st.session_state["prefill_query"] = s
                    st.rerun()

    if show_admin:
        show_admin_stats()
        st.markdown("---")

    # Block app if OpenAI client is unavailable
    if not st.session_state.model_available:
        st.error(
            f"🔴 OpenAI access is not configured for model **{MODEL}**. "
            "Set `OPENAI_API_KEY` and refresh this page."
        )
        st.stop()

    # ── Title ─────────────────────────────────────────────────────────────────
    if age_tier == 3:
        st.title(" SHS & Child Health — Parent Support")
        st.caption(
            "Evidence-based secondhand smoke information to help you protect your child."
        )
    else:
        st.title("🚭 Smoking & Health Education")
        st.caption(
            f"Age-appropriate health information | Model: {MODEL} | 14-step safety pipeline"
        )

    # ── Conversation history ──────────────────────────────────────────────────
    if st.session_state.conversation_history:
        st.markdown("### Conversation")
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["text"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["text"])

    # ── Single bottom query input ─────────────────────────────────────────────
    prefill = st.session_state.pop("prefill_query", "").strip()
    queued_query = (st.session_state.get("queued_query", "") or "").strip()
    if prefill:
        queued_query = prefill
        st.session_state["queued_query"] = prefill

    typed_query = st.chat_input("Your question:")
    submitted_query = queued_query or ((typed_query or "").strip())

    if submitted_query:
        query = submitted_query.strip()
        # Consume queued query once this run starts processing.
        st.session_state["queued_query"] = ""
        if not query:
            st.warning("Please enter a question.")
            st.stop()

        # ── Speed Fix 3: Check session cache before running any pipeline ──────
        normalized_query = normalize_query_for_cache(query)
        cache_key = hashlib.md5(f"{normalized_query}{age_tier}".encode()).hexdigest()
        if cache_key in st.session_state.response_cache:
            cached = st.session_state.response_cache[cache_key]
            cached_query = normalize_query_for_cache(cached.get("query", ""))
            if cached_query == normalized_query:
                display_response = strip_response_headings(cached["response"])
                display_response = strip_role_labels(display_response)
                display_response = clean_generated_response_artifacts(display_response)
                st.info("⚡ Retrieved from session cache")
                st.success("✅ Response Generated")
                st.markdown("### Response:")
                st.write(display_response)
                
                # Check if medication-related and display clinical note
                is_med_related, med_name = detect_medication_mention(query)
                if is_med_related:
                    st.markdown("---")
                    st.info(
                        f"💊 **Important:** Please consult with a doctor, clinician, or healthcare provider "
                        f"regarding **{med_name}** and any potential interactions with secondhand smoke exposure."
                    )
                
                if should_render_confidence_badge(cached["confidence_level"]):
                    st.markdown(cached["badge"])
                render_parent_action_box(age_tier, query, filter_system)
                st.session_state.conversation_history.append({"role": "user", "text": query})
                st.session_state.conversation_history.append(
                    {"role": "assistant", "text": display_response}
                )
                st.session_state.pipeline_result = {
                    "question": query,
                    "response": display_response,
                    "age_tier": age_tier,
                    "safety_triggered": False,
                    "fact_check_result": "cached",
                    "confidence_level": cached["confidence_level"],
                }
                st.session_state.feedback_state = "pending"
                st.session_state.session_stats["total_questions"] += 1
                st.rerun()
            # Remove stale/incompatible cache entries instead of returning wrong content.
            del st.session_state.response_cache[cache_key]

        # ── Pipeline tracking variables ───────────────────────────────────────
        safety_triggered  = False
        fact_check_result = "pass"
        hallu_detected    = False
        resp_discarded    = False
        encoding_errors   = 0
        regenerations     = 0
        pipeline_labels: List[str] = []

        # Speed Fix 6: Progress bar with step labels
        progress = st.progress(0, text=" Starting pipeline…")

        # 
        # STEP 1 — Age gate already enforced above
        # 

        slang_detected = filter_system.detect_slang_evasion(query)
        risk_level, keywords = filter_system.classify_query_risk(query)
        should_block, block_reason = filter_system.should_block_query(age_category, query)

        # Slang evasion: if a minor uses slang to bypass filters, treat as high-risk
        # and either escalate the block decision or warn them clearly.
        if slang_detected and age_tier == 2 and not should_block:
            # Elevate risk — re-check blocking at high_risk threshold
            if risk_level in ("medium", "low"):
                should_block = True
                block_reason = (
                    "Access denied: Your question appears to use informal language about "
                    "tobacco or smoking products. Users under 18 can ask about health effects, "
                    "prevention, and how to seek support."
                )

        if should_block:
            progress.empty()
            st.error(f"🚫 {block_reason}")
            if age_tier == 2:
                st.info(
                    "You can ask about:\n"
                    "- Health effects of smoking\n"
                    "- Why smoking is harmful\n"
                    "- How to say no to peer pressure\n"
                    "- Addiction and prevention"
                )
            log_interaction(user_age, query, "", blocked=True, action="blocked", reason=block_reason)
            log_question(
                user_age_tier=age_tier,
                question=query,
                confidence_level="N/A",
                safety_filter_triggered=False,
                fact_check_result="blocked",
                hallucination_detected=False,
                response_discarded=False,
            )
            st.stop()

        # 
        # STEP 2 — RAG Retrieval
        # Speed Fix 7: RAG loaded via @st.cache_resource (once per server).
        # Called here (on first question) NOT at app startup.
        # 
        progress.progress(15, text=" Step 2 — Loading knowledge base…")
        rag = get_rag_layer()

        # Cache RAG status in session_state so sidebar can show it
        st.session_state.rag_is_ready = rag.is_ready
        st.session_state.rag_error = rag.error_message or ""

        fast_mode = is_simple_formatting_query(query)
        rag_matches: List[Dict] = []
        rag_context = ""
        conversation_context = build_recent_conversation_context(
            st.session_state.conversation_history
        )
        use_conversation_context = bool(conversation_context) and is_follow_up_query(query)
        retrieval_query = query
        if use_conversation_context and len(query.split()) <= 12:
            retrieval_query = (
                f"Recent conversation:\n{conversation_context}\n\n"
                f"Follow-up question: {query}"
            )
        if rag.is_ready:
            progress.progress(22, text=" Step 2 — Searching knowledge base…")
            retrieval_top_k = 3 if fast_mode else 5
            rag_matches = rag.search(
                retrieval_query,
                top_k=retrieval_top_k,
                age_category=age_category,
            )
            rag_context = rag.build_rag_context(rag_matches)

        # 
        # STEP 3 — Confidence Score
        # 
        confidence = conf_calc.calculate(rag_matches)

        # 
        # STEP 4 — Low confidence + teen → safe message, stop pipeline
        # 
        if confidence["level"] == "LOW" and age_tier == 2:
            progress.empty()
            st.warning(
                "⚠ We don't have enough verified information to answer this safely. "
                "Please speak to a doctor, school nurse, or call 1-800-QUIT-NOW "
                "(1-800-784-8669)."
            )
            log_question(
                user_age_tier=age_tier,
                question=query,
                confidence_level="LOW",
                safety_filter_triggered=False,
                fact_check_result="skipped",
                hallucination_detected=False,
                response_discarded=True,
            )
            st.session_state.session_stats["total_questions"] += 1
            st.stop()

        temp = 0.05 if confidence["level"] == "LOW" else 0.1

        # 
        # STEP 5 — Generate response via Ollama
        # MODEL SWITCH: uses MODEL constant (qwen2.5:3b)
        # Speed Fix 4: streaming display via st.empty() placeholder
        # 
        progress.progress(35, text=f"💬 Step 5 — Generating response ({MODEL})…")
        evidence_block = build_rag_evidence_block(rag_matches, max_items=5)
        system_prompt = build_shs_structured_prompt(
            query=query,
            evidence_block=evidence_block or "No relevant evidence retrieved.",
            age_tier=age_tier,
            conversation_context=conversation_context if use_conversation_context else "",
        )
        regen_prompt_with_context = (
            f"{system_prompt}\n\n"
            "Regeneration requirements:\n"
            "- Use only evidence from EVIDENCE BLOCK.\n"
            "- Answer all parts of the user's question directly.\n"
            "- Do not ask the user follow-up questions unless absolutely necessary.\n"
            "- Do not repeat generic smoking warnings unless directly relevant.\n"
            "- Keep wording concise and non-repetitive.\n"
            "ANSWER:"
        )
        question_count = len(re.findall(r"\?", query))
        response_token_budget = 360 if question_count >= 2 else (220 if fast_mode else 300)
        stream_placeholder = st.empty()
        raw_response = rag.generate_answer(
            system_prompt, model=MODEL, temperature=temp,
            stream_placeholder=stream_placeholder,
            num_predict=response_token_budget,
        )
        stream_placeholder.empty()  # Clear streaming cursor; final shown below

        # 
        # Speed Fix 1: Python quality checks FIRST before any more LLM calls.
        # Check 1: loops, Check 2: length, Check 3: language,
        # Check 4: stage model terms, Check 5: contradiction
        # 
        progress.progress(46, text="🔎 Checking response quality…")
        quality_issues = check_response_quality(raw_response)
        if quality_issues:
            log_hallucination_error(
                "QUALITY CHECK FAILED: " + "; ".join(quality_issues), raw_response
            )
            regenerations += 1
            st.session_state.session_stats["encoding_errors"] += 1
            raw_response = rag.generate_answer(
                regen_prompt_with_context, model=MODEL, temperature=temp
            )

        # 
        # STEP 6 — Language / Encoding Guard (up to 2 attempts)
        # 
        progress.progress(52, text="🔤 Step 6 — Checking encoding…")
        max_encoding_attempts = 1 if fast_mode else 2
        for _attempt in range(max_encoding_attempts):
            encoding_ok, encoding_detail = h_guard.check_encoding(raw_response)
            if encoding_ok:
                break
            encoding_errors += 1
            regenerations += 1
            log_hallucination_error(
                f"ENCODING ERROR (attempt {_attempt + 1}): {encoding_detail}", raw_response
            )
            st.session_state.session_stats["encoding_errors"] += 1
            regen_prompt = (
                f"{regen_prompt_with_context}\n\n"
                "Additional requirement: use plain English characters only."
            )
            raw_response = rag.generate_answer(
                regen_prompt,
                model=MODEL,
                temperature=temp,
                num_predict=180 if fast_mode else 260,
            )
        else:
            raw_response = filter_system.response_validator.get_safe_fallback(age_category, query)
            resp_discarded = True

        # 
        # STEP 7 — Hallucination Guard
        # 
        progress.progress(61, text="🧠 Step 7 — Hallucination check…")
        h_result = h_guard.run(raw_response)
        if h_result["incorrect_facts"] or h_result["contradictions"]:
            hallu_detected = True
            resp_discarded = True
            regenerations += 1
            st.session_state.session_stats["hallucinations_detected"] += 1
            issues_found = h_result["incorrect_facts"] + h_result["contradictions"]
            log_hallucination_error(
                "HALLUCINATION DETECTED: " + "; ".join(issues_found), raw_response
            )
            regen_prompt = (
                f"{regen_prompt_with_context}\n\n"
                "Additional requirement: do not invent any medical terminology."
            )
            raw_response = rag.generate_answer(regen_prompt, model=MODEL, temperature=temp)

        # Output validation (forbidden terms / substance scan)
        validated = filter_system.validate_output(raw_response, age_category, query)
        current_response = validated["final_response"]
        current_response = strip_response_headings(current_response)
        current_response = strip_role_labels(current_response)
        current_response = clean_generated_response_artifacts(current_response)
        current_response = strip_leading_followup_questions(current_response)

        if looks_incomplete_response(current_response):
            regenerations += 1
            completion_prompt = (
                f"{regen_prompt_with_context}\n\n"
                "Additional requirement: complete the response fully and do not end mid-sentence."
            )
            retry_response = rag.generate_answer(
                completion_prompt,
                model=MODEL,
                temperature=temp,
                num_predict=260 if fast_mode else 340,
            )
            retry_response = clean_generated_response_artifacts(
                strip_role_labels(strip_response_headings(retry_response))
            )
            retry_response = strip_leading_followup_questions(retry_response)
            if retry_response and not looks_incomplete_response(retry_response):
                current_response = retry_response
        if validated["action"] == "blocked":
            pipeline_labels.append("⚠ Response adjusted for age-appropriateness.")
        elif validated["action"] == "adjusted":
            pipeline_labels.append(" Minor language adjustments applied.")

        # 
        # STEP 8 — Fact-Checking Layer
        # Speed Fix 2: For 18+ users, fact-check runs via ThreadPoolExecutor.
        # MODEL SWITCH: uses MODEL constant
        # 
        progress.progress(72, text="🔬 Step 8 — Fact-checking…")
        run_fact_check = True
        if (
            fast_mode
            and confidence["level"] in ["HIGH", "MEDIUM"]
            and not quality_issues
            and not hallu_detected
            and encoding_errors == 0
        ):
            run_fact_check = False
            fact_check_result = "skipped_simple_request"

        if run_fact_check:
            if age_tier == 3:
                # 18+ — no safety filter; use thread pool for fact-check
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    fc_future = executor.submit(fc.check, query, current_response, MODEL)
                    fc_result = fc_future.result()
            else:
                # teens — sequential (safety filter depends on fact-check output)
                fc_result = fc.check(query, current_response, MODEL)
        else:
            fc_result = {"result": "PASS", "issues": []}

        if fc_result["result"] == "FAIL":
            fact_check_result = "fail"
            st.session_state.session_stats["fact_check_failures"] += 1
            log_fact_check_failure(query, current_response, fc_result["issues"])

            # 
            # STEP 9 — Accuracy rewrite
            # MODEL SWITCH: uses MODEL constant
            # 
            progress.progress(81, text=" Step 9 — Rewriting for accuracy…")
            current_response = fc.rewrite(current_response, fc_result["issues"], MODEL, age_tier=age_tier)
            pipeline_labels.append("⚠ Some information was reviewed and corrected for accuracy")
        elif not run_fact_check:
            pipeline_labels.append("⚡ Fast mode: skipped extra fact-check for low-risk response")

        # 
        # STEP 10 — Safety Filter (teens ≤17 only)
        # MODEL SWITCH: uses MODEL constant
        # 
        if age_tier == 2:
            progress.progress(88, text="🛡 Step 10 — Safety review…")
            sf_result = sf.check(current_response, user_age, MODEL)

            if sf_result["result"] == "UNSAFE":
                safety_triggered = True
                st.session_state.session_stats["safety_triggers"] += 1
                log_safety_rewrite(
                    user_age=user_age,
                    original=current_response,
                    rewritten=sf_result["rewritten"] or current_response,
                    reason="SafetyFilter returned UNSAFE",
                )
                # STEP 11 — Safety rewrite applied
                current_response = sf_result["rewritten"] or current_response
                pipeline_labels.append(" Response adapted for your age group")

        # 
        # STEP 12 — Display final response with confidence badge
        # 
        progress.progress(100, text="✅ Complete!")
        progress.empty()

        for label in pipeline_labels:
            if label.startswith("⚠"):
                st.warning(label)
            else:
                st.info(label)

        st.success("✅ Response Generated")
        st.markdown("### Response:")
        st.write(current_response)
        
        # Check if medication-related and display clinical note
        is_med_related, med_name = detect_medication_mention(query)
        if is_med_related:
            st.markdown("---")
            st.info(
                f"💊 **Important:** Please consult with a doctor, clinician, or healthcare provider "
                f"regarding **{med_name}** and any potential interactions with secondhand smoke exposure."
            )

        if should_render_confidence_badge(confidence["level"]):
            st.markdown(confidence["badge"])
        if confidence["level"] == "LOW":
            st.error(
                "🔴 Low confidence — This answer is based on limited verified information. "
                "Please verify with a doctor or call 1-800-QUIT-NOW (1-800-784-8669)."
            )

        render_parent_action_box(age_tier, query, filter_system)

        if rag_matches:
            st.markdown("### Top 3 Sources")
            for i, m in enumerate(rag_matches[:3], 1):
                source_name = m.get("verified_source") or m.get("source", "Unknown")
                st.markdown(
                    f"{i}. **{source_name}**"
                    f" ({m.get('source', 'Unknown')})"
                    f" · relevance {m.get('score', 0):.3f}"
                )

            with st.expander("📚 Verified Sources Used"):
                for i, m in enumerate(rag_matches, 1):
                    source_name = m.get("verified_source") or m.get("source", "Unknown")
                    st.markdown(
                        f"**Source {i}:** {source_name} "
                        f"· Relevance: {m.get('score', 0):.3f}"
                    )
                    snippet = (m.get("text") or "")[:300]
                    if snippet:
                        st.caption(snippet)

        # Speed Fix 3: Store in session cache
        # Never cache LOW confidence or safety-rewritten responses
        if confidence["level"] != "LOW" and not safety_triggered:
            cache = st.session_state.response_cache
            if len(cache) >= 20:
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            cache[cache_key] = {
                "query": query,
                "response": current_response,
                "badge": confidence["badge"],
                "confidence_level": confidence["level"],
            }

        st.session_state.conversation_history.append({"role": "user", "text": query})
        st.session_state.conversation_history.append(
            {"role": "assistant", "text": current_response}
        )

        st.session_state.pipeline_result = {
            "question": query,
            "response": current_response,
            "age_tier": age_tier,
            "safety_triggered": safety_triggered,
            "fact_check_result": fact_check_result,
            "confidence_level": confidence["level"],
        }
        st.session_state.feedback_state = "pending"
        st.session_state.session_stats["total_questions"] += 1
        if regenerations:
            st.session_state.session_stats["regenerations"] += regenerations

        # 
        # STEP 14 — Log question
        # 
        log_question(
            user_age_tier=age_tier,
            question=query,
            confidence_level=confidence["level"],
            safety_filter_triggered=safety_triggered,
            fact_check_result=fact_check_result,
            hallucination_detected=hallu_detected,
            response_discarded=resp_discarded,
        )
        log_interaction(
            age=user_age,
            query=query,
            response=current_response,
            blocked=False,
            action=validated["action"],
            reason=validated["reason"],
            rag_sources=rag_matches,
            retrieval_scores=[m.get("score", 0) for m in rag_matches],
        )
        
        # Rerun to show updated conversation history with the new question
        st.rerun()

    # ── STEP 13 — Feedback Buttons ────────────────────────────────────────────
    pr = st.session_state.get("pipeline_result")
    fb_state = st.session_state.get("feedback_state", "none")

    if pr and fb_state == "pending":
        st.markdown("---")
        st.markdown("**Was this response helpful?**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Helpful", key="btn_thumbs_up"):
                log_feedback(
                    user_age_tier=pr["age_tier"],
                    question=pr["question"],
                    response_snippet=pr["response"],
                    rating="positive",
                    user_comment="",
                    safety_filter_triggered=pr["safety_triggered"],
                    fact_check_result=pr["fact_check_result"],
                )
                st.session_state.session_stats["positive_feedback"] += 1
                st.session_state.feedback_state = "submitted"
                st.rerun()
        with col2:
            if st.button("👎 Not Helpful", key="btn_thumbs_down"):
                st.session_state.feedback_state = "thumbs_down"
                st.rerun()

    elif pr and fb_state == "thumbs_down":
        st.markdown("---")
        comment = st.text_input(
            "What was wrong with this answer? (optional, max 200 chars)",
            max_chars=200,
            key="feedback_comment_input",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Submit feedback", key="btn_submit_fb"):
                log_feedback(
                    user_age_tier=pr["age_tier"],
                    question=pr["question"],
                    response_snippet=pr["response"],
                    rating="negative",
                    user_comment=comment,
                    safety_filter_triggered=pr["safety_triggered"],
                    fact_check_result=pr["fact_check_result"],
                )
                st.session_state.session_stats["negative_feedback"] += 1
                st.session_state.feedback_state = "submitted"
                st.rerun()
        with c2:
            if st.button("Skip", key="btn_skip_fb"):
                log_feedback(
                    user_age_tier=pr["age_tier"],
                    question=pr["question"],
                    response_snippet=pr["response"],
                    rating="skipped",
                    user_comment="",
                    safety_filter_triggered=pr["safety_triggered"],
                    fact_check_result=pr["fact_check_result"],
                )
                # Do NOT count "skipped" as negative — user chose not to rate
                st.session_state.feedback_state = "submitted"
                st.rerun()

    elif fb_state == "submitted":
        st.markdown("---")
        st.success("Thanks for helping us improve! 💙")


# ╔╗
# ║           CHUNKING HELPER — Recursive Character Splitting + Overlap        ║
# ╚

def chunk_document(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
    min_chunk_size: int = 100,
) -> List[str]:
    """
    Split text using recursive character splitting with overlap.

    Strategy (RecursiveCharacterTextSplitter-style):
    - Try paragraph boundaries first, then newlines, then sentences, then words
    - chunk_size=800 chars (~120 tokens, within bge-large-en-v1.5's 512-token limit)
    - overlap=200 chars ensures no medical fact is lost at chunk boundaries
    - min_chunk_size=100 chars skips tiny fragments
    """
    if not text or not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " "]
    chunks: List[str] = []

    def _split_recursive(text_block: str, sep_index: int = 0) -> List[str]:
        if len(text_block) <= chunk_size:
            return [text_block] if len(text_block.strip()) >= min_chunk_size else []
        if sep_index < len(separators):
            sep = separators[sep_index]
            parts = text_block.split(sep)
            if len(parts) <= 1:
                return _split_recursive(text_block, sep_index + 1)
            merged: List[str] = []
            current = ""
            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        merged.append(current)
                    if len(part) > chunk_size:
                        merged.extend(_split_recursive(part, sep_index + 1))
                    else:
                        current = part
            if current:
                merged.append(current)
            return merged
        else:
            result = []
            for i in range(0, len(text_block), chunk_size - overlap):
                window = text_block[i: i + chunk_size]
                if len(window.strip()) >= min_chunk_size:
                    result.append(window)
            return result if result else [text_block]

    raw_chunks = _split_recursive(text.strip())
    if not raw_chunks:
        return []

    chunks.append(raw_chunks[0].strip())
    for i in range(1, len(raw_chunks)):
        prev = raw_chunks[i - 1]
        curr = raw_chunks[i].strip()
        if len(prev) > overlap:
            overlap_text = prev[-overlap:]
            if not curr.startswith(overlap_text.strip()):
                curr = overlap_text.strip() + " " + curr
        if len(curr.strip()) >= min_chunk_size:
            chunks.append(curr.strip())

    return chunks


# ╔╗
# ║           ADMIN PANEL — Knowledge Base Upload (Pinecone)                   ║
# ╚

def _guess_source_from_filename(filename: str) -> str:
    name = filename.lower()
    if "cdc" in name:
        return "CDC"
    if "nih" in name:
        return "NIH"
    if "who" in name:
        return "WHO"
    if "pubmed" in name or "pmc" in name:
        return "PubMed"
    return "Other"


def _guess_topic_from_text(text: str) -> str:
    text_lower = text[:3000].lower()
    topic_keywords = {
        "cessation": ["quit", "cessation", "stop smoking", "quitting", "nicotine replacement", "nrt", "withdrawal"],
        "prevention": ["prevention", "prevent", "youth", "school", "peer pressure", "say no", "never start"],
        "statistics": ["prevalence", "mortality rate", "percentage", "survey", "statistics", "epidemiology"],
        "policy": ["legislation", "regulation", "law", "ban", "tax", "policy", "fctc", "smoke-free law"],
        "health_effects": ["cancer", "lung", "disease", "health effect", "cardiovascular", "copd", "mortality"],
    }
    scores = {
        topic: sum(1 for kw in keywords if kw in text_lower)
        for topic, keywords in topic_keywords.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "health_effects"


def admin_panel() -> None:
    """Admin panel for uploading verified medical documents into Pinecone."""
    st.title(" Admin — Knowledge Base Management")
    st.caption("Upload verified medical documents (CDC, NIH, PubMed) into Pinecone")
    st.markdown(
        "> **Roles:** Hassan (Admin) · Mukesh (Admin) · Prof. Tan (Super Admin) · Regular User (No Access)"
    )

    # Speed Fix 7: use cached RAG layer (same @st.cache_resource instance)
    rag: RAGKnowledgeLayer = get_rag_layer()

    if not rag.is_ready:
        st.error(f"Pinecone not connected: {rag.error_message}")
        st.stop()

    st.success("✅ Pinecone connected — ready for uploads")

    uploaded_files = st.file_uploader(
        "Upload PDFs or text documents (select multiple files at once)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    st.markdown("#### Default Metadata (applied to all files)")
    st.info(
        "**Tip:** Leave **Age Relevance = all** if you're unsure — the safety filter "
        "already controls what each age group sees. Source and Topic are auto-detected "
        "from filenames and content, but you can override below."
    )
    col1, col2 = st.columns(2)
    with col1:
        override_source = st.selectbox(
            "Verified Source",
            ["Auto-detect from filename", "CDC", "NIH", "WHO", "PubMed", "Other"],
        )
    with col2:
        age_relevance = st.selectbox("Age Relevance", ["all", "child", "teen", "adult"])
    override_topic = st.selectbox(
        "Topic Category",
        ["Auto-detect from content", "health_effects", "cessation", "prevention", "statistics", "policy"],
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            src_guess = _guess_source_from_filename(f.name)
            st.write(f"- {f.name} ({f.size:,} bytes) — detected source: **{src_guess}**")

        st.warning(
            "⚠ All uploaded documents must be from verified sources only "
            "(CDC, NIH, PubMed, WHO). Admin upload does not bypass source verification policy."
        )

        if st.button("Process & Upload All to Pinecone", type="primary"):
            total_success = total_chunks = 0

            for file_idx, uploaded_file in enumerate(uploaded_files):
                st.markdown(
                    f"---\n**Processing {file_idx + 1}/{len(uploaded_files)}: {uploaded_file.name}**"
                )
                source_type = "pdf" if uploaded_file.name.lower().endswith(".pdf") else "txt"
                verified_source = (
                    _guess_source_from_filename(uploaded_file.name)
                    if override_source == "Auto-detect from filename"
                    else override_source
                )

                with st.spinner(f"Extracting text from {uploaded_file.name}…"):
                    raw_text = ""
                    if uploaded_file.name.lower().endswith(".pdf"):
                        try:
                            from PyPDF2 import PdfReader
                            reader = PdfReader(uploaded_file)
                            for page in reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    raw_text += page_text + "\n"
                        except Exception as e:
                            st.error(f"PDF extraction failed for {uploaded_file.name}: {e}")
                            continue
                    else:
                        raw_text = uploaded_file.read().decode("utf-8", errors="replace")

                if not raw_text.strip():
                    st.warning(f"No text extracted from {uploaded_file.name} — skipping.")
                    continue

                if override_topic == "Auto-detect from content":
                    topic = _guess_topic_from_text(raw_text)
                    st.write(f"Auto-detected topic: **{topic}**")
                else:
                    topic = override_topic

                with st.spinner("Chunking text (800-char chunks, 200-char overlap)…"):
                    chunks = chunk_document(raw_text, chunk_size=800, overlap=200)
                    if not chunks:
                        st.warning(f"No valid chunks from {uploaded_file.name} — skipping.")
                        continue
                    avg_len = sum(len(c) for c in chunks) // len(chunks)
                    st.write(
                        f"Created **{len(chunks)}** chunks (avg {avg_len} chars) | "
                        f"Source: **{verified_source}** | Age: **{age_relevance}** | Topic: **{topic}**"
                    )

                progress = st.progress(0, text=f"Uploading {uploaded_file.name}…")
                success_count = 0
                source_name = uploaded_file.name
                timestamp = datetime.utcnow().isoformat()

                for idx, chunk in enumerate(chunks):
                    try:
                        embedding = rag._get_embedding(chunk)
                        vector_id = f"admin-{source_name}-{idx}-{int(datetime.utcnow().timestamp())}"
                        rag._index.upsert(vectors=[(
                            vector_id,
                            embedding,
                            {
                                "text": chunk[:4000],
                                "source": source_name,
                                "uploaded_at": timestamp,
                                "chunk_index": idx,
                                "source_type": source_type,
                                "age_relevance": age_relevance,
                                "topic": topic,
                                "char_count": len(chunk),
                                "verified_source": verified_source,
                            },
                        )])
                        success_count += 1
                    except Exception as e:
                        st.warning(f"Chunk {idx} failed: {e}")
                    progress.progress(
                        (idx + 1) / len(chunks),
                        text=f"Chunk {idx + 1}/{len(chunks)}",
                    )

                progress.empty()
                st.success(
                    f"✅ {uploaded_file.name}: **{success_count}/{len(chunks)}** chunks uploaded"
                )
                total_success += success_count
                total_chunks += len(chunks)

            st.markdown("---")
            st.success(
                f"🎉 Batch complete: **{total_success}/{total_chunks}** total chunks uploaded "
                f"across **{len(uploaded_files)}** files"
            )


# ╔╗
# ║                              ENTRY POINT                                   ║
# ╚

if __name__ == "__main__":
    # Minimal admin guard — in production use proper auth
    params = st.query_params if hasattr(st, "query_params") else {}
    admin_val = params.get("admin", "0")
    if isinstance(admin_val, list):
        admin_val = admin_val[0] if admin_val else "0"
    is_admin = str(admin_val) == "1"

    with st.sidebar:
        if st.button(" Admin Panel"):
            st.query_params["admin"] = "1"
            st.rerun()
        if is_admin and st.button(" Back to Chat"):
            st.query_params["admin"] = "0"
            st.rerun()

    if is_admin:
        admin_panel()
    else:
        main()

