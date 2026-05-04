import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from dotenv import load_dotenv, dotenv_values

from core.context_manager import ContextManager
from core.prompt_builder import PromptBuilder
from core.system_prompt import MEDICAL_PREFIX
from core.token_budget import TokenBudgeter
from core.retrieval_manager import RetrievalManager
from core.conversation_summarizer import ConversationSummarizer
from core.memory_store import MemoryStore

MODEL = "gpt-4o-mini"

_ENV_PATH = Path(__file__).parent / ".env"
_ENV_TXT_PATH = Path(__file__).parent / ".env.txt"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
elif _ENV_TXT_PATH.exists():
    load_dotenv(dotenv_path=_ENV_TXT_PATH, override=True)

_dotenv_map: Dict[str, str] = {}
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
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
store = MemoryStore(Path(__file__).parent / "logs" / "conversation_memory.sqlite")
context_manager = ContextManager(recent_min=4, recent_max=8)
summarizer = ConversationSummarizer()
budgeter = TokenBudgeter(max_input_tokens=1800)
retrieval_manager = RetrievalManager(store)
prompt_builder = PromptBuilder(MEDICAL_PREFIX)

app = FastAPI(title="SHS Chat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(event: str, data: Dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


def _is_shs_related(query: str) -> bool:
    q = (query or "").lower()
    terms = [
        "smoke", "smoking", "secondhand", "shs", "thirdhand", "nicotine", "cigarette",
        "tobacco", "quit", "cessation", "asthma", "sids", "baby", "child", "vape",
    ]
    return any(t in q for t in terms)


def _fallback_evidence(query: str) -> str:
    return (
        "No verified retrieval snippets were available. Use established CDC/NHS/WHO facts only, "
        "avoid speculation, and provide practical actions for reducing SHS exposure."
    )


def _looks_incomplete(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 60:
        return True
    if re.search(r"\b(to|for|with|about|and|or|because|including|such as)\s*$", t, re.IGNORECASE):
        return True
    if t.endswith(":"):
        return True
    if not re.search(r"[.!?]\s*$", t):
        return True
    return False


async def _generate_stream(session_id: str, user_age: int, query: str):
    t0 = time.time()
    age_tier = 3 if user_age >= 18 else (2 if user_age >= 14 else 1)
    if age_tier == 1:
        yield _sse("error", {"message": "This app is for parents/caregivers and teens 14+."})
        return

    store.upsert_session(session_id, user_age, age_tier)
    history = store.get_messages(session_id, limit=250)
    summary = store.get_summary(session_id)
    memory = store.get_user_memory(session_id)
    if not memory:
        memory = {
            "age": user_age,
            "quit_progress": "",
            "cigarettes_per_day": "",
            "quit_date": "",
            "relapse_count": 0,
            "tone_preference": "supportive",
        }
        store.upsert_user_memory(session_id, memory)

    ctx = context_manager.build(history, summary, memory, query)
    yield _sse("status", {"step": "context", "message": "Building context"})

    rag_docs: List[Dict] = []
    if _is_shs_related(query):
        cache_key = retrieval_manager.make_cache_key("adult" if age_tier == 3 else "teen", query)
        cached = store.get_retrieval_cache(cache_key)
        if cached is not None:
            rag_docs = cached
        else:
            # Lightweight lexical fallback retrieval from prior assistant facts.
            scored = []
            q_terms = set(re.findall(r"\w+", query.lower()))
            for m in history[-30:]:
                if m.get("role") != "assistant":
                    continue
                txt = (m.get("text") or "").strip()
                if not txt:
                    continue
                terms = set(re.findall(r"\w+", txt.lower()))
                inter = len(q_terms & terms)
                if inter > 0:
                    scored.append({"text": txt, "score": inter / max(1, len(q_terms)), "source": "Session"})
            rag_docs = retrieval_manager.rerank_dedupe(query, scored, keep=5)
            store.upsert_retrieval_cache(cache_key, rag_docs)

    evidence = retrieval_manager.evidence_block(rag_docs, max_items=5) or _fallback_evidence(query)

    sections = {
        "system_prompt": MEDICAL_PREFIX,
        "safety_rules": "No unsafe advice. No acquisition instructions for minors. Use grounded facts only.",
        "user_query": query,
        "recent_turns": ContextManager.recent_turns_text(ctx.recent_turns),
        "summary": ctx.summary_text,
        "memory": ctx.memory_text,
        "rag_evidence": evidence,
        "followup_link": ctx.linked_followup,
    }
    budget = budgeter.fit(
        sections,
        priorities=[
            "system_prompt", "safety_rules", "user_query", "recent_turns",
            "summary", "memory", "rag_evidence", "followup_link",
        ],
    )

    messages = prompt_builder.build_messages(
        safety_rules=budget.sections["safety_rules"],
        user_query=budget.sections["user_query"],
        recent_turns_text=budget.sections["recent_turns"],
        summary_text=budget.sections["summary"],
        memory_text=budget.sections["memory"],
        rag_evidence=budget.sections["rag_evidence"],
        followup_link=budget.sections["followup_link"],
    )

    yield _sse("status", {"step": "generation", "message": "Generating response"})
    stream = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=420,
        stream=True,
        timeout=120,
    )
    full = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full += delta
            yield _sse("token", {"text": delta})
            await asyncio.sleep(0)

    cleaned = re.sub(r"\n{3,}", "\n\n", full).strip()
    if _looks_incomplete(cleaned):
        completion_messages = [
            {"role": "system", "content": MEDICAL_PREFIX},
            {
                "role": "user",
                "content": (
                    f"Complete this response naturally without repeating earlier points.\n\n"
                    f"Original question: {query}\n\n"
                    f"Current draft:\n{cleaned}\n\n"
                    "Write only the missing continuation in 1-3 short sentences."
                ),
            },
        ]
        completion_rsp = openai_client.chat.completions.create(
            model=MODEL,
            messages=completion_messages,
            temperature=0.1,
            max_tokens=120,
            timeout=60,
        )
        extra = (completion_rsp.choices[0].message.content or "").strip() if completion_rsp.choices else ""
        if extra:
            cleaned = f"{cleaned} {extra}".strip()
            yield _sse("token", {"text": f" {extra}"})
    turn_index = len(history) + 1
    store.append_message(session_id, "user", query, turn_index)
    store.append_message(session_id, "assistant", cleaned, turn_index + 1)

    updated_summary = summarizer.update(summary, store.get_messages(session_id, limit=250))
    store.upsert_summary(session_id, updated_summary)
    store.log_metric(
        session_id=session_id,
        tokens_in=budget.tokens_in,
        tokens_out=max(1, len(cleaned) // 4),
        latency_ms=int((time.time() - t0) * 1000),
        summary_used=bool(ctx.summary_text and ctx.summary_text != "{}"),
        rag_docs_used=len(rag_docs),
        trimmed_context=budget.trimmed_context,
        cache_hit=False,
    )
    yield _sse(
        "done",
        {
            "final": cleaned,
            "session_id": session_id,
            "rag_docs_used": len(rag_docs),
            "trimmed_context": budget.trimmed_context,
        },
    )


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}


@app.get("/chat/stream")
async def chat_stream(
    session_id: str = Query(..., min_length=3, max_length=64),
    user_age: int = Query(18, ge=14, le=100),
    query: str = Query(..., min_length=1, max_length=1500),
):
    return StreamingResponse(
        _generate_stream(session_id=session_id, user_age=user_age, query=query),
        media_type="text/event-stream",
    )


@app.post("/chat")
async def chat(payload: Dict):
    session_id = str(payload.get("session_id") or "").strip()
    query = str(payload.get("query") or "").strip()
    user_age = int(payload.get("user_age") or 18)
    if not session_id or not query:
        raise HTTPException(status_code=400, detail="session_id and query are required")
    if user_age < 14 or user_age > 100:
        raise HTTPException(status_code=400, detail="user_age must be between 14 and 100")

    collected = []
    async for line in _generate_stream(session_id, user_age, query):
        collected.append(line)
    final = ""
    for item in collected:
        if item.startswith("event: done"):
            try:
                data_line = item.split("data: ", 1)[1].strip()
                final = json.loads(data_line).get("final", "")
            except Exception:
                pass
    return JSONResponse({"response": final})
