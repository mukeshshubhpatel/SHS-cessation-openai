import hashlib
import re
from typing import Dict, List, Tuple


class RetrievalManager:
    """Retrieval rerank + dedupe + cache-key helpers."""

    def __init__(self, store):
        self.store = store

    @staticmethod
    def make_cache_key(age_category: str, query: str) -> str:
        norm = re.sub(r"\s+", " ", (query or "").strip().lower())
        return hashlib.md5(f"{age_category}:{norm}".encode("utf-8")).hexdigest()

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        sa = set(re.findall(r"\w+", a.lower()))
        sb = set(re.findall(r"\w+", b.lower()))
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    def rerank_dedupe(self, query: str, docs: List[Dict], keep: int = 5) -> List[Dict]:
        q = (query or "").lower()
        ranked: List[Tuple[float, Dict]] = []
        for d in docs:
            text = (d.get("text") or "").strip()
            if not text:
                continue
            base = float(d.get("score", 0.0))
            overlap = self._jaccard(q, text[:800])
            score = (base * 0.55) + (overlap * 1.25)
            if overlap < 0.05:
                score -= 0.35
            ranked.append((score, d))
        ranked.sort(key=lambda x: x[0], reverse=True)

        result: List[Dict] = []
        for _, doc in ranked:
            text = (doc.get("text") or "").strip()
            if any(self._jaccard(text[:700], (x.get("text") or "")[:700]) > 0.82 for x in result):
                continue
            result.append(doc)
            if len(result) >= keep:
                break
        return result

    @staticmethod
    def evidence_block(docs: List[Dict], max_items: int = 5) -> str:
        lines = []
        for i, d in enumerate(docs[:max_items], 1):
            source = (d.get("verified_source") or d.get("source") or "Unknown").strip()
            score = float(d.get("score", 0.0))
            snippet = re.sub(r"\s+", " ", (d.get("text") or "").strip())[:360]
            if snippet:
                lines.append(f"[{i}] {source} | relevance={score:.3f} | {snippet}")
        return "\n".join(lines)
