import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


FOLLOW_UP_PATTERNS = [
    r"^what about",
    r"^and ",
    r"^also",
    r"^side effects\??$",
    r"^dosage\??$",
    r"^and cravings\??$",
    r"^cravings\??$",
    r"^why\??$",
    r"^how\??$",
]


@dataclass
class ContextBundle:
    recent_turns: List[Dict[str, str]]
    summary_text: str
    memory_text: str
    linked_followup: str


class ContextManager:
    def __init__(self, recent_min: int = 4, recent_max: int = 8):
        self.recent_min = recent_min
        self.recent_max = recent_max

    def is_follow_up(self, query: str) -> bool:
        q = (query or "").strip().lower()
        return any(re.search(p, q) for p in FOLLOW_UP_PATTERNS)

    def build(
        self,
        history: List[Dict[str, str]],
        summary: Dict,
        memory: Dict,
        query: str,
    ) -> ContextBundle:
        recent_count = max(self.recent_min, min(self.recent_max, len(history)))
        recent = history[-recent_count:] if history else []
        summary_text = json.dumps(summary or {}, ensure_ascii=True)
        memory_text = json.dumps(memory or {}, ensure_ascii=True)

        linked = ""
        if self.is_follow_up(query):
            for msg in reversed(history[-10:]):
                if msg.get("role") == "assistant":
                    linked = (msg.get("text") or "")[:280]
                    break
        return ContextBundle(
            recent_turns=recent,
            summary_text=summary_text,
            memory_text=memory_text,
            linked_followup=linked,
        )

    @staticmethod
    def recent_turns_text(turns: List[Dict[str, str]]) -> str:
        lines = []
        for t in turns:
            role = "User" if t.get("role") == "user" else "Assistant"
            text = (t.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text[:350]}")
        return "\n".join(lines)

