from typing import Dict, List


def _extract_goals(text: str) -> List[str]:
    cues = ["quit", "stop smoking", "protect my child", "secondhand smoke", "reduce"]
    lower = text.lower()
    return [c for c in cues if c in lower]


class ConversationSummarizer:
    """Rule-first summarizer with deterministic JSON shape."""

    def empty(self) -> Dict:
        return {
            "goals": [],
            "health_facts": [],
            "open_questions": [],
            "commitments": [],
            "tone_preferences": [],
        }

    def should_refresh(self, total_messages: int, last_summary_at: int, overflow: bool) -> bool:
        if overflow:
            return True
        return (total_messages - last_summary_at) >= 6

    def update(self, summary: Dict, history: List[Dict[str, str]]) -> Dict:
        base = self.empty()
        base.update(summary or {})
        recent = history[-12:]
        for msg in recent:
            role = msg.get("role", "")
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            if role == "user":
                for goal in _extract_goals(text):
                    if goal not in base["goals"]:
                        base["goals"].append(goal)
                if "?" in text and text not in base["open_questions"]:
                    base["open_questions"].append(text[:180])
                if any(k in text.lower() for k in ["please be brief", "short", "simple", "detailed"]):
                    pref = text[:120]
                    if pref not in base["tone_preferences"]:
                        base["tone_preferences"].append(pref)
            else:
                if any(k in text.lower() for k in ["cdc", "who", "nhs", "asthma", "sids", "nicotine"]):
                    fact = text[:180]
                    if fact not in base["health_facts"]:
                        base["health_facts"].append(fact)
                if any(k in text.lower() for k in ["you can", "step", "plan", "call"]):
                    action = text[:180]
                    if action not in base["commitments"]:
                        base["commitments"].append(action)

        for key in ("goals", "health_facts", "open_questions", "commitments", "tone_preferences"):
            base[key] = base[key][-8:]
        return base

