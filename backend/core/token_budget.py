from dataclasses import dataclass
from typing import Dict, List


def rough_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class BudgetResult:
    sections: Dict[str, str]
    trimmed_context: bool
    tokens_in: int


class TokenBudgeter:
    """Priority-based token allocation and trimming."""

    def __init__(self, max_input_tokens: int = 1800):
        self.max_input_tokens = max_input_tokens

    def fit(self, sections: Dict[str, str], priorities: List[str]) -> BudgetResult:
        final = dict(sections)
        trimmed = False

        def total() -> int:
            return sum(rough_token_count(v) for v in final.values())

        if total() <= self.max_input_tokens:
            return BudgetResult(sections=final, trimmed_context=False, tokens_in=total())

        for key in reversed(priorities):
            if key in ("system_prompt", "safety_rules", "user_query"):
                continue
            text = final.get(key, "")
            if not text:
                continue
            while text and total() > self.max_input_tokens:
                text = text[: int(len(text) * 0.85)].rstrip()
                final[key] = text
                trimmed = True
            if total() <= self.max_input_tokens:
                break

        return BudgetResult(sections=final, trimmed_context=trimmed, tokens_in=total())

