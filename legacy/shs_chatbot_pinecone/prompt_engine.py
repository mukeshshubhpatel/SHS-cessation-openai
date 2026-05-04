"""Stub for prompt_engine — real implementation not present."""
from typing import Any, List


class PromptEngine:
    def build(self, query: str, contexts: List[Any] = []) -> str:
        return query


prompt_engine = PromptEngine()
