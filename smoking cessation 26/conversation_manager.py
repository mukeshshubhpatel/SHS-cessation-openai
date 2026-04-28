"""Stub for conversation_manager — real implementation not present."""
from typing import Any, List, Optional


class ConversationState:
    def __init__(self):
        self.history: List[dict] = []


class ConversationManager:
    def get_state(self) -> ConversationState:
        return ConversationState()

    def reset(self) -> None:
        pass


conversation_manager = ConversationManager()
