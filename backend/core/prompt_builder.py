from typing import Dict, List


class PromptBuilder:
    """Composes role-separated chat messages from layered context."""

    def __init__(self, base_system_prompt: str):
        self.base_system_prompt = base_system_prompt.strip()

    def build_messages(
        self,
        safety_rules: str,
        user_query: str,
        recent_turns_text: str,
        summary_text: str,
        memory_text: str,
        rag_evidence: str,
        followup_link: str,
    ) -> List[Dict[str, str]]:
        developer_rules = (
            "Follow these rules strictly:\n"
            f"{(safety_rules or '').strip()}\n"
            "- Use only grounded evidence for medical claims.\n"
            "- If evidence is missing, state uncertainty briefly.\n"
            "- Keep tone empathetic and practical."
        )
        user_payload = (
            f"User query:\n{user_query}\n\n"
            f"Recent turns:\n{recent_turns_text or '(none)'}\n\n"
            f"Rolling summary JSON:\n{summary_text or '{}'}\n\n"
            f"User memory JSON:\n{memory_text or '{}'}\n\n"
            f"Follow-up anchor:\n{followup_link or '(none)'}\n\n"
            f"RAG evidence:\n{rag_evidence or '(none)'}\n\n"
            "Write the best possible final answer now."
        )
        return [
            {"role": "system", "content": self.base_system_prompt},
            {"role": "developer", "content": developer_rules},
            {"role": "user", "content": user_payload},
        ]

