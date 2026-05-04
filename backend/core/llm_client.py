import time
from typing import Dict, List, Tuple


class LLMClient:
    """OpenAI chat wrapper that returns text + metric-friendly counters."""

    def __init__(self, openai_client):
        self.client = openai_client

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ) -> Tuple[str, int, int, int]:
        start = time.time()
        if stream:
            rsp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                timeout=120,
            )
            content = ""
            for chunk in rsp:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    content += delta
            latency_ms = int((time.time() - start) * 1000)
            tokens_in = sum(max(1, len((m.get("content") or "")) // 4) for m in messages)
            tokens_out = max(1, len(content) // 4) if content else 0
            return content.strip(), tokens_in, tokens_out, latency_ms

        rsp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
        )
        content = rsp.choices[0].message.content if rsp.choices else ""
        latency_ms = int((time.time() - start) * 1000)
        usage = getattr(rsp, "usage", None)
        if usage is not None:
            tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
            tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        else:
            tokens_in = sum(max(1, len((m.get("content") or "")) // 4) for m in messages)
            tokens_out = max(1, len((content or "")) // 4) if content else 0
        return (content or "").strip(), tokens_in, tokens_out, latency_ms

