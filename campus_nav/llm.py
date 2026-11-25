"""DeepSeek chat client -- standard library only.

Why no `openai` package: DeepSeek speaks the OpenAI-compatible protocol, so a
single urllib POST is enough. Fewer dependencies means the demo runs on any
machine without a network install step.

Design notes that matter for the report
---------------------------------------
* Two callers, one transport. The slot parser asks for JSON in the prompt and
  parses it defensively (`extract_json`); the map assistant (`agent.py`) uses
  native tool-calling. Both read the same `LLMResult`.
* The API rate-limits under load, so 429/5xx are retried with exponential
  backoff, and every failure is returned as a `LLMResult(ok=False)` instead of
  raising -- the dialogue layer always has a grammar fallback to fall back to.
* Nothing here ever routes or answers a factual question. The model's only job is
  to turn a sentence into slots. The router decides routes; the knowledge base
  supplies facts. That separation is deliberate and worth saying in Q&A.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import Settings


@dataclass
class LLMResult:
    ok: bool
    text: str = ""
    error: str = ""
    model: str = ""
    latency_ms: int = 0
    attempts: int = 0
    usage: dict = field(default_factory=dict)
    # OpenAI-style tool calls, exactly as the provider returned them. Empty when the model
    # answered in prose instead -- which is the normal end of an agent turn.
    tool_calls: list[dict] = field(default_factory=list)

    def json(self) -> dict | None:
        return extract_json(self.text) if self.ok else None


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply.

    Models wrap JSON in prose or code fences even when told not to, so this is
    intentionally forgiving: try the whole string, then fenced blocks, then the
    first balanced {...} span.
    """
    if not text:
        return None
    candidates = [text.strip()]

    for fence in ("```json", "```"):
        if fence in text:
            for part in text.split(fence)[1::2]:
                candidates.append(part.strip())

    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break

    for cand in candidates:
        cand = cand.strip().rstrip(".").strip()
        if not cand.startswith("{"):
            continue
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


class DeepSeekClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # -- low level ---------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 400,
        retries: int = 3,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResult:
        if not self.settings.llm_available:
            return LLMResult(ok=False, error="LLM disabled or no DEEPSEEK_API_KEY set")

        url = f"{self.settings.base_url}/chat/completions"
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            # "auto" is the only choice worth offering here: the model must be free to answer
            # a greeting without calling anything.
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        started = time.monotonic()
        last_error = ""
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.settings.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                choice = (body.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
                calls = message.get("tool_calls") or []
                ok = bool(text.strip()) or bool(calls)
                return LLMResult(
                    ok=ok,
                    text=text,
                    model=body.get("model", self.settings.model),
                    latency_ms=int((time.monotonic() - started) * 1000),
                    attempts=attempt,
                    usage=body.get("usage") or {},
                    error="" if ok else "empty completion",
                    tool_calls=list(calls),
                )
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:200]
                except Exception:  # noqa: BLE001
                    pass
                last_error = f"HTTP {exc.code}: {detail}"
                # 4xx other than rate limiting will not fix themselves.
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < retries:
                time.sleep(min(2 ** attempt * 0.5, 6.0))

        return LLMResult(
            ok=False,
            error=last_error or "unknown error",
            latency_ms=int((time.monotonic() - started) * 1000),
            attempts=retries,
        )
