"""Dense embeddings from an OpenAI-compatible endpoint, standard library only.

GitHub Models' inference API -- the service whose token this module reads as
`GITHUB_TOKEN` -- was retired on 2026-07-30. The endpoint is therefore a setting
(`EMBEDDINGS_BASE_URL`) rather than a constant, defaulting to GitHub Models' URL so an
existing `.env` keeps its meaning, and repointable at any endpoint that speaks
`POST {base}/embeddings` and returns `{"data": [{"index": 0, "embedding": [...]}]}`.

Nothing here is retried silently into a wrong answer: every failure raises
`EmbeddingError`, and the caller decides. A place name resolved from a stale or
partial embedding would be worse than an error the user can see.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .config import Settings

# Endpoints cap the number of inputs per request, and the cap is not uniform: Alibaba
# Cloud's text-embedding-v4 accepts 10 rows while its Qwen3 embedding models accept 20.
# Ten is legal everywhere, and 284 names cost 29 requests instead of 18 -- irrelevant
# next to being rejected by the one model somebody happens to configure.
BATCH_SIZE = 10


class EmbeddingError(RuntimeError):
    """The endpoint could not turn text into vectors."""


class EmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def model(self) -> str:
        return self.settings.embeddings_model

    def embed(self, texts: list[str], *, retries: int = 3) -> list[list[float]]:
        """One vector per input, in input order."""
        if not texts:
            return []
        if not self.settings.embeddings_api_key:
            raise EmbeddingError(
                "No embedding key. Set EMBEDDINGS_API_KEY (or GITHUB_TOKEN) in .env."
            )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._request(texts[start : start + BATCH_SIZE], retries))
        return vectors

    def embed_one(self, text: str, *, retries: int = 3) -> list[float]:
        return self.embed([text], retries=retries)[0]

    # -- low level ---------------------------------------------------------
    def _request(self, batch: list[str], retries: int) -> list[list[float]]:
        url = f"{self.settings.embeddings_base_url}/embeddings"
        payload = {"model": self.model, "input": batch}
        headers = {
            "Authorization": f"Bearer {self.settings.embeddings_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error = ""
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self.settings.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                rows = sorted(body.get("data") or [], key=lambda row: row.get("index", 0))
                if len(rows) != len(batch):
                    raise EmbeddingError(
                        f"expected {len(batch)} embeddings, received {len(rows)}"
                    )
                return [[float(value) for value in row["embedding"]] for row in rows]
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:200]
                except Exception:  # noqa: BLE001 - the body is only for the message
                    pass
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < retries:
                time.sleep(min(2 ** attempt * 0.5, 6.0))

        raise EmbeddingError(f"{url} failed: {last_error or 'unknown error'}")
