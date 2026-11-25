"""Pinecone REST client -- standard library only, no `pinecone` package.

Two planes, as Pinecone splits them:

  * control  https://api.pinecone.io/indexes        create / describe an index
  * data     https://{index host}/query             read and write vectors

The index host is discovered once per process from the control plane and cached, so a
conversation costs one query request per place phrase and no extra round-trip.

Every failure raises `PineconeError`. The caller decides what to do with it:
`place_index.PlaceIndex` falls back to the built-in lexical scorer
(`campus_nav/lexical.py`), so a dead index costs semantic matching, not the app.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .config import Settings

CONTROL_PLANE = "https://api.pinecone.io"
API_VERSION = "2024-07"
UPSERT_BATCH = 100


class PineconeError(RuntimeError):
    """The index could not be reached, created, read or written."""


class PineconeStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._host: str | None = None

    # -- transport ---------------------------------------------------------
    def _call(self, method: str, url: str, payload: dict | None = None,
              *, retries: int = 3, allow_404: bool = False) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {
            "Api-Key": self.settings.pinecone_api_key,
            "X-Pinecone-API-Version": API_VERSION,
            "Accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"

        last_error = ""
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.settings.timeout_s) as resp:
                    raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                except Exception:  # noqa: BLE001 - the body is only for the message
                    pass
                if exc.code == 404 and allow_404:
                    return {}
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < retries:
                time.sleep(min(2 ** attempt * 0.5, 6.0))

        raise PineconeError(f"{method} {url} failed: {last_error or 'unknown error'}")

    # -- control plane -----------------------------------------------------
    def describe(self) -> dict | None:
        name = self.settings.pinecone_index
        return self._call(
            "GET", f"{CONTROL_PLANE}/indexes/{name}", allow_404=True
        ) or None

    def host(self) -> str:
        """The data-plane host for the configured index, discovered once."""
        if self._host:
            return self._host
        info = self.describe()
        if not info or not info.get("host"):
            raise PineconeError(
                f"Pinecone index {self.settings.pinecone_index!r} does not exist. "
                "Create and fill it with: python tools/build.py embed"
            )
        self._host = str(info["host"])
        return self._host

    def ensure(self, dimension: int, *, recreate: bool = False) -> dict:
        """Describe the index, creating it with `dimension` and cosine metric if absent."""
        name = self.settings.pinecone_index
        info = self.describe()
        if info is None:
            info = self._call(
                "POST",
                f"{CONTROL_PLANE}/indexes",
                {
                    "name": name,
                    "dimension": dimension,
                    "metric": "cosine",
                    "spec": {
                        "serverless": {
                            "cloud": self.settings.pinecone_cloud,
                            "region": self.settings.pinecone_region,
                        }
                    },
                },
            )
            info = self._wait_ready(name, info)
        elif int(info.get("dimension") or 0) != dimension:
            raise PineconeError(
                f"Pinecone index {name!r} has dimension {info.get('dimension')}, "
                f"but {self.settings.embeddings_model} produces {dimension}. "
                "Recreate it with --recreate, or point PINECONE_INDEX at another index."
            )
        self._host = str(info.get("host") or self._host or "") or None
        if recreate:
            self.delete_all()
        return info

    def _wait_ready(self, name: str, info: dict, *, seconds: int = 90) -> dict:
        if (info.get("status") or {}).get("ready"):
            return info
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(2)
            info = self.describe() or {}
            if (info.get("status") or {}).get("ready"):
                return info
        raise PineconeError(
            f"Pinecone index {name!r} was created but is not ready yet. Retry in a minute."
        )

    # -- data plane --------------------------------------------------------
    def stats(self) -> dict:
        return self._call("POST", f"https://{self.host()}/describe_index_stats")

    def upsert(self, vectors: list[dict]) -> int:
        """Write `{"id", "values", "metadata"}` records; returns how many were accepted."""
        namespace = self.settings.pinecone_namespace
        written = 0
        for start in range(0, len(vectors), UPSERT_BATCH):
            batch = vectors[start : start + UPSERT_BATCH]
            result = self._call(
                "POST",
                f"https://{self.host()}/vectors/upsert",
                {"vectors": batch, "namespace": namespace},
            )
            written += int(result.get("upsertedCount") or len(batch))
        return written

    def query(self, vector: list[float], top_k: int) -> list[tuple[float, dict]]:
        """Nearest place records, best first, as `(score, metadata)`."""
        result = self._call(
            "POST",
            f"https://{self.host()}/query",
            {
                "vector": vector,
                "topK": top_k,
                "includeMetadata": True,
                "namespace": self.settings.pinecone_namespace,
            },
        )
        return [
            (float(match.get("score") or 0.0), match.get("metadata") or {})
            for match in (result.get("matches") or [])
        ]

    def delete_all(self) -> None:
        """Empty the namespace. A namespace that does not exist yet is already empty.

        Pinecone answers 404 "Namespace not found" for the default namespace of a freshly
        created index, and `--recreate` on a first build would otherwise fail before it
        wrote anything.
        """
        self._call(
            "POST",
            f"https://{self.host()}/vectors/delete",
            {"deleteAll": True, "namespace": self.settings.pinecone_namespace},
            allow_404=True,
        )
