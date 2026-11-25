"""Configuration, folder layout, and the single graph file the app reads.

Two jobs:
  1. load .env by hand (no third-party package needed) and expose the settings
  2. say where the data lives, and where the one routing graph is

Folder layout
-------------
    data/raw_data/       what Overpass and the PolyU websites published, untouched
    data/authored/       the GeoJSON you draw and edit in geojson.io
    data/processed_data/ campus_graph.json -- built from data/authored/, read by the app

One graph, no fallback. If campus_graph.json or campus_shapes.geojson is missing, the
app stops and names the file to create. A navigator that silently routes on invented
geometry is worse than one that refuses to start.

Precedence for every setting: real environment variable > .env file > default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_data"
AUTHORED_DIR = DATA_DIR / "authored"
PROCESSED_DIR = DATA_DIR / "processed_data"
ENV_PATH = PROJECT_ROOT / ".env"

NODES_FILE = AUTHORED_DIR / "campus_pois.geojson"  # Points: PolyU campus-map amenities
SHAPES_FILE = AUTHORED_DIR / "campus_buildings.geojson"  # Polygons: the structures
PATHS_FILE = AUTHORED_DIR / "campus_paths.geojson"    # Lines: the walkways you drew

GRAPH_FILE = PROCESSED_DIR / "campus_graph.json"

DEFAULT_MODEL = "deepseek-chat"

# Place-name matching is dense retrieval: the question is embedded, and the nearest
# place names are read back from the vector index. Any OpenAI-compatible
# POST {base}/embeddings endpoint works; the default is Alibaba Cloud Model Studio
# (DashScope), whose embedding models are multilingual -- which matters here because a
# campus place name is published in English, Traditional Chinese and Simplified Chinese.
DEFAULT_EMBEDDINGS_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDINGS_MODEL = "text-embedding-v4"
DEFAULT_PINECONE_INDEX = "campus-navigator-places"


def rel(path: Path) -> str:
    """Path relative to the project root, for messages that a person reads."""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace(chr(92), "/")
    except ValueError:
        return str(path)


# ------- settings
def load_env(path: Path = ENV_PATH) -> None:
    """Minimal .env reader: KEY=VALUE lines, # comments, optional quotes.

    Values already present in the real environment are never overwritten.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # A trailing comment documents the setting; it is not part of its value. Without
        # this, "LLM_ENABLED=1  # 0 = grammar only." sets the flag to a string that is not
        # "1", and every int() below silently returns its default instead.
        for marker in (" #", "\t#"):
            if marker in value:
                value = value.split(marker, 1)[0]
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    llm_enabled: bool
    timeout_s: int
    narrate: bool = True
    base_url: str = "https://api.deepseek.com/v1"
    # -- place-name retrieval (embeddings + Pinecone), not the fact knowledge base
    embeddings_api_key: str = ""
    embeddings_model: str = DEFAULT_EMBEDDINGS_MODEL
    embeddings_base_url: str = DEFAULT_EMBEDDINGS_BASE_URL
    pinecone_api_key: str = ""
    pinecone_index: str = DEFAULT_PINECONE_INDEX
    pinecone_namespace: str = ""
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    place_top_k: int = 8
    place_min_similarity: float = 0.35
    place_ambiguity_band: float = 0.04

    @property
    def llm_available(self) -> bool:
        """True only when the LLM layer is switched on AND a key is configured."""
        return self.llm_enabled and bool(self.api_key)

    @property
    def vectors_available(self) -> bool:
        """Whether the dense index can be used at all.

        False is not an error: the place index falls back to the built-in lexical
        scorer (campus_nav/lexical.py), so the app still resolves place names offline.
        """
        return bool(self.embeddings_api_key and self.pinecone_api_key)

    @property
    def narration_available(self) -> bool:
        """True when the model is allowed to write the reply (Level B)."""
        return self.llm_available and self.narrate

    @property
    def mode_label(self) -> str:
        if self.llm_available and self.narrate:
            return "LLM understands and narrates (Level B), grammar fallback"
        if self.llm_available:
            return "LLM understands, template replies (Level A)"
        return "grammar only (no API key)"


def get_settings() -> Settings:
    load_env()
    return Settings(
        api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
        model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        llm_enabled=_flag("LLM_ENABLED", True),
        timeout_s=_int("LLM_TIMEOUT", 20),
        narrate=_flag("LLM_NARRATE", True),
        embeddings_api_key=os.environ.get("EMBEDDINGS_API_KEY", "").strip(),
        embeddings_model=(
            os.environ.get("EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL).strip()
            or DEFAULT_EMBEDDINGS_MODEL
        ),
        embeddings_base_url=(
            (os.environ.get("EMBEDDINGS_BASE_URL", DEFAULT_EMBEDDINGS_BASE_URL).strip()
             or DEFAULT_EMBEDDINGS_BASE_URL).rstrip("/")
        ),
        pinecone_api_key=os.environ.get("PINECONE_API_KEY", "").strip(),
        pinecone_index=(
            os.environ.get("PINECONE_INDEX", DEFAULT_PINECONE_INDEX).strip()
            or DEFAULT_PINECONE_INDEX
        ),
        pinecone_namespace=os.environ.get("PINECONE_NAMESPACE", "").strip(),
        pinecone_cloud=os.environ.get("PINECONE_CLOUD", "aws").strip() or "aws",
        pinecone_region=os.environ.get("PINECONE_REGION", "us-east-1").strip() or "us-east-1",
        place_top_k=_int("PLACE_TOP_K", 8),
        place_min_similarity=_float("PLACE_MIN_SIMILARITY", 0.35),
        place_ambiguity_band=_float("PLACE_AMBIGUITY_BAND", 0.04),
    )


# ------- the one graph
def graph_path() -> Path:
    """The routing graph, or a stop with the exact command that creates it."""
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            "No routing graph.\n"
            f"  expected: {rel(GRAPH_FILE)}\n"
            "  create it:  python tools/build.py graph"
        )
    return GRAPH_FILE


def shapes_path() -> Path:
    """The 2.5D map geometry, or a stop saying how to make one."""
    if not SHAPES_FILE.exists():
        raise FileNotFoundError(
            "No map geometry.\n"
            f"  expected: {rel(SHAPES_FILE)}\n"
            "  draw the buildings in geojson.io and save them there, or start from\n"
            "  the Overpass extract:  python tools/build.py shapes"
        )
    return SHAPES_FILE

