"""A self-hosted web interface for the navigator -- standard library only.

    python -m campus_nav.web               # http://127.0.0.1:8000
    python -m campus_nav.web --port 9000   # somewhere else
    python -m campus_nav.web --open        # also launch the browser

Why http.server and not FastAPI/Flask
-------------------------------------
The whole project installs nothing, and a live demo is the worst possible moment to
discover that a virtualenv is broken. http.server ships with Python, so
"unplug the network, run one command, show the app" works on any machine with
Python 3.10+. The API is four endpoints and about 200 lines, which is smaller than
the dependency-management risk it replaces.

The pages are built from web/ (React + TypeScript) into campus_nav/static/, and the
built files are committed. Nothing here needs npm at run time, and nothing is
fetched from a CDN -- so the demo also works offline, which matters because the
speech-recognition fallback and the map must both keep working when the
lecture-room Wi-Fi does not.

Endpoints
---------
    GET  /                 the application shell
    GET  /static/<file>    its assets
    GET  /api/status       engine mode, graph source, knowledge-base coverage
    GET  /api/graph        nodes + edges, for drawing the map
    POST /api/chat         {"text": "...", "lang": "zh-HK"} -> one reply (lang optional)
    POST /api/route-draft   preview the start and destination without sending the question
    GET  /api/speech-status   local Whisper availability and maximum recording length
    POST /api/transcribe?lang=zh-HK    PCM WAV -> editable transcript
    POST /api/reset        clear the conversation state
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import sys
import threading
import webbrowser
from collections import OrderedDict
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .config import (
    ENV_PATH,
    GRAPH_FILE,
    RAW_DIR,
    get_settings,
    graph_path,
    rel,
    shapes_path,
)
from .dialogue import Dialogue
from .embeddings import EmbeddingError
from .format import local_hours
from .render import facility_details, facility_speech
from .graph import CampusGraph
from .kb import KnowledgeBase
from .landmarks import landmarks_along
from .llm import DeepSeekClient
from .local_names import building_name, floor_name, localize_label, place_name
from .narrate import Narrator
from .pinecone import PineconeError
from . import render
from .place_index import PlaceIndex, PlaceIndexError
from .places import feature_height, shape_kind
from .speech import MAX_WAV_BYTES, LocalTranscriber, SpeechError

STATIC_DIR = Path(__file__).resolve().parent / "static"


def load_shapes(path) -> list[dict]:
    """Read the map geometry you drew in geojson.io.

    data/authored/campus_buildings1.geojson is the whole 2.5D layer: every polygon you
    keep is drawn, every polygon you delete is gone. Nothing is clipped, filtered or
    guessed afterwards -- if it is in the file, it is deliberate.

        {"type": "Feature",
         "properties": {"name": "Block Z", "height": 28},
         "geometry": {"type": "Polygon",
                      "coordinates": [[[114.1781, 22.3050], [114.1783, 22.3050], ...]]}}

    Two things worth remembering: GeoJSON lists coordinates as [longitude, latitude]
    (the opposite of how they are usually spoken), and the first and last point of a
    ring must be identical.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    def rings_of(geometry: dict) -> list[dict]:
        """Polygons AND lines. A drawn path (LineString) is a shape: it renders as a
        polyline instead of being silently ignored."""
        kind = (geometry or {}).get("type")
        coords = (geometry or {}).get("coordinates") or []
        if kind == "Polygon":
            return [{"ring": coords[0], "line": False}] if coords else []
        if kind == "MultiPolygon":
            return [{"ring": polygon[0], "line": False} for polygon in coords if polygon]
        if kind == "LineString":
            return [{"ring": coords, "line": True}] if len(coords) >= 2 else []
        if kind == "MultiLineString":
            return [{"ring": line, "line": True} for line in coords if len(line) >= 2]
        return []

    rings = []
    broken: list[str] = []
    for index, feature in enumerate(raw.get("features", [])):
        props = feature.get("properties") or {}
        for shape in rings_of(feature.get("geometry") or {}):
            ring, is_line = shape["ring"], shape["line"]
            points = [[float(point[1]), float(point[0])] for point in ring if len(point) >= 2]
            if len(points) < (2 if is_line else 3):
                broken.append(str(props.get("id") or props.get("name") or f"feature #{index}"))
                continue
            entry = {
                "id": str(props.get("id") or f"drawn-{index}"),
                "ring": points,
                # English first, because a Chinese-first name sorts and reads worse in
                # the tooltip: "賽馬會創新樓 Jockey Club Innovation Tower"
                "name": str(props.get("name:en") or props.get("name") or ""),
                # "plaza" / "green" / "pitch" / "water" are drawn FLAT: a square, a lawn
                # or a pool has no height to extrude, and saying otherwise would be a lie
                "kind": str(props.get("kind") or shape_kind(props, is_line)),
                "line": is_line,
            }
            height = feature_height(props)
            if height:
                entry["height"] = height
            # "tone": "grey" is the one way a block may be drawn differently from the
            # rest of the campus; the value is passed through untouched.
            if props.get("tone"):
                entry["tone"] = str(props["tone"])
            # The letter to print on this structure: an explicit "display" column if you
            # ever add one, otherwise the campus block code already in "ref" -- L for the
            # library, VA for Shaw Amenities, Y for the Lee Shau Kee Building, V for the
            # Innovation Tower. No ref and no display means no letter at all: the map
            # must never abbreviate a name into a code of its own.
            label = (str(props.get("display") or "").strip()
                     or str(props.get("ref") or "").strip())
            if label:
                entry["display"] = label
            rings.append(entry)
    if broken:
        print(f"[map] {len(broken)} feature(s) in {Path(path).name} cannot be drawn "
              f"(too few points): {', '.join(broken[:6])}")
    return rings


# One dialogue object serves one conversation. The server is a demo, not a service:
# a lock keeps two browser tabs from interleaving their state, and /api/reset starts
# a fresh conversation. A production deployment would key state by session cookie.
LOCK = threading.Lock()


# --------------------------------------------------------------------- payloads
def _oriented_segments(graph: CampusGraph, route) -> list[tuple]:
    """Each segment's geometry walked in the direction the route travels it.

    An edge stores its polyline one fixed way (u -> v); appending the segments in that
    order whatever the travel throws a straight chord across every bend taken
    backwards, which is what drew long straight lines through the middle of a route.
    """
    cursor = None
    if route.path:
        start = graph.node(route.path[0])
        cursor = [start.lat, start.lng]
    oriented = []
    for seg in route.segments:
        points = [list(point) for point in (seg.path or [])]
        if len(points) < 2:
            points = [[graph.node(seg.u).lat, graph.node(seg.u).lng],
                      [graph.node(seg.v).lat, graph.node(seg.v).lng]]
        if cursor is not None:
            head = (points[0][0] - cursor[0]) ** 2 + (points[0][1] - cursor[1]) ** 2
            tail = (points[-1][0] - cursor[0]) ** 2 + (points[-1][1] - cursor[1]) ** 2
            if tail < head:
                points.reverse()
        cursor = points[-1]
        oriented.append((seg, points))
    return oriented


def route_payload(graph: CampusGraph, route, lang: str = "en-US") -> dict | None:
    """The route, with coordinates, so the browser can draw it without guessing."""
    if route is None:
        return None
    # one dot per PLACE. A junction is where the walkway bends, not a stop the user
    # should see a pin on; the drawn line still follows the junctions exactly,
    # because it comes from the segments below.
    path = [
        {
            "id": node_id,
            "name": place_name(graph, node_id, lang),
            "lat": graph.node(node_id).lat,
            "lng": graph.node(node_id).lng,
        }
        for node_id in route.path
        if node_id in graph.nodes and graph.node(node_id).kind != "waypoint"
    ]
    # Each segment's geometry is walked in the direction the route travels it, so the
    # browser can append them end to end without crossing the map.
    segments = [
        {
            "from": seg.u,
            "to": seg.v,
            "metres": round(seg.metres, 1),
            "stairs": seg.stairs,
            "path": points,
        }
        for seg, points in _oriented_segments(graph, route)
    ]

    return {
        "reachable": route.reachable,
        "avoid_stairs": bool(route.blocking.get("avoid_stairs")),
        "metres": round(route.total_metres, 1) if route.reachable else 0.0,
        "path": path,
        "segments": segments,
        "stats": (
            {
                "algorithm": route.stats.algorithm,
                "nodes_expanded": route.stats.nodes_expanded,
                "frontier_peak": route.stats.frontier_peak,
                "ms": round(route.stats.elapsed_ms, 3),
            }
            if route.stats
            else None
        ),
        "explain": route.explain[:8],
        # The blocks the walk passes, in order, so the map can put names on the line.
        "landmarks": landmarks_along(graph, route),
    }


def route_option(graph: CampusGraph, route, lang: str = "en-US") -> dict | None:
    """One pickable way there: what the chooser shows, plus the line to draw."""
    if route is None or not route.reachable or len(route.path) < 2:
        return None
    places = [node for node in route.path if graph.node(node).kind != "waypoint"]
    # The real places between the ends: the stops of a via journey, named in order, so the
    # chooser says which walk it is ("via Core P, then Block L") instead of only the first.
    via = ", ".join(place_name(graph, node, lang) for node in places[1:-1][:2])
    return {
        "metres": round(route.total_metres, 1),
        "legs": max(len(places) - 1, 1),
        "flights": sum(1 for seg in route.segments if seg.stairs),
        "covered": sum(1 for seg in route.segments if seg.covered),
        "via": via,
        "landmarks": landmarks_along(graph, route),
        "start": place_name(graph, route.path[0], lang),
        "goal": place_name(graph, route.path[-1], lang),
        "path": [
            {
                "id": node_id,
                "name": place_name(graph, node_id, lang),
                "lat": graph.node(node_id).lat,
                "lng": graph.node(node_id).lng,
            }
            for node_id in places
        ],
        "segments": [
            {"path": points} for _, points in _oriented_segments(graph, route)
        ],
    }



def _parser_label(slots) -> str:
    """'agent', or 'grammar (model abstained)' and friends: which layer decided, and why."""
    source = getattr(slots, "source", "") or ""
    note = (getattr(slots, "parse_note", "") or "").strip()
    return f"{source} ({note})" if source and note else source


def chat_payload(graph: CampusGraph, reply, lang: str = "en-US") -> dict:
    route = getattr(reply, "route", None)
    explicit = getattr(reply, "spoken", "")
    facilities = getattr(reply, "facilities", [])
    facility_lead = facility_speech(reply, graph, lang) if lang in ("zh-HK", "zh-CN") else None
    # The language the user asked in is the language they are answered in. There used to be
    # a heuristic here that kept English whenever the reply carried its own sentence --
    # which is precisely when a Chinese user was answered in English.
    speech_lang = lang
    # One renderer decides what is said: model narration if there is any, then the
    # catalogue (campus_nav/copy.py), then the reply's own words. This call used to be a
    # chain of `if lang ==` arms spread over three modules.
    speech = render.speech(reply, graph, speech_lang)
    # A reply may insist on one fact first ("Closed Catering Outlets."). It is part of the
    # answer, not a detail, so it leads and it is spoken; the wording is the handler's, so
    # the lead cannot drift from what actually changed.
    lead_prefix = (getattr(reply, "lead_prefix", "") or "").strip()
    if lead_prefix and speech:
        speech = lead_prefix + " " + speech
    # A reply that carries a route or a list of places has a short, friendly summary
    # (the same sentence that gets spoken). The UI leads with that and puts the
    # segment table / A* statistics behind a details toggle, so the interface does not
    # hand the user a receipt when a key is missing and a sentence when one is present.
    structural = facility_lead or explicit or route is not None or facilities
    cards = list(facilities or [])
    if lang in ("zh-HK", "zh-CN"):
        cards = [
            {**card,
             "display_name": place_name(graph, card["node_id"], lang) if card.get("node_id") in graph.nodes
                             else localize_label(graph, card["name"], lang),
             "display_building": building_name(graph, card.get("building", ""), lang),
             "display_floor": floor_name(card.get("floor", ""), lang),
             "display_hours": local_hours(card.get("hours_today", ""), lang)}
            for card in cards
        ]
    return {
        "text": (facility_details(reply, graph, lang) or facility_lead) if facility_lead else reply.text,
        "lead": speech if structural else "",
        "speech": speech,
        "speech_lang": speech_lang,
        "narrated": bool(getattr(reply, "narrated", False)),
        "notes": list(getattr(reply, "notes", []) or []),
        "intent": getattr(getattr(reply, "slots", None), "intent", ""),
        # which NLU implementation produced these slots, and -- when it was the grammar --
        # why the model was not the one answering
        "parser": _parser_label(getattr(reply, "slots", None)),
        "route": route_payload(graph, route, speech_lang),
        "routes": [
            option
            for option in (
                route_option(graph, route, speech_lang) for route in (getattr(reply, "routes", None) or [])
            )
            if option is not None
        ],
        "facilities": cards,
    }


# ------------------------------------------------------------------------ env file
def write_env(updates: dict[str, str], path: Path = ENV_PATH) -> None:
    """Create or update .env, preserving comments and unrelated keys.

    The browser posts a key here, so this is the one place in the project that
    writes a secret to disk. It writes only to .env, which .gitignore covers, and
    never echoes the value back.
    """
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if not existing:
        # Start from the documented template when there is one, so the file the dialog
        # creates carries every setting and its explanation -- not just the four keys the
        # dialog happens to write. Values in the template are the defaults.
        template = path.with_name(".env.example")
        if template.exists():
            lines = template.read_text(encoding="utf-8").splitlines()
            while lines and (lines[0].startswith("#") or not lines[0].strip()):
                lines.pop(0)          # the template's "copy this file" preamble
            existing = ["# Local settings. Never commit this file -- it is in .gitignore.", ""]
            existing += lines
        else:
            existing = ["# Local settings. Never commit this file -- it is in .gitignore.", ""]

    remaining = dict(updates)
    out: list[str] = []
    for line in existing:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                out.append(f"{key}={remaining.pop(key)}")
                continue
        out.append(line)
    out.extend(f"{key}={value}" for key, value in remaining.items())
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


# ----------------------------------------------------------------------- app state
class App:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.transcriber = LocalTranscriber()
        self.graph = CampusGraph.load(graph_path())
        self.knowledge = KnowledgeBase.from_raw(RAW_DIR)
        self.footprints = load_shapes(shapes_path())
        client = DeepSeekClient(self.settings) if self.settings.llm_available else None
        narrator = Narrator(client) if self.settings.narration_available else None
        self.places = PlaceIndex(self.graph, self.settings)
        self.index_info = self.places.check()
        # The graph, the knowledge base and the place index are shared: read-only and
        # expensive. The *conversation* is not shared -- two browsers, or two tabs, must
        # not overwrite each other's origin, destination or closures.
        self.client = client
        self.narrator = narrator
        self._dialogues: OrderedDict[str, Dialogue] = OrderedDict()

    MAX_SESSIONS = 32

    def dialogue_for(self, session: str) -> Dialogue:
        """This browser's conversation, created on first use and remembered after."""
        found = self._dialogues.get(session)
        if found is not None:
            self._dialogues.move_to_end(session)
            return found
        dialogue = Dialogue(self.graph, self.client, self.narrator,
                            knowledge=self.knowledge, places=self.places)
        self._dialogues[session] = dialogue
        while len(self._dialogues) > self.MAX_SESSIONS:
            self._dialogues.popitem(last=False)      # oldest browser loses its state
        return dialogue

    def apply_settings(self, api_key: str | None = None, model: str | None = None,
                       *, embeddings_api_key: str | None = None,
                       pinecone_api_key: str | None = None) -> dict:
        """Persist new settings and use them immediately, without a restart."""
        updates: dict[str, str] = {}
        changes: dict[str, str] = {}
        if api_key is not None:
            updates["DEEPSEEK_API_KEY"] = api_key.strip()
            changes["api_key"] = api_key.strip()
        if model is not None and model.strip():
            updates["DEEPSEEK_MODEL"] = model.strip()
            changes["model"] = model.strip()
        if embeddings_api_key is not None:
            updates["EMBEDDINGS_API_KEY"] = embeddings_api_key.strip()
            changes["embeddings_api_key"] = embeddings_api_key.strip()
        if pinecone_api_key is not None:
            updates["PINECONE_API_KEY"] = pinecone_api_key.strip()
            changes["pinecone_api_key"] = pinecone_api_key.strip()
        if not updates:
            return self.status()

        write_env(updates)
        for key, value in updates.items():
            os.environ[key] = value          # keep the process in step with the file
        self.settings = replace(self.settings, **changes)

        client = DeepSeekClient(self.settings) if self.settings.llm_available else None
        narrator = Narrator(client) if self.settings.narration_available else None
        self.client, self.narrator = client, narrator
        for dialogue in self._dialogues.values():
            dialogue.client = client
            dialogue.narrator = narrator

        # A new embeddings or vector-store key changes how place names are resolved, so
        # the index is re-probed now: the very next message uses the new path instead of
        # waiting for a restart.
        if "embeddings_api_key" in changes or "pinecone_api_key" in changes:
            self.index_info = self.places.reconfigure(self.settings)
        return self.status()

    def status(self) -> dict:
        return {
            "mode": self.settings.mode_label,
            "model": self.settings.model if self.settings.llm_available else "",
            "api_key_present": bool(self.settings.api_key),
            "embeddings_key_present": bool(self.settings.embeddings_api_key),
            "pinecone_key_present": bool(self.settings.pinecone_api_key),
            "embeddings_model": self.settings.embeddings_model,
            "embeddings_base_url": self.settings.embeddings_base_url,
            "graph_source": rel(GRAPH_FILE),
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "knowledge": self.knowledge.coverage(),
            "places": {
                "index": self.index_info["index"],
                "vectors": self.index_info["vectors"],
                "mode": self.index_info["mode"],
                "warning": self.index_info["warning"],
            },
            "footprints": len(self.footprints),
            "map_meta": {
                k: self.graph.meta.get(k)
                for k in ("title", "generated_at", "inputs", "real_data", "derived", "counts")
                if self.graph.meta.get(k)
            },
        }

    def graph_payload(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "names": {lang: place_name(self.graph, n.id, lang) for lang in ("zh-CN", "zh-HK")} if n.kind != "waypoint" else {},
                    "aliases": [n.name_zh] if n.name_zh else [],
                    "lat": n.lat,
                    "lng": n.lng,
                    "kind": n.kind,
                    "category": n.category,
                            "building": n.building,
                    "service": n.service,
                    "accepts": list(n.accepts),
                    "floor": n.floor,
                    # Published by PolyU, but no walkway reaches it: drawn, not routable.
                    "off_network": n.off_network,
                    "nearest_m": n.nearest_m,
                }
                for n in self.graph.nodes.values()
            ],
            "footprints": self.footprints,
            "edges": [
                {
                    "id": e.id,
                    "u": e.u,
                    "v": e.v,
                    "metres": e.metres,
                    "covered": e.covered,
                    "stairs": e.stairs,
                    "path": [list(point) for point in e.path],
                }
                for e in self.graph.edges
            ],
        }


# ----------------------------------------------------------------------- web server
class Handler(BaseHTTPRequestHandler):
    app: App  # injected by serve()

    server_version = "CampusNavigator/1.0"
    SESSION_COOKIE = "campus_session"

    # -- session -----------------------------------------------------------
    def _session(self) -> str:
        """This browser's conversation id, minted on the first request it makes.

        A cookie, because the alternative is one conversation for the whole server: with a
        single Dialogue two tabs shared one origin, and "from block L instead" in one tab
        moved the other. The cookie is opaque and per-browser; the state it points at never
        leaves this process.
        """
        if getattr(self, "_sid", None):
            return self._sid
        for part in (self.headers.get("Cookie") or "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == self.SESSION_COOKIE and value:
                self._sid, self._fresh = value, False
                return value
        self._sid = secrets.token_urlsafe(12)
        self._fresh = True
        return self._sid

    # -- helpers -----------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if getattr(self, "_fresh", False):
            self.send_header(
                "Set-Cookie",
                f"{self.SESSION_COOKIE}={self._session()}; Path=/; SameSite=Lax; HttpOnly",
            )
            self._fresh = False
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        return

    # -- routes ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/api/status":
            return self._json(self.app.status())
        if path == "/api/speech-status":
            return self._json(self.app.transcriber.status())
        if path == "/api/graph":
            return self._json(self.app.graph_payload())
        return self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/transcribe":
            return self._transcribe()
        if path in ("/api/chat", "/api/route-draft"):
            payload = self._read_json()
            if not isinstance(payload, dict):
                return self._json({"error": "Expected a question."}, 400)
            text = str(payload.get("text") or "").strip()
            if not text:
                return self._json({"error": "empty message"}, 400)
            lang = payload.get("lang", "en-US")
            if lang not in ("zh-CN", "zh-HK", "en-US"):
                lang = "en-US"
            if path == "/api/route-draft":
                try:
                    with LOCK:
                        draft = self.app.dialogue_for(self._session()).draft_endpoints(text)
                except (PlaceIndexError, EmbeddingError, PineconeError) as exc:
                    return self._json({"error": f"place index unavailable: {exc}"}, 502)
                if draft is None:
                    return self._json({"is_route": False})
                for item in draft.values():
                    item["label"] = place_name(self.app.graph, item["id"], lang) if item["id"] else item["text"]
                return self._json({"is_route": True, **draft})
            try:
                with LOCK:
                    reply = self.app.dialogue_for(self._session()).handle(
                        text, endpoints=payload.get("endpoints"), language=lang)
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            except (PlaceIndexError, EmbeddingError, PineconeError) as exc:
                # Place names come from a remote index; a failure there is a visible
                # 502, never a silently guessed place.
                return self._json({"error": f"place index unavailable: {exc}"}, 502)
            return self._json(chat_payload(self.app.graph, reply, lang))
        if path == "/api/settings":
            payload = self._read_json()
            # One shape for every key field: absent means "leave it alone", a string
            # means "store this". Nothing that arrives here is echoed back.
            limits = {
                "api_key": 200,
                "model": 120,
                "embeddings_api_key": 200,
                "pinecone_api_key": 200,
            }
            values: dict[str, str | None] = {}
            for name, limit in limits.items():
                raw = payload.get(name)
                if raw is None:
                    values[name] = None
                    continue
                text = str(raw).strip()
                if len(text) > limit or "\n" in text or "\r" in text:
                    return self._json(
                        {"error": f"that does not look like a {name.replace('_', ' ')}"}, 400
                    )
                values[name] = text
            with LOCK:
                status = self.app.apply_settings(
                    values["api_key"],
                    values["model"],
                    embeddings_api_key=values["embeddings_api_key"],
                    pinecone_api_key=values["pinecone_api_key"],
                )
            return self._json({"ok": True, "status": status, "env": str(ENV_PATH.name)})
        if path == "/api/reset":
            with LOCK:
                # state *and* history: clearing one left the parser able to read a
                # conversation the user had just erased
                self.app.dialogue_for(self._session()).reset()
            return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)

    def _transcribe(self) -> None:
        if self.headers.get_content_type() != "audio/wav":
            return self._json({"error": "Send the recording as audio/wav."}, 415)
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._json({"error": "A recording length is required."}, 411)
        if length <= 0:
            return self._json({"error": "The recording is empty."}, 400)
        if length > MAX_WAV_BYTES:
            return self._json({"error": "Record up to 30 seconds at a time."}, 413)
        lang = parse_qs(urlparse(self.path).query).get("lang", ["en-US"])[0]
        try:
            self.connection.settimeout(10)
            audio = self.rfile.read(length)
            if len(audio) != length:
                raise SpeechError(400, "The recording is incomplete. Please record again.")
            text = self.app.transcriber.transcribe(audio, lang)
        except TimeoutError:
            return self._json({"error": "The recording upload timed out. Try again."}, 408)
        except SpeechError as exc:
            return self._json({"error": str(exc)}, exc.status)
        return self._json({"text": text, "lang": lang})

    def _serve_static(self, name: str) -> None:
        target = (STATIC_DIR / unquote(name)).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in {"application/javascript", "image/svg+xml"}:
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)


class _Server(ThreadingHTTPServer):
    """A server that will not share its port.

    On Windows, SO_REUSEADDR -- which ThreadingHTTPServer sets for you -- lets a second
    process bind a port that is already listening. Both sockets appear in netstat and the
    OS hands each connection to either one, so a stale copy answers some requests and the
    code that runs is not the code on disk. This bit during development: the fixes below
    were live on disk and a server an hour old kept answering. Turning the option off makes
    the second start fail with "address already in use", which is the answer a developer
    wants. Linux does not need the option to be off (its SO_REUSEADDR cannot steal a live
    listener), so the change is Windows-only and restart-after-crash still works.
    """
    if os.name == "nt":
        allow_reuse_address = False


def serve(port: int | None = None, host: str | None = None,
          open_browser: bool = False) -> int:
    port = port or int(os.environ.get("PORT") or 8000)
    host = host or os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    try:
        app = App()
    except (PlaceIndexError, EmbeddingError, PineconeError) as exc:
        # The place index itself never fails closed any more (it falls back to the
        # lexical scorer), so reaching here means the app could not start at all.
        print(f"startup failed: {exc}", file=sys.stderr)
        return 2
    Handler.app = app
    try:
        httpd = _Server((host, port), Handler)
    except OSError as exc:
        # Saying so beats serving stale code from a copy nobody can see.
        print(f"cannot listen on {host}:{port} ({exc}); is another copy already running?",
              file=sys.stderr)
        return 3
    url = f"http://{host}:{port}/"
    print("=" * 66)
    print("  COMP4431 Group 21 - Campus Navigator (web)")
    print(f"  map    : {app.graph.summary()}")
    print(f"  graph  : {rel(GRAPH_FILE)}")
    print(f"  parser : {app.settings.mode_label}")
    mode = app.index_info["mode"]
    detail = (f" ({app.index_info['index']}, {app.index_info['vectors']} vectors)"
              if mode == "dense" else "")
    print(f"  places : {mode}{detail}")
    print(f"  facts  : {app.knowledge.coverage()}")
    print(f"  open   : {url}")
    print("  ctrl-c to stop")
    print("=" * 66)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the Campus Navigator web interface")
    # A hosting platform injects PORT and expects the app on 0.0.0.0. Locally the default
    # stays 127.0.0.1:8000, so nothing changes for the demo and a container still works.
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT") or 8000))
    parser.add_argument("--host", default=os.environ.get("HOST") or
                        ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"))
    parser.add_argument("--open", action="store_true", help="open the browser once it is up")
    args = parser.parse_args(argv)
    return serve(args.port, args.host, args.open)


if __name__ == "__main__":
    raise SystemExit(main())
