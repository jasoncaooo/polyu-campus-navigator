"""The map assistant: the model chooses the steps, the engine does the arithmetic.

The grammar pipeline (`nlu.py` + `dialogue.py`) reads a sentence into slots and then decides
what to do with them. This module replaces that division of labour *when a model is
configured*: the model chooses when to search, when to ask, and when to plan a path. What it
never does is compute. Every id is validated against the graph and every number in the reply
came out of a tool result.

The contract, in the order the model is told to use it::

    search(name)             -> real place ids, best match first (names or descriptions)
    ask(question)            -> one question, and the turn ends
    path(src, dest, via)     -> a measured walk between two real ids
    facilities(item, near)   -> what the map publishes, nearest first
    hours(place, when)       -> quoted opening hours
    close(place)             -> close a place or link, then re-plan
    reset()                  -> forget the journey

The tools are implemented by calling the same engine functions the grammar path calls, so a
route planned by the model is the same route planned without it -- including alternatives,
stops, closures, stair avoidance and the indoor floor note.

Failure is not fatal: `handle()` returns None whenever the model is unreachable, and
`Dialogue` falls back to the grammar for that turn. The offline demo therefore still works,
and a dead API costs coverage rather than correctness.
"""
from __future__ import annotations

import json
import re

from . import lexical
from .embeddings import EmbeddingError
from .format import walk_minutes
from .nlu import Slots
from .pinecone import PineconeError
from .place_index import PlaceIndexError
from .places import amenity_in_text
from .replies import Reply

# One turn may search, ask or plan a few times, but never unboundedly: a model that loops
# on search() must not turn one message into an invoice.
MAX_STEPS = 8

_LANGUAGES = {
    "en-US": "English",
    "zh-HK": "Cantonese (Traditional Chinese)",
    "zh-CN": "Simplified Chinese",
}

SYSTEM_PROMPT = """You are the map assistant for the PolyU campus. You understand the user's message and return a walking answer.

Tools:
- search(name): find campus places by name or description. Returns the real ids.
- ask(question): ask ONE short question when you cannot tell which place the user means.
- path(src, dest, via): plan the walk between two ids returned by search().
- facilities(item, near): published amenities (toilets, AEDs, water, canteens, clinics), nearest first.
- hours(place, when): the published opening hours of a place.
- close(place): close a place or the link between two places, then re-plan.
- reset(): forget the current journey.

Rules:
1. Never invent a place, an id, a distance or an opening time. Call search() before path(), hours() or close(); only ids returned by search() may be passed to path(). Every number you state must come from a tool result.
2. If search() returns several plausible places and the message does not pick one, call ask() with one question that names them. If the user is answering an earlier question, the conversation above says which candidates were offered: a distinguishing word ("forest", "staff") means the candidate whose name contains it. Never guess between candidates.
3. When both endpoints are known, call path() and then answer with what it returned. The app draws the route and the details; you write one or two short sentences.
4. The CURRENT JOURNEY state below is authoritative for "from here", "the nearest one" and "the way back". Do not ask for a place it already gives you.
5. Answer in {language}. Never mention tools, tool results, ids or JSON to the user."""


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOLS = [
    _tool(
        "search",
        "Find campus places whose published name or description matches the phrase. "
        "Also understands service words ('food', 'toilet', 'AED', 'water', 'clinic'): "
        "'va food place' returns the catering outlets in and around Block VA. "
        "Returns real ids with name, building and floor. Use it before any other place tool. "
        "For 'the nearest toilet/AED/water from here', use facilities() instead.",
        {"name": {"type": "string", "description": "The user's words for the place."}},
        ["name"],
    ),
    _tool(
        "ask",
        "Ask the user one short question when the place they mean cannot be decided. "
        "The turn ends with the question; the next message is their answer.",
        {"question": {"type": "string", "description": "One question, in the user's language."}},
        ["question"],
    ),
    _tool(
        "path",
        "Plan the walking route between two real place ids returned by search(). "
        "Returns distance, walking minutes and the route summary.",
        {
            "src": {"type": "string", "description": "Origin place id from search()."},
            "dest": {"type": "string", "description": "Destination place id from search()."},
            "via": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Place ids to walk through, in order.",
            },
            "avoid_stairs": {
                "type": "boolean",
                "description": "True to exclude mapped flights of steps.",
            },
        },
        ["src", "dest"],
    ),
    _tool(
        "facilities",
        "Find published amenities on the map, nearest first, with distance and today's hours.",
        {
            "item": {
                "type": "string",
                "description": "What the user wants ('toilet', 'AED', 'water', 'canteen', "
                               "'clinic'), or empty for the catalogue of everything the map "
                               "publishes.",
            },
            "near": {
                "type": "string",
                "description": "Optional place name or id to measure from.",
            },
        },
        [],
    ),
    _tool(
        "hours",
        "Quote the published opening hours of one place. Never invent a timetable.",
        {
            "place": {"type": "string", "description": "Place name or id from search()."},
            "when": {"type": "string", "description": "Optional day, e.g. 'Tuesday'."},
        },
        ["place"],
    ),
    _tool(
        "close",
        "Report a place or the link between two places as closed, then re-plan the current walk.",
        {
            "place": {
                "type": "string",
                "description": "A place name, two place names, 're-open it', or an edge id.",
            }
        },
        ["place"],
    ),
    _tool("reset", "Forget the current journey and start over.", {}, []),
]


class MapAgent:
    """One conversation with the model, holding a reference to the engine that computes."""

    def __init__(self, dialogue) -> None:
        # Typed loosely on purpose: dialogue imports this module, so naming its type here
        # would be a cycle for the sake of a hint.
        self.dialogue = dialogue
        self._reply: Reply | None = None
        self._question = ""
        self._intent = "out_of_scope"

    # -- public ------------------------------------------------------------
    def handle(self, text: str, language: str | None = None) -> Reply | None:
        """Answer one message, or return None so the grammar path answers instead."""
        dialogue = self.dialogue
        client = dialogue.client
        if client is None or not getattr(client.settings, "llm_available", False):
            return None

        self._reply, self._question, self._intent = None, "", "out_of_scope"
        language = dialogue._language(text, language)
        dialogue.display_language = language

        messages = [{
            "role": "system",
            "content": SYSTEM_PROMPT.format(language=_LANGUAGES.get(language, "English")),
        }]
        digest = dialogue._context_digest()
        if digest:
            messages.append(digest)
        messages.extend(dialogue.history)
        messages.append({"role": "user", "content": text})

        # The tool replies are the facts; the model's last sentence is the prose. Narration
        # inside the tools would ask the same model a second question about its own answer.
        narrate = dialogue.narrate
        dialogue.narrate = False
        final = ""
        try:
            for _step in range(MAX_STEPS):
                result = client.chat(messages, tools=TOOLS, max_tokens=700, retries=2)
                if not getattr(result, "ok", False):
                    return None          # let the grammar answer this turn
                calls = list(getattr(result, "tool_calls", None) or [])
                if not calls:
                    final = (result.text or "").strip()
                    break
                messages.append({
                    "role": "assistant",
                    "content": result.text or "",
                    "tool_calls": calls,
                })
                for call in calls:
                    payload = self._invoke(call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or ""),
                        "content": json.dumps(payload, ensure_ascii=False),
                    })
                    if self._question:
                        break
                if self._question:
                    break
        finally:
            dialogue.narrate = narrate

        return self._finish(final)

    # -- turn bookkeeping --------------------------------------------------
    def _finish(self, final: str) -> Reply | None:
        slots = self._slots()
        if self._question:
            return Reply(self._question, slots=slots, spoken=self._question)
        base = self._reply
        if base is None:
            if not final:
                return None
            reply = Reply(final, slots=slots, spoken=final)
            reply.narrated = True
            return reply
        reply = Reply(
            base.text,
            route=base.route,
            slots=slots,
            notes=list(base.notes),
            narrated=bool(final),
            facilities=list(base.facilities),
            routes=list(base.routes),
            spoken=final,
            lead_prefix=base.lead_prefix,
            facility_result=base.facility_result,
        )
        if final:
            reply.notes.append("the model chose the steps; the engine supplied every number")
        return reply

    def _slots(self) -> Slots:
        return Slots(intent=self._intent, source="agent")

    def _keep(self, reply: Reply, intent: str) -> None:
        """Remember the structured reply a tool produced; the last one is what gets drawn."""
        self._reply, self._intent = reply, intent

    # -- dispatch ----------------------------------------------------------
    def _invoke(self, call: dict) -> dict:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        raw = function.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"ok": False, "error": "arguments must be a JSON object"}
        handler = getattr(self, "_tool_" + name, None)
        if handler is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        try:
            return handler(args)
        except (EmbeddingError, PineconeError, PlaceIndexError):
            # A dead index is the user's problem to see (the web layer answers 502), not a
            # tool error for the model to retry against.
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the model, which can recover
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # -- tools -------------------------------------------------------------
    def _tool_search(self, args: dict) -> dict:
        phrase = str(args.get("name") or "").strip()
        if not phrase:
            return {"ok": False, "error": "name is required"}
        found = list(self.dialogue.places.resolve(phrase))
        if not found:
            # resolve() answers with the best band only; a description the scorer is unsure
            # about still has neighbours worth offering before the model gives up.
            found = [node for _score, node in self.dialogue.places.scored(phrase)
                     if node.kind != "waypoint"]
        # A wish ("va food place") names no place, so name matching alone returns nothing --
        # the amenity vocabulary is the other half of what the user might mean, and it knows
        # "food" is a canteen.
        seen = {node.id for node in found}
        found += [node for node in self._described(phrase) if node.id not in seen]
        matches = [self._place(node) for node in found[:8]]
        payload: dict = {"ok": True, "query": phrase, "matches": matches}
        if not matches:
            payload["suggestions"] = self.dialogue._near_names(phrase)
        return payload

    def _tool_ask(self, args: dict) -> dict:
        question = str(args.get("question") or "").strip()
        if not question:
            return {"ok": False, "error": "question is required"}
        self._question = question
        self._intent = "ask"
        return {"ok": True, "asked": True}

    def _tool_path(self, args: dict) -> dict:
        graph = self.dialogue.graph
        src = str(args.get("src") or "").strip()
        dest = str(args.get("dest") or "").strip()
        via = [str(item).strip() for item in (args.get("via") or [])
               if isinstance(item, str) and item.strip()]
        if not src or not dest:
            return {"ok": False, "error": "src and dest are required"}
        unknown = [value for value in (src, dest, *via)
                   if value not in graph.nodes or graph.node(value).kind == "waypoint"]
        if unknown:
            return {
                "ok": False,
                "error": "unknown place id(s): " + ", ".join(unknown)
                         + ". Call search() and pass ids it returned.",
            }

        slots = Slots(intent="route", origin_text=src, destination_text=dest, vias=via)
        avoid = args.get("avoid_stairs")
        if isinstance(avoid, bool):
            slots.avoid_stairs = avoid
        self.dialogue.state.pending = None
        reply = self.dialogue._route_reply(slots)
        self._keep(reply, "route")

        route = reply.route
        if route is None:
            return {"ok": False, "error": reply.text[:600]}
        payload: dict = {
            "ok": True,
            "reachable": route.reachable,
            "src": graph.node(src).name,
            "dest": graph.node(dest).name,
            "via": [graph.node(value).name for value in via],
        }
        if route.reachable:
            payload["distance_m"] = round(route.total_metres)
            payload["walk_minutes"] = walk_minutes(route.total_metres)
            flights = sum(1 for segment in route.segments if segment.stairs)
            if flights:
                payload["flights_of_stairs"] = flights
            payload["alternative_routes"] = max(len(reply.routes) - 1, 0)
        payload["app_text"] = reply.text[:1200]
        return payload

    def _tool_facilities(self, args: dict) -> dict:
        item = str(args.get("item") or "").strip()
        near = str(args.get("near") or "").strip()
        slots = Slots(intent="facility_query", subject=item, need=item,
                      facility_origin=self._name_of(near))
        if item:
            # The engine decides what the words mean; the model only reports them.
            slots.item = amenity_in_text(item) or ""
        reply = self.dialogue._facilities(slots)
        self._keep(reply, "facility_query")
        if reply.facility_result:
            self.dialogue.state.last_facilities = reply.facility_result
        return {
            "ok": True,
            "count": len(reply.facilities),
            "text": reply.text[:1500],
            "places": [
                {
                    "id": card.get("node_id"),
                    "name": card.get("name"),
                    "building": card.get("building"),
                    "floor": card.get("floor"),
                    "metres": card.get("metres"),
                    "hours_today": card.get("hours_today"),
                    "open_now": card.get("open_now"),
                }
                for card in reply.facilities[:5]
            ],
        }

    def _tool_hours(self, args: dict) -> dict:
        place = self._name_of(str(args.get("place") or "").strip())
        when = str(args.get("when") or "").strip()
        if not place:
            return {"ok": False, "error": "place is required"}
        slots = Slots(intent="hours_query", subject=place, when=when)
        reply = self.dialogue._hours(slots)
        self._keep(reply, "hours_query")
        return {"ok": True, "text": reply.text[:1500]}

    def _tool_close(self, args: dict) -> dict:
        place = self._name_of(str(args.get("place") or "").strip())
        if not place:
            return {"ok": False, "error": "place is required"}
        note = self.dialogue._close(place)
        self._intent = "close"
        state = self.dialogue.state
        if not (state.origin and state.destination):
            return {"ok": True, "note": note}
        slots = Slots(intent="route", origin_text=state.origin,
                      destination_text=state.destination)
        reply = self.dialogue._route_reply(slots, prefix=note)
        self._keep(reply, "route")
        return {"ok": True, "note": note, "text": reply.text[:1200]}

    def _tool_reset(self, args: dict) -> dict:
        self.dialogue.reset()
        self._intent = "reset"
        return {"ok": True, "forgotten": True}

    # -- helpers -----------------------------------------------------------
    _GENERIC = frozenset({"place", "places", "somewhere", "spot", "thing", "things",
                          "something", "anywhere"})

    def _described(self, phrase: str) -> list:
        """Amenity or meaning matches for a phrase that names no place.

        "va food place" is not a name; it is a kind of place plus a part of campus. The
        engine's amenity table turns the kind into a service ("food" -> canteen ->
        "Catering Outlets"), and the words left over ("va") anchor the ranking, so the answer
        is the canteens around Block VA rather than eight arbitrary ones.
        """
        wanted = amenity_in_text(phrase)
        if wanted:
            candidates = self.dialogue._service_matches(
                Slots(intent="facility_query", item=wanted))
            anchors = self._anchors(phrase, wanted)
            if anchors:
                candidates.sort(
                    key=lambda node: min(anchor.distance_to(node) for anchor in anchors))
            if candidates:
                return candidates
        return list(self.dialogue.places.semantic_places(phrase))

    def _anchors(self, phrase: str, wanted: str) -> list:
        """Places named by the words the amenity phrase did not spend: "va" in "va food place"."""
        spent = set(wanted.lower().split())
        words = [
            word for word in re.findall(r"[a-z0-9]+", phrase.lower())
            if len(word) > 1 and word not in spent
            and word not in lexical.STOPWORDS and word not in self._GENERIC
        ]
        found: dict[str, object] = {}
        for word in words[:3]:
            for node in self.dialogue.places.resolve(word):
                if node.kind != "waypoint":
                    found.setdefault(node.id, node)
        return list(found.values())

    def _place(self, node) -> dict:
        return {
            "id": node.id,
            "name": node.name,
            "kind": node.kind,
            "building": node.building or "",
            "floor": node.floor or "",
            "category": node.category or "",
            "service": node.service or "",
        }

    def _name_of(self, value: str) -> str:
        """Accept an id where a name is expected: the model has ids and the engine has names."""
        node = self.dialogue.graph.nodes.get(value)
        return node.name if node is not None else value
