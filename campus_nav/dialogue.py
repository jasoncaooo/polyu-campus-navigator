"""Multi-turn dialogue: remembers places, accumulates constraints, explains failures.

The behaviours below are the ones the marking scheme actually looks at, so they
are implemented explicitly rather than left to the language model:

  * destination known, origin missing      -> ask for the origin, keep destination
  * "the link between C and D is closed"    -> close it, re-route, report the delta
  * "the library is closed"                 -> close the place, re-route, offer options
  * no path exists                         -> name every blocking constraint and offer
                                              concrete ways forward, never a blank refusal
  * off-domain question                    -> say what the bot does cover
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from . import copy
from .config import RAW_DIR, get_settings
from .format import FacilityResult, requested_day, walk_minutes
from . import lexical
from .graph import CampusGraph, Node
from .kb import KnowledgeBase, hours_summary, open_state, today_hours
from . import hours
from .landmarks import landmarks_along
from .llm import DeepSeekClient
from .narrate import Facts, Narration, Narrator
from .nlu import (REOPEN, Slots, asks_for_everything, facility_words, has_cjk,
                  names_destination, parse, parse_with_grammar, refers_to_last_distance,
                  stair_request)
from .place_index import PlaceIndex
from .places import (
    CAPABILITY_LABELS,
    SERVICE_WORDS,
    amenity_in_text,
    facility_capability,
    human_category,
)
from .local_names import service_label
from .replies import Reply
from .search import RouteResult, compare, search


@dataclass
class SessionState:
    origin: str | None = None
    destination: str | None = None
    closed_edges: set[str] = field(default_factory=set)
    closed_nodes: set[str] = field(default_factory=set)
    last_route: RouteResult | None = None
    pending: str | None = None  # "origin" | "destination" | None
    avoid_stairs: bool = False
    last_facilities: FacilityResult | None = None
    # The places this journey goes through, in order. Remembered across turns, so "and I
    # want to pass through Block X" extends the walk instead of replacing it.
    stops: list[str] = field(default_factory=list)


def place_legs(graph: CampusGraph, route: RouteResult) -> list[tuple[str, str, float, bool]]:
    """The route as a person walks it: one leg per pair of PLACES.

    The walkway network needs a node at every bend and junction, and a junction has
    no name -- the graph carries 258 of them. Listing those hops tells a reader
    nothing, so they are folded into the leg that leads to the next real place.

    Exactly one function decides what a "leg" is, because two places ask the same
    question: the printed route, and the facts handed to the narrator. When they
    computed it separately the narrator was told " to : 16 m" once per graph edge
    while the reader was told "1 leg", so the model wrote prose about a walk made
    of twenty-nine unnamed hops.

    Returns (from label, to label, metres, uses stairs). The metres of the legs add
    up to route.total_metres: folding groups the segments, it never drops one.
    """
    def is_junction(node_id: str) -> bool:
        return graph.node(node_id).kind == "waypoint"

    def label(node_id: str) -> str:
        return graph.node(node_id).name or "the walkway"

    legs: list[tuple[str, str, float, bool]] = []
    start = route.path[0] if route.path else ""
    carried = 0.0
    stairs = False
    for seg in route.segments:
        carried += seg.metres
        stairs = stairs or seg.stairs
        if is_junction(seg.v):
            continue
        legs.append((label(start), label(seg.v), carried, stairs))
        start, carried, stairs = seg.v, 0.0, False
    return legs


# How many messages of history the parser sees: two exchanges is enough to resolve "the
# nearest one" or "make it go via X", and short enough that an old turn cannot be mistaken
# for the current question.
HISTORY_TURNS = 6


def fmt_route(graph: CampusGraph, route: RouteResult) -> str:
    """The route as a person reads it: legs between PLACES."""
    if not route.reachable:
        return "No route found."

    legs = place_legs(graph, route)
    stops = [legs[0][0], *(leg[1] for leg in legs)] if legs else [
        graph.node(node_id).name or "the walkway"
        for node_id in route.path
        if graph.node(node_id).kind != "waypoint"
    ]
    lines = [f"Route: {' -> '.join(stops)}"]
    lines.extend(
        f"  - {start} -> {end}: {metres:.0f} m{' (steps)' if stairs else ''}"
        for start, end, metres, stairs in legs
    )

    stats = route.stats
    flights = sum(1 for seg in route.segments if seg.stairs)
    count = max(len(legs), 1)
    extras = (
        f", {flights} flight{'s' if flights != 1 else ''} of steps" if flights else ""
    )
    lines.append(
        f"Total: {route.total_metres:.0f} m over {count} leg{'s' if count != 1 else ''}{extras}"
        f" (about {walk_minutes(route.total_metres)} min at an easy pace)"
    )
    if stats:
        lines.append(
            f"Search ({stats.algorithm}): expanded {stats.nodes_expanded} nodes, "
            f"peak frontier {stats.frontier_peak}, {stats.elapsed_ms:.2f} ms, "
            f"heuristic={stats.heuristic}"
        )
    return "\n".join(lines)


# General accessibility questions beyond the mapped outdoor stair filter.
_STEP_FREE = re.compile(
    r"\b(stairs?|stairway|steps|step[- ]free|wheelchair|barrier[- ]free|"
    r"lift|lifts|elevator|escalator|accessible route|trolley)\b",
    re.I,
)


# A greeting is not a place question. "Hi" reached the resolver, which found the HTI
# department and asked which of three places the user meant -- so this is answered before
# anything is parsed. Only a message that is *nothing but* a greeting matches.
_GREETING = re.compile(
    r"^(?:hi+|hiya|hey+|hello+|yo|howdy|greetings|goods+(?:morning|afternoon|evening)|"
    r"hows+ares+you(?:s+doing)?|nices+tos+meets+you|哈囉|哈罗|你好|您好|早晨|"
    r"嗨|喂)"
    r"[s!,.?。，！？~]*$",
    re.I,
)
# Phrase matching lives in nlu.py, not here: the model reads the words, the grammar reads
# them offline, and this layer reads the resulting slots. What is left in this file that looks
# like language -- the greeting, the thanks, "along the way" -- are *decisions about the
# conversation* (a greeting is not a place question; a wish attached to the walk in progress
# is a stop rather than a new journey), and they are documented where they sit.

# How much walking a stop found by description may add before it is offered instead of
# inserted. Four hundred metres is about five minutes: past that, the user is choosing a
# detour, and choosing is theirs to do.
STOP_DETOUR_M = 400.0

_THANKS = re.compile(
    r"^(?:thanks|thanks+you|thank\s*you|thx|ty|cheers|gots+it|perfect|great|nice|"
    r"唔該|多谢|多謝|谢谢|謝\s*謝)"
    # Cantonese puts a particle after the thanks: 唔該晒, 多謝你
    r"(?:晒|哂|你|您)?[s!,.?。，！？~]*$",
    re.I,
)


class Dialogue:
    def __init__(self, graph: CampusGraph, client: DeepSeekClient | None = None,
                 narrator: Narrator | None = None,
                 knowledge: KnowledgeBase | None = None,
                 places: PlaceIndex | None = None) -> None:
        self.graph = graph
        self.client = client
        self.narrator = narrator
        self.state = SessionState()
        self._places = places
        # The model-first path (agent.py). Built lazily, because the client may be replaced
        # when a key is entered in the settings dialog.
        self._agent = None
        # Whether this turn may use the model for anything besides the agent loop. The agent
        # turns it off when the model is unreachable, so the grammar fallback does not ask
        # the same dead endpoint a second and third time.
        self._llm_ok_for_turn = True
        # The agent writes the prose itself; narration would ask the model again.
        self.narrate = True
        # Recent exchanges, for the parser only: the engine reads its own state, not this.
        self.history: list[dict] = []
        # The language the current question is answered in, set per turn from the request.
        # Not part of the journey state: it belongs to the message being answered.
        self.display_language = "en-US"
        # Built lazily: 876 documents cost about a tenth of a second, which is too
        # much to spend on a /help or /quit, and nothing for a real question.
        self._knowledge = knowledge

    @property
    def places(self) -> PlaceIndex:
        """The place-name index: embeddings in, Pinecone out. No keyword fallback."""
        if self._places is None:
            self._places = PlaceIndex(self.graph, get_settings())
        return self._places

    @property
    def knowledge(self) -> KnowledgeBase | None:
        """The facts the bot may state: opening hours and facility details."""
        if self._knowledge is None:
            try:
                self._knowledge = KnowledgeBase.from_raw(RAW_DIR)
            except (OSError, ValueError, KeyError):
                # A missing or unreadable raw file must never break routing.
                self._knowledge = KnowledgeBase([])
        return self._knowledge

    @property
    def agent(self):
        """The model-first assistant, or None when no model is configured."""
        if self.client is None:
            return None
        if self._agent is None:
            from .agent import MapAgent
            self._agent = MapAgent(self)
        return self._agent

    # -- public ------------------------------------------------------------
    def _interpret_pending_place(self, text: str, slots: Slots) -> None:
        """A bare place name answers the last question about a missing endpoint.

        "Bare" is doing work here: the pending question swallows whatever arrives next, so
        "but I want to make sure I pass through Core P on the way" was read as an answer
        about a place called "want". A sentence that long is not a name -- it goes to the
        parser, where a stop and a re-plan are understood.
        """
        if len(re.findall(r"[\w']+", text or "")) > 6:
            return
        if (self.state.pending and text and slots.intent in ("route", "out_of_scope")
                and (slots.avoid_stairs is None or slots.mentions_places or self._preferred_matches(text))
                and not slots.origin_text and not names_destination(text, self.places)):
            if not text.rstrip().endswith(("?", "？")) or self._preferred_matches(text):
                if self.state.pending == "origin":
                    slots.origin_text, slots.destination_text = text, ""
                else:
                    slots.destination_text = text
                slots.intent = "route"

    def draft_slots(self, text: str) -> Slots:
        """Preview a voice draft without running a search or changing the dialogue."""
        slots = parse_with_grammar(text, self.places)
        if slots.intent == "route" and slots.facility_choice is not None:
            last = self.state.last_facilities
            index = slots.facility_choice
            if last and 0 <= index < len(last.rows):
                destination = last.rows[index].get("node_id")
                if destination in self.graph.nodes:
                    slots.destination_text = destination
                    slots.origin_text = last.route_origin
                    slots.facility_choice = None
            return slots
        self._interpret_pending_place(stair_request(text)[0], slots)
        return slots

    def draft_endpoints(self, text: str) -> dict | None:
        slots = self.draft_slots(text)
        if slots.intent not in ("route", "distance") or slots.facility_choice is not None:
            return None
        inherit = slots.mentions_places or slots.avoid_stairs is not None or refers_to_last_distance(text)
        endpoints = {}
        for role, phrase in (("origin", slots.origin_text), ("destination", slots.destination_text)):
            matches = self._preferred_matches(phrase) if phrase else []
            node_id = matches[0].id if len(matches) == 1 else None
            if not phrase and inherit:
                node_id = getattr(self.state, role)
            endpoints[role] = {"text": phrase, "id": node_id}
        return endpoints

    def handle(self, text: str, *, endpoints: dict | None = None,
               language: str | None = None) -> Reply:
        """Answer one message, and remember the exchange for the next one.

        With a model configured, the model drives the turn through the tools in `agent.py`;
        the grammar answers only when there is no model, when the endpoints came from the map
        (a click is not language), or when the model could not be reached at all. The
        engine's state stays the authority for facts in either case.
        """
        self._llm_ok_for_turn = self.client is not None
        if endpoints is None:
            agent = self.agent
            if agent is not None:
                reply = agent.handle(text, language)
                if reply is not None:
                    self._remember(text, reply)
                    return reply
                self._llm_ok_for_turn = False
        reply = self._handle(text, endpoints=endpoints, language=language)
        self._remember(text, reply)
        return reply

    def reset(self) -> None:
        """Forget the conversation: the engine's state and the parser's history both.

        The history existed as a second kind of memory, and resetting only the first one
        left the parser able to read a conversation the user had just cleared.
        """
        self.state = SessionState()
        self.history.clear()

    def _context_digest(self) -> dict | None:
        """The journey in one line, from the engine's state rather than from the prose.

        Six messages of history resolve "the nearest one"; they cannot carry a journey
        through a long aside. The state can, and it is a fact rather than a transcript, so it
        is handed over as a separate system message: the model may repeat these endpoints,
        and it may never invent one.
        """
        state = self.state
        if not (state.origin or state.destination or state.stops):
            return None
        def label(node_id: str | None) -> str:
            return self.graph.node(node_id).name if node_id in self.graph.nodes else "?"
        parts = [f"origin: {label(state.origin)}" if state.origin else "origin: not known"]
        parts.append(f"destination: {label(state.destination)}" if state.destination
                     else "destination: not known")
        if state.stops:
            parts.append("stops in order: " + ", ".join(label(stop) for stop in state.stops))
        if state.avoid_stairs:
            parts.append("stairs avoided")
        return {"role": "system",
                "content": "CURRENT JOURNEY, from the app's own state (not the user's words "
                           "above): " + "; ".join(parts) + "."}

    def _remember(self, text: str, reply: "Reply") -> None:
        said = (reply.spoken or reply.text or "").strip().replace("\n", " ")
        self.history.append({"role": "user", "content": text.strip()[:400]})
        self.history.append({"role": "assistant", "content": said[:400]})
        del self.history[:-HISTORY_TURNS]

    @staticmethod
    def _language(text: str, language: str | None) -> str:
        """The language to write in: the request's own, else what the words are written in.

        This layer writes a handful of sentences that are not about any place (a greeting,
        the "I can only do campus things" reply). They were English whatever the user
        spoke, because nothing here knew the display language -- the browser does, so it
        says so, and the words themselves are the fallback.
        """
        if language in ("en-US", "zh-HK", "zh-CN"):
            return language
        return "zh-HK" if has_cjk(text) else "en-US"

    def _handle(self, text: str, *, endpoints: dict | None = None,
                language: str | None = None) -> Reply:
        spoken_language = self._language(text, language)
        # The facts handed to the model carry it: the model cannot see the browser, so the
        # engine has to say which language the answer is owed in.
        self.display_language = spoken_language
        if endpoints is None and _GREETING.match(text.strip()):
            warm = ", ".join(self._warm_examples(spoken_language))
            return Reply(copy.t("speech.greeting", spoken_language, examples=warm),
                         notes=["greeting"])
        if endpoints is None and _THANKS.match(text.strip()):
            return Reply(copy.t("speech.thanks", spoken_language), notes=["greeting"])
        if endpoints is None:
            digest = self._context_digest()
            context = ([digest] if digest else []) + self.history
            slots = parse(text, self.graph,
                          self.client if self._llm_ok_for_turn else None,
                          self.places, context)
            self._interpret_pending_place(stair_request(text)[0], slots)
        else:
            # The selections replace only the place slots. Keep the original
            # transcript's distance intent and restrictions (e.g. no stairs).
            slots = self.draft_slots(text)
            if slots.intent not in ("route", "distance") or slots.facility_choice is not None:
                raise ValueError("These endpoints need a route question.")
            if not isinstance(endpoints, dict) or not endpoints:
                raise ValueError("Choose a place on the campus map.")
            for role, node_id in endpoints.items():
                node = self.graph.nodes.get(node_id) if isinstance(node_id, str) else None
                if role not in ("origin", "destination") or not node or node.kind == "waypoint":
                    raise ValueError("Choose a named place on the campus map.")
                setattr(slots, role + "_text", node_id)
        if endpoints is None and slots.clear_stops:
            # "no stops", "just the direct route": drop the places added along the way and
            # plan the walk again without them. The parser read the words; the state change
            # is the dialogue's.
            self.state.stops = []
            if slots.intent == "out_of_scope":
                slots.intent = "route"
        intent = slots.intent

        if intent == "help":
            return self._help(spoken_language)
        if intent == "reset":
            self.reset()
            return Reply(copy.t("speech.reset", spoken_language))

        if slots.facility_choice is not None:
            return self._facility_choice(slots)
        if intent in ("facility_query", "hours_query"):
            # The single place where a user's words become a service lookup. The model used
            # to fill this slot from a list it was shown, and "I want to have a swim" arrived
            # as "water station" -- a wrong answer stated confidently. Now the model reports
            # the words and this line decides, from the map's own vocabulary.
            if not slots.item:
                slots.item = amenity_in_text(slots.need or slots.subject)
            if intent == "facility_query" and slots.along_the_way:
                # "I also want a quick breakfast along the way" is not a question about where
                # catering is; it is an instruction about the walk in progress.
                inserted = self._stop_for_need(slots)
                if inserted is not None:
                    return inserted
            reply = self._hours(slots) if intent == "hours_query" else self._facilities(slots)
            if reply.facility_result:
                self.state.last_facilities = reply.facility_result
            if reply.routes or reply.route is not None:
                return reply
            if self.state.last_route is not None:
                # Answering a question must not rub out the walk on the map. The route is
                # re-attached as a *background* option: the reply's own text is about the
                # question, and with one option there is nothing to choose between.
                reply.routes = [self.state.last_route]
            return reply
        self.state.last_facilities = None

        if slots.avoid_stairs is not None:
            self.state.avoid_stairs = slots.avoid_stairs

        if intent == "close" or slots.closed_text:
            closed_note = self._close(slots.closed_text)
            if not (self.state.origin and self.state.destination):
                return Reply(
                    f"{closed_note}\nTell me your start and destination and I will check "
                    "whether a route still exists.",
                    slots=slots,
                )
            return self._route_reply(slots, prefix=closed_note)

        if intent in ("route", "distance"):
            if intent == "distance" and not slots.mentions_places and not refers_to_last_distance(text):
                # Neither place in the new question was understood. Ask for both
                # again rather than quietly measuring the previous pair.
                self.state.origin = self.state.destination = None
                self.state.last_route = None
            return self._route_reply(slots)

        # Somebody will ask about stairs, lifts or wheelchair access at Open Day. Two
        # thirds of that answer is now known and one third is not, so say exactly
        # which: outdoor flights of steps ARE mapped (OpenStreetMap tags them), lifts
        # and indoor stairs are not, in any source this project can cite.
        if _STEP_FREE.search(text):
            return Reply(
                "I mark flights of steps -- they are drawn as rungs on the map, and a "
                "route that uses one says so. What I do not have is lifts or the stairs "
                "inside a building, so I cannot promise a step-free route.\n"
                "What I can do: find the nearest accessible toilet, water station or AED, "
                "quote opening hours, and re-plan if you tell me a link is closed.",
                slots=slots,
            )

        return Reply(copy.t("speech.out_of_scope", spoken_language), slots=slots)

    # -- pieces ------------------------------------------------------------
    def _warm_examples(self, language: str = "en-US") -> list[str]:
        """Three real amenity words from this map, in the language being spoken."""
        words = facility_words(self.graph)[:3] or ["a nearby amenity"]
        return [service_label(word, language) for word in words]

    def _help(self, language: str = "en-US") -> Reply:
        return Reply(copy.t("speech.help", language),
                     notes=[f"graph: {self.graph.summary()}"])

    def _preferred_matches(self, text: str) -> list[Node]:
        """The places a phrase names, best first, from the embedding index.

        An empty list is a real answer: the phrase names nothing on this map, and the
        dialogue asks a clarifying question instead of guessing.
        """
        return self.places.resolve(text)

    def _close(self, closed_text: str) -> str:
        """Close something the user says is closed: a named link, or a whole place.

        Three ways to be understood, in order of precision:
          1. an edge id            "close E123"
          2. two places            "the link between Core C and Core D is closed"
          3. one place             "the library is closed today", "the bookshop is closed"
        Anything vaguer gets a question instead of a guess -- closing the wrong
        part of the map would silently invalidate every later answer.
        """
        text = (closed_text or "").strip()
        if not text:
            return "I could not tell what is closed, so nothing changed."

        # 0) "re-open it" -- the escape hatch from a dead end
        if REOPEN.search(text):
            had = len(self.state.closed_edges) + len(self.state.closed_nodes)
            self.state.closed_edges.clear()
            self.state.closed_nodes.clear()
            if not had:
                return "Nothing is closed, so there was nothing to re-open."
            return "Re-opened everything that was closed."

        # 1) an explicit edge id, exactly as the map tooltip shows it
        for edge in self.graph.edges:
            if edge.id.lower() in text.lower():
                self.state.closed_edges.add(edge.id)
                return (
                    f"Closed link {edge.id} "
                    f"({self.graph.node(edge.u).name} - {self.graph.node(edge.v).name})."
                )

        # 2) two places -> the link that joins them. Facilities count: closing one is a
        # real request ("the bookshop is closed"), and a facility node sits off the walkway,
        # so blocking it only affects routes that end there. Where the phrase matches
        # several places the reply names them and asks, exactly as routing does.
        places = self._preferred_matches(text)
        if len(places) >= 2:
            # Name them in the order the user wrote them ("the link between Core C and Core
            # D"), not in the resolver's confidence order, which is not a fact about the
            # sentence and reads as a mistake when it disagrees with it.
            lowered_text = text.lower()
            places = sorted(
                places[:2],
                key=lambda node: lowered_text.find(node.name.lower()) % (len(lowered_text) + 1),
            )
            ids = {
                e.id for e in self.graph.edges
                if {e.u, e.v} == {places[0].id, places[1].id}
            }
            if ids:
                self.state.closed_edges |= ids
                return (
                    f"Closed the link between {places[0].name} and {places[1].name} "
                    f"({', '.join(sorted(ids))})."
                )
            # Two named places are usually joined by a *stretch of walkway*, not one drawn
            # link: Core C to Core D is five links and 55 m. Requiring a single link meant
            # "the link between Core C and Core D is closed" -- the most natural way anybody
            # describes a closure here -- closed nothing at all. "The link between them" is
            # that stretch, so the stretch is what closes. Capped, because naming two distant
            # buildings should not silently shut half the campus.
            stretch = search(self.graph, places[0].id, places[1].id,
                             closed_edges=self.state.closed_edges,
                             closed_nodes=self._blocked(),
                             avoid_stairs=self.state.avoid_stairs)
            if stretch.reachable and stretch.segments and len(stretch.segments) <= 8:
                closed = {segment.edge_id for segment in stretch.segments}
                self.state.closed_edges |= closed
                return (
                    f"Closed the {len(closed)} links between {places[0].name} and "
                    f"{places[1].name}: the whole {stretch.total_metres:.0f} m stretch "
                    "between them, which is what a link between two places means here."
                )

        # 3) one place -> close the place itself
        if len(places) == 1:
            self.state.closed_nodes.add(places[0].id)
            return f"Closed {places[0].name}."
        # More than one place, and no link between the first two: the phrase was ambiguous
        # ("the canteen is closed" -- this campus has three). Naming the candidates is more
        # useful than "I could not tell", and closing a guess would invalidate later answers.
        if len(places) > 1:
            names = ", ".join(node.name for node in places[:4])
            return f'"{text}" could be several places: {names}. Which one is closed?'

        return "I could not tell which place or link you mean, so nothing changed."

    def _absorb_places(self, slots: Slots) -> Reply | None:
        """Record any place named this turn. Returns a Reply only if we must ask.

        Both endpoints are absorbed before either is questioned, so a destination
        given in turn 1 survives a clarifying question about the origin.
        """
        unresolved = []
        for role, text in (("origin", slots.origin_text), ("destination", slots.destination_text)):
            if not text:
                continue
            if text in self.graph.nodes and self.graph.node(text).kind != "waypoint":
                # Already an id: a list choice or a confirmed endpoint names a node, not a
                # phrase. Embedding an id string returns unrelated neighbours -- "the nearest"
                # chosen from a canteen list arrived as Core B, because the id went through
                # the resolver as if it were a name.
                setattr(self.state, role, text)
                continue
            matches = self._preferred_matches(text)
            if len(matches) == 1:
                setattr(self.state, role, matches[0].id)
            else:
                setattr(self.state, role, None)
                # The index already ordered them by similarity, so the short spoken
                # list is simply the best few.
                names = tuple(dict.fromkeys(node.name for node in matches))
                unresolved.append((role, text, names[:4]))
        if unresolved:
            self.state.last_route = None
            return self._ask_place(*unresolved[0])
        return None

    def _ask_place(self, role: str, query: str = "", candidates: tuple[str, ...] = ()) -> Reply:
        self.state.pending = role
        known_id = self.state.destination if role == "origin" else self.state.origin
        known = self.graph.node(known_id).name if known_id else ""
        if candidates:
            prompt = f'"{query}" could be several places: {", ".join(candidates)}. Which one?'
        elif query:
            prompt = f'I do not know "{query}". Which {role} do you mean?'
            near = self._near_names(query)
            if near:
                prompt += " Did you mean " + ", ".join(near) + "?"
        elif role == "origin":
            prompt = "Where are you starting from?"
        else:
            prompt = "Where would you like to go?"
        if known:
            other_role = "Destination" if role == "origin" else "Starting point"
            prompt += f" {other_role}: {known}."
        return Reply(prompt)

    def _near_names(self, text: str, limit: int = 3) -> list[str]:
        """The closest published names to a phrase the resolver would not accept.

        A zero match is not the moment to stop talking: "block b" should come back with
        "Block B", not with a lecture about what the map covers.
        """
        if not (text or "").strip():
            return []
        # A structural code that names nothing is a letter, not a meaning: offering the
        # same letter under another head is what a person would say ("there is no Core Y,
        # there is a Block Y"), and it is the one suggestion the scorer cannot make.
        siblings = self.places.coded_siblings(text)
        if siblings:
            return [node.name for node in siblings][:limit]
        try:
            hits = self.places.scored(text)
        except Exception:      # a dead index must not turn a refusal into a 500
            return []
        names: list[str] = []
        for _score, node in hits:
            if node.name and node.name not in names:
                names.append(node.name)
            if len(names) >= limit:
                break
        if names:
            return names
        # Nothing scored, which is what the built-in scorer does for a word it has never
        # seen: it has no notion of a near miss, and a near miss is the whole point here.
        # Spelling is enough -- "libarry" is one transposition from a place on this map --
        # and comparing a handful of words against the published vocabulary costs nothing.
        vocabulary: dict[str, str] = {}
        for node in self.graph.nodes.values():
            for word in lexical.normalise(node.name or ""):
                vocabulary.setdefault(word, node.name)
        for said in lexical.normalise(text):
            if len(said) < 4:
                continue
            for match in difflib.get_close_matches(said, vocabulary, n=1, cutoff=0.75):
                name = vocabulary[match]
                if name and name not in names:
                    names.append(name)
            if len(names) >= limit:
                break
        return names

    def _indoor_text(self, node_id: str) -> str:
        """What to say when the destination is inside a building.

        The route is drawn to the building because that is the last piece of geometry
        anyone published. The floor is the part that actually gets you to the thing.
        """
        node = self.graph.node(node_id)
        building = node.building or "the building"
        floor = f" on {node.floor}" if node.floor else ""
        return (
            f"{node.name} is inside {building}{floor}. The walk above ends at the "
            f"building -- I have no indoor route, so the last part is the entrance, "
            f"the lift and the floor number."
        )

    def _blocked(self) -> set[str]:
        """Places the user has reported closed."""
        return set(self.state.closed_nodes)

    def _best_route(self, origin: str, destination: str, avoid: set[str] | None = None):
        """The shortest walk to the destination. `avoid` closes link ids for this search
        only, which is how the alternatives are generated.

        This used to consider every door of the destination as a candidate goal, and fall
        back to the building anchor when all of them failed. That machinery could never
        run: the graph has no `entrance` nodes, and the parent link it grouped them by was
        discarded at build time. It is gone; a door-aware variant would be a data change
        (authoring entrances), not a routing one.
        """
        return search(
            self.graph, origin, destination,
            algorithm="astar",
            closed_edges=set(self.state.closed_edges) | set(avoid or ()),
            closed_nodes=self._blocked(),
            avoid_stairs=self.state.avoid_stairs,
        )

    def _route_options(self, origin: str, destination: str, limit: int = 3) -> list[RouteResult]:
        """The cheapest route, plus genuinely different ways of getting there.

        Each alternative is a real A* result with one leg of the best route closed for
        that search: no path is smoothed, spliced or invented, and a candidate that
        comes back identical is dropped because that leg was never the only way
        through. The longest leg goes first -- removing it changes the most.
        """
        primary = self._best_route(origin, destination)
        if not primary.reachable:
            return [primary]
        options = [primary]
        seen = {tuple(primary.path)}
        for segment in sorted(primary.segments, key=lambda s: s.metres, reverse=True):
            candidate = self._best_route(origin, destination, {segment.edge_id})
            if not candidate.reachable:
                continue
            key = tuple(candidate.path)
            if key in seen:
                continue
            seen.add(key)
            options.append(candidate)
            if len(options) >= limit:
                break
        return options

    def _via_stops(self, slots: Slots, origin: str, goal: str) -> tuple[list[str], str]:
        """The places the user asked to walk through, in order, plus a note for any lost.

        A stop that names nothing, names several places, or duplicates an endpoint is not an
        error worth failing on: the note explains it and the rest of the walk still runs.
        Skipping an ambiguous one is deliberate -- going through whichever of three canteens
        the score happened to prefer would be a guess dressed as an answer.
        """
        notes: list[str] = []
        stops: list[str] = []
        for phrase in (getattr(slots, "vias", []) or []):
            text = (phrase or "").strip()
            if not text:
                continue
            # A node id is a place, not something to resolve: the model's `path` tool passes
            # the ids `search` returned, and re-embedding an id lands on unrelated neighbours.
            if text in self.graph.nodes and self.graph.node(text).kind != "waypoint":
                hits = [self.graph.node(text)]
            else:
                hits = self._preferred_matches(text)
            if not hits:
                notes.append(f'There is no place called "{text}" here, so the route skips it.')
                continue
            if len(hits) > 1:
                names = ", ".join(node.name for node in hits[:3])
                notes.append(f'"{text}" could be {names}; name one and I will put it on the way.')
                continue
            stop = hits[0].id
            if stop in (origin, goal) or stop in stops:
                continue
            stops.append(stop)
        return stops, " ".join(notes)

    def _stop_for_need(self, slots: Slots) -> Reply | None:
        """Insert a stop that was *described* rather than named: "a quick breakfast on the way".

        The engine makes both decisions the model cannot: which kind of place the words mean
        (amenity wording first, meaning second), and which particular one -- by measuring the
        detour each candidate would add to the walk already planned, and taking the smallest.

        The reply states the detour it accepted. A stop inserted without a visible cost is a
        guess wearing a number, and past STOP_DETOUR_M the choice belongs to the user: the
        reply names the place and asks instead of bending the walk.
        """
        if not (self.state.origin and self.state.destination):
            return None
        need = (slots.need or slots.item or slots.subject or "").strip()
        if not need:
            return None
        wanted = amenity_in_text(need)
        candidates = self._service_matches(Slots(item=wanted)) if wanted else []
        if not candidates:
            # the same two machines as everywhere else: a published amenity first, then what
            # the words mean
            candidates = self.places.semantic_places(need)
        candidates = [node for node in candidates
                      if node.id not in (self.state.origin, self.state.destination)
                      and node.id not in self.state.stops]
        if not candidates:
            return None

        waypoints = [self.state.origin, *self.state.stops, self.state.destination]
        legs = [self._best_route(a, b) for a, b in zip(waypoints, waypoints[1:])]
        if any(not leg.reachable for leg in legs):
            return None      # the walk itself is broken; the route reply explains that better

        def off_route(node: Node) -> float:
            """How far a candidate sits from the walk already planned -- screening only."""
            return min(
                (self.graph.node(point).distance_to(node)
                 for leg in legs for point in leg.path),
                default=0.0,
            )

        best: tuple[float, int, Node] | None = None
        for node in sorted(candidates, key=off_route)[:4]:
            for index, leg in enumerate(legs):
                there = self._best_route(waypoints[index], node.id)
                onward = self._best_route(node.id, waypoints[index + 1])
                if not (there.reachable and onward.reachable):
                    continue
                detour = there.total_metres + onward.total_metres - leg.total_metres
                if best is None or detour < best[0]:
                    best = (detour, index, node)
        if best is None:
            return None

        detour, index, node = best
        total = sum(leg.total_metres for leg in legs)
        label = wanted or need
        if detour > STOP_DETOUR_M:
            reply = Reply(
                f'The closest {label} on this walk is {node.name}, and going through it would '
                f'add {detour:.0f} m to a {total:.0f} m route. Say "add {node.name}" and I '
                "will walk you through it.",
                slots=slots,
                notes=[f'stop detour {detour:.0f} m exceeds {STOP_DETOUR_M:.0f} m'],
            )
            reply.routes = [self.state.last_route] if self.state.last_route else []
            return reply
        self.state.stops.insert(index, node.id)
        # "on the way" is the truthful wording when the detour is under five metres, which
        # happens whenever the place stands on the walkway itself; "+0 m" reads like a bug.
        cost = "on the way" if detour < 5 else f"adding {detour:.0f} m"
        return self._route_reply(
            slots, prefix=f"Added a stop at {node.name} for {label}, {cost}.",
        )

    def _remember_stops(self, slots: Slots, said: list[str], origin: str,
                        goal: str) -> list[str]:
        """The stops of this journey: the ones just said, plus the ones already remembered.

        Three rules, and the whole lifecycle is these three:

          * naming an origin or a destination starts a *new* journey, whose stops are the
            ones said with it (usually none);
          * a turn that only adds a stop extends the journey in progress -- "along the way,
            I want to pass through Block X";
          * "no stops" / "the direct route" clears them, in handle().

        Without the first rule a remembered stop would silently follow you to the next
        question; without the second, every stop would last exactly one reply.
        """
        known = ([] if (slots.mentions_places or getattr(slots, "clear_stops", False))
                 else list(self.state.stops))
        for stop in said:
            if stop not in known:
                known.append(stop)
        known = [stop for stop in known if stop not in (origin, goal)]
        self.state.stops = known
        return known

    def _join(self, legs: list[RouteResult]) -> RouteResult:
        """Several searches stitched into one walk: the stop appears once, in order.

        Everything downstream -- the leg count, the metre total, the drawn line, the spoken
        summary, the landmarks -- reads the joined result, so nothing has to know that it
        was more than one search.
        """
        path: list[str] = []
        segments = []
        for leg in legs:
            path.extend(leg.path if not path else leg.path[1:])
            segments.extend(leg.segments)
        return RouteResult(
            reachable=True,
            algorithm="astar",
            path=path,
            total_metres=sum(leg.total_metres for leg in legs),
            segments=segments,
            stats=None,          # several searches: a single node count would be a fiction
            # and neither is one search's expansion list: the trace is a claim about how the
            # answer was found, and this answer was found more than once. Empty, so the UI
            # shows the legs and the totals without pretending there was one A* run.
            explain=[],
            blocking={key: value for leg in legs for key, value in leg.blocking.items()},
        )

    def _routed_via(self, origin: str, stops: list[str], goal: str,
                    limit: int = 3) -> list[RouteResult]:
        """A walk through every stop in order, with the alternatives that exist.

        Each leg is searched on its own -- origin to the first stop, then stop to stop, then
        the last stop to the goal. That is the only honest way to force a visit: closing the
        stop node would forbid passing through it, and closing an edge would forbid one way
        in.

        Alternatives come from swapping ONE leg for its next-best route, not from the cross
        product of every leg: three ways per leg over four legs is eighty-one journeys, and
        "the third-best way out of the second stop" is not an option anybody asked for. One
        swap at a time gives the genuinely different shapes -- a different way to reach the
        first stop, a different way out of the last -- and keeps the list short enough to
        choose from.
        """
        waypoints = [origin, *stops, goal]
        legs = [self._route_options(a, b, limit=limit)
                for a, b in zip(waypoints, waypoints[1:])]
        for options in legs:
            if not options or not options[0].reachable:
                # the leg that fails is the answer: it says which part of the walk is blocked
                return [options[0]] if options else []
        best = [options[0] for options in legs]
        primary = self._join(best)
        routes = [primary]
        seen = {tuple(primary.path)}
        for index, options in enumerate(legs):
            for alternative in options[1:]:
                combination = list(best)
                combination[index] = alternative
                route = self._join(combination)
                key = tuple(route.path)
                if key in seen:
                    continue
                seen.add(key)
                routes.append(route)
        routes.sort(key=lambda route: route.total_metres)
        return routes[:limit]

    def _describe_alternatives(self, routes: list[RouteResult]) -> str:
        """One line per alternative, in the text, for the terminal and for search."""
        lines = []
        for index, route in enumerate(routes[1:], start=2):
            places = [n for n in route.path if self.graph.node(n).kind != "waypoint"]
            legs = max(len(places) - 1, 1)
            flights = sum(1 for seg in route.segments if seg.stairs)
            bits = f"{route.total_metres:.0f} m over {legs} leg{'s' if legs != 1 else ''}"
            if flights:
                bits += f", {flights} flight{'s' if flights != 1 else ''} of steps"
            lines.append(f"  {index}. {bits}")
        count = len(lines)
        head = f"There {'is' if count == 1 else 'are'} {count} other way{'s' if count != 1 else ''} there:"
        return head + "\n" + "\n".join(lines)

    def _route_reply(self, slots: Slots, prefix: str = "") -> Reply:
        ask = self._absorb_places(slots)
        if not ask and not self.state.destination:
            ask = self._ask_place("destination")
        if not ask and not self.state.origin:
            ask = self._ask_place("origin")
        if ask:
            if prefix:
                ask.text = prefix + "\n" + ask.text
            ask.slots = slots
            return ask

        self.state.pending = None
        origin, destination = self.state.origin, self.state.destination
        if origin and origin == destination:
            # "from VA to Core H" collapsing onto one node is not a 0 m journey, it is a
            # failed reading of the sentence. Two different situations live here, and they
            # need different answers: if the sentence named no endpoint at all, the user is
            # asking about the place they are standing at ("where can I swim?" while at the
            # pool), which is a question, not a broken route.
            self.state.destination = None
            where = self.graph.node(origin).name
            if not slots.mentions_places and not slots.vias:
                self.state.pending = None
                return Reply(
                    (prefix + "\n" if prefix else "")
                    + f"You are at {where}. Where would you like to go?",
                    slots=slots,
                )
            self.state.pending = "destination"
            return Reply(
                (prefix + "\n" if prefix else "")
                + f"Both ends came out as {where}. Where would you like to go instead?",
                slots=slots,
            )

        # A place PolyU publishes but no walkway reaches cannot be walked to. Say exactly
        # that instead of "I cannot find a route", which reads like a bug in the search.
        for node_id, role in ((destination, "destination"), (origin, "start")):
            endpoint = self.graph.node(node_id)
            if getattr(endpoint, "off_network", False):
                return Reply(
                    (prefix + "\n" if prefix else "")
                    + f"{endpoint.name} is published by PolyU, but nothing joins it to "
                    "the mapped walkway network (the nearest path point is "
                    f"{endpoint.nearest_m:.0f} m away), so no walking route can reach it. "
                    "It is on the map, and I can point at it, but I cannot make it your "
                    f"{role}.",
                    slots=slots,
                )

        previous_total = self.state.last_route.total_metres if self.state.last_route else None

        # A facility inside a building is reached by walking to the building: there is no
        # indoor geometry, so the walk ends there and the reply says which floor to look
        # on. That note used to depend on a parent link the build discards, so it never
        # appeared; the published building and floor fields were there all along. Note only:
        # the goal stays the facility's own anchor, which is the building's coordinate.
        goal = destination
        place = self.graph.node(destination)
        indoor_note = ""
        if place.kind == "facility" and place.building and (place.building, place.floor) != (
            place.name, ""
        ):
            indoor_note = self._indoor_text(destination)
        # A stop on the way is two searches stitched into one walk: into the stop, then
        # out of it. Everything downstream (leg count, metre total, the drawn line, the
        # spoken summary) reads the joined result, so nothing has to know it was two.
        stops, via_note = self._via_stops(slots, origin, goal)
        stops = self._remember_stops(slots, stops, origin, goal)
        if stops:
            routes = self._routed_via(origin, stops, goal)
        else:
            routes = self._route_options(origin, goal)
        route = routes[0]

        notes: list[str] = []
        if via_note:
            notes.append(via_note)
        constraints = []
        if self.state.avoid_stairs:
            constraints.append("avoid mapped stairs")
        if self.state.closed_edges:
            constraints.append(f"closed: {', '.join(sorted(self.state.closed_edges))}")
        if constraints:
            notes.append("Constraints: " + "; ".join(constraints))

        if not route.reachable:
            self.state.last_route = None
            suggestions = self._suggestions(origin, destination)
            body = [
                f"I cannot find a route from {self.graph.node(origin).name} to "
                f"{self.graph.node(destination).name} with the current restrictions.",
                f"Blocking: {self._describe_blocking()}",
                "Options:",
            ]
            body += [f"  - {s}" for s in suggestions]
            body.append("Which would you like?")
            template = "\n".join(body)
            facts = self._facts_from_route(route, origin, destination, constraints, previous_total)
            narration = self._narrate(facts)
            reply = Reply(
                (prefix + "\n" if prefix else "") + template,
                route=route, slots=slots, notes=notes, routes=routes,
                # The model's sentence is what gets spoken and what the bubble leads with;
                # the template stays behind the details toggle. Prepending it to the text
                # showed the same answer twice in two wordings.
                spoken=narration.text if narration.ok else "",
                lead_prefix=prefix,
            )
            self._attach_narration(reply, narration)
            return reply

        text = fmt_route(self.graph, route)
        if len(routes) > 1:
            text += "\n" + self._describe_alternatives(routes)
        # The walkway has no names; the buildings beside it do. Naming a few gives the
        # reader something to look for on the way, and they are published places, not
        # guesses: each one is a node whose edge the measured line passes within 20 m.
        marks = landmarks_along(self.graph, route)
        if marks:
            text += "\nYou will pass: " + ", ".join(mark["name"] for mark in marks[:4]) + "."
        if indoor_note:
            text += "\n" + indoor_note
        if previous_total is not None:
            delta = route.total_metres - previous_total
            if abs(delta) >= 0.5:
                sign = "+" if delta > 0 else ""
                text += (
                    f"\nCompared with the previous route ({previous_total:.0f} m): "
                    f"{sign}{delta:.0f} m."
                )
        # Retain the requested destination for replanning: the way in may change when a
        # staircase is excluded or a passage closes.
        self.state.last_route = route
        comparison = None
        if slots.intent == "distance":
            comparison = compare(
                self.graph,
                origin,
                route.path[-1],
                closed_edges=self.state.closed_edges,
                closed_nodes=self._blocked(),
                avoid_stairs=self.state.avoid_stairs,
            )
            notes.append(
                f"A* expanded {comparison['astar']['expanded']} nodes vs UCS "
                f"{comparison['ucs']['expanded']} for the same {comparison['astar']['metres']:.0f} m route."
            )

        # Level B: the model phrases the answer, the calculation supplies the facts.
        facts = self._facts_from_route(route, origin, destination, constraints, previous_total)
        narration = self._narrate(facts)
        reply = Reply(
            (prefix + "\n" if prefix else "") + text,
            route=route, slots=slots, notes=notes, routes=routes,
            spoken=narration.text if narration.ok else "",
            lead_prefix=prefix,
        )
        self._attach_narration(reply, narration)
        return reply

    # -- narration helpers -------------------------------------------------
    def _facts_from_route(self, route, origin: str, destination: str,
                          constraints: list[str], previous_total: float | None) -> Facts:
        """Collect exactly what the model may talk about.

        The legs come from place_legs(), the same function that prints the route:
        a graph edge between two unnamed junctions is not a leg, and describing one
        as "to : 16 m" gives the model nothing to write a sentence about.
        """
        legs = place_legs(self.graph, route)
        segments = [
            f"{start} to {end}: {metres:.0f} m{' (steps)' if stairs else ''}"
            for start, end, metres, stairs in legs
        ]
        options, option_metres = [], []
        if not route.reachable:
            for suggestion in self._suggestions(origin, destination):
                options.append(suggestion)
                match = re.search(r"\((\d+)\s*m\)", suggestion)
                if match:
                    option_metres.append(float(match.group(1)))
        return Facts(
            kind="route" if route.reachable else "no_route",
            origin=self.graph.node(origin).name,
            destination=self.graph.node(destination).name,
            reachable=route.reachable,
            total_metres=route.total_metres if route.reachable else None,
            walk_minutes=walk_minutes(route.total_metres) if route.reachable else None,
            segment_lines=segments,
            segment_metres=[metres for _, _, metres, _ in legs],
            constraints=constraints,
            previous_metres=previous_total,
            blocking=[self._describe_blocking()] if not route.reachable else [],
            options=options,
            option_metres=option_metres,
            language=self.display_language,
        )

    def _narrate(self, facts: Facts) -> Narration:
        if self.narrator is None or not self.narrate or not self._llm_ok_for_turn:
            return Narration(False, reason="no narrator configured")
        return self.narrator.narrate(facts)

    @staticmethod
    def _attach_narration(reply: "Reply", narration: Narration) -> None:
        reply.narrated = narration.ok
        if narration.ok:
            reply.notes.append(f"Narrated by the model ({narration.reason})")
        elif narration.reason not in {"no LLM configured", "no narrator configured"}:
            # a configured model that failed is worth reporting; an absent one is not
            reply.notes.append(f"Template reply used ({narration.reason})")

    def _describe_blocking(self) -> str:
        parts = []
        if self.state.avoid_stairs:
            parts.append("mapped stairs excluded")
        if self.state.closed_edges:
            parts.append(f"closed passages: {', '.join(sorted(self.state.closed_edges))}")
        if self.state.closed_nodes:
            parts.append(f"closed places: {', '.join(sorted(self.state.closed_nodes))}")
        return "; ".join(parts) or "no usable path in the demo map"

    def _suggestions(self, origin: str, destination: str) -> list[str]:
        """Priced ways out of a dead end. Every figure below is a real search."""
        if self.state.avoid_stairs:
            return ["try a different starting point or destination while avoiding mapped stairs"]
        out = []
        if self.state.closed_edges or self.state.closed_nodes:
            relaxed = search(
                self.graph, origin, destination,
                algorithm="astar", closed_edges=set(), closed_nodes=set(),
            )
            if relaxed.reachable:
                out.append(f"re-open what is closed ({relaxed.total_metres:.0f} m)")
            out.append("keep the closure and choose a different destination")
        if not out:
            out.append("choose a different destination")
        return out

    # -- places, facilities and published facts ---------------------------
    def _service_matches(self, slots: Slots, text: str | None = None) -> list[Node]:
        """Service nodes the question is about: 'aed', 'water station', 'batteries'."""
        raw = (slots.item or "").strip()
        if not raw and not text:
            return []
        words = [raw] if raw else []
        words += [w for w in SERVICE_WORDS if text and w in (text or "").lower()]

        wanted_caps = {facility_capability(w) for w in words if w}

        matches: list[Node] = []
        for node in self.graph.nodes.values():
            if node.kind != "facility":
                continue
            doc = self._facility_doc(node)
            if raw == "accessible toilet" and slots.language == "zh":
                # 'toilet' alone says nothing about wheelchair access.
                accessible = doc and str(doc.fields.get("toilet_for_disabled")).lower() in {"true", "1", "yes", "y"}
                if node.name.lower().startswith("accessible toilet") or accessible:
                    matches.append(node)
            elif wanted_caps & set(node.accepts):
                matches.append(node)
            elif (doc and (slots.language == "zh" or "defibrillator" in wanted_caps)
                  and raw == doc.fields.get("category", "").lower()):
                matches.append(node)
            elif (node.service.lower() in wanted_caps if slots.language == "zh" else
                  any(label in node.service.lower()
                      for cap in wanted_caps
                      for label in CAPABILITY_LABELS.get(cap, (cap,)))):
                # joined on the map's own service wording: the graph says "Catering Outlets",
                # not "canteen", and "Clinics", not "clinic"
                matches.append(node)
        return matches

    def _facility_doc(self, node):
        # Some authored POI node IDs use shortened source IDs. Exact official
        # titles join those records without a fuzzy match to a different facility.
        kb = self.knowledge
        # joined on the published title: that is the only key these two sources share
        return kb.by_title(node.name) if kb else None

    @property
    def _default_reference(self) -> str | None:
        """The map reference distances are measured from: Core P, or the first place if not."""
        for node in self.graph.nodes.values():
            if node.name.casefold() == "core p":
                return node.id
        for node in self.graph.nodes.values():
            if node.kind == "place":
                return node.id
        return None

    @property
    def _distance_reference(self) -> str | None:
        return (self.state.origin or (self.state.last_route.path[0] if self.state.last_route
                and self.state.last_route.path else None) or self._default_reference)

    def _distance_from(self, node: Node, reference: str | None = None) -> float | None:
        """Metres from where the user is standing.

        With no user origin, name the map reference used to measure the distance.
        """
        reference = reference or self._distance_reference
        if not reference or reference not in self.graph.nodes:
            return None
        return self.graph.node(reference).distance_to(node)

    def _card(self, node: Node, reference: str | None = None) -> dict:
        """Structured facts about one place, for the reply and for a map or a card view."""
        doc = self._facility_doc(node)
        hours = doc.hours if doc else {}
        is_open, why = open_state(hours) if hours else (None, "")
        metres = self._distance_from(node, reference)
        return {
            "node_id": node.id,
            "name": node.name,
            "service": node.service,
            "building": node.building,
            "floor": node.floor,
            "lat": node.lat,
            "lng": node.lng,
            "accepts": list(node.accepts),
            "hours_today": today_hours(hours),
            "open_now": is_open,
            "open_reason": why,
            "hours_summary": hours_summary(hours),
            "opening_hours": hours,
            "name_zh": doc.fields.get("name_zh", "") if doc else "",
            "name_zh_cn": doc.fields.get("name_zh_cn", "") if doc else "",
            "address": (doc.fields.get("address") if doc else "") or "",
            "metres": round(metres) if metres is not None else None,
            "source": node.source,
        }

    def _facilities(self, slots: Slots) -> Reply:
        """'where is the nearest AED', 'where is the water station', 'what is there'."""
        raw = (slots.item or "").strip().lower()
        # One matcher decides this: the same table that turns wording into a capability.
        wants_service = bool(amenity_in_text(raw)) or bool(facility_capability(raw) != raw)

        subject = (raw or (slots.subject or "")).strip()
        if not raw and (not subject or asks_for_everything(subject)):
            # An explicit request for the catalogue ("what facilities are there?") gets the
            # catalogue. A description ("somewhere to sit outside and study") reaches the
            # search below instead -- the model leaves "item" empty for both, and listing
            # forty amenities at somebody who asked where to sit is not an answer.
            nodes = [n for n in self.graph.nodes.values() if n.kind == "facility"]
            counts: dict[str, int] = {}
            for node in nodes:
                counts[node.service] = counts.get(node.service, 0) + 1
            lines = ["Places I can guide you to on this map:"]
            for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"  - {name} ({count})")
            lines.append(
                'Ask about a facility, e.g. "where is the nearest AED?", '
                '"where is the nearest AED?", "is the canteen open?".')
            return Reply("\n".join(lines), slots=slots, facility_result=FacilityResult("overview", counts=counts))

        # Two questions, answered by two different machines. If the words name an amenity
        # this map publishes, the service search answers precisely (with hours, counts and
        # distances). If they describe a thing or an activity instead, the places themselves
        # are searched by meaning. Nothing here has to guess which case it is: the amenity
        # wording either is in the user's words or it is not.
        matching: list[Node] = self._service_matches(slots) if raw else []
        by_meaning = False
        named = False
        if not matching and subject:
            # A phrase can *name* a place or *describe* one, and the two want different
            # searches. "Where is the library" names Block L; asked to describe it instead,
            # the facility records came back with the defibrillator mounted inside the
            # library, the café inside it and a bookshop, defibrillator nearest. The
            # resolver already knows which case this is -- it answers only when the words
            # carry a name -- so it is asked first and the meaning search is the fallback.
            matching = self._preferred_matches(subject)
            named = bool(matching)
            if not matching:
                matching = self.places.semantic_places(subject)
                by_meaning = bool(matching)
        if not matching:
            reply = self._dont_know(
                raw,
                why=("no facility on the map accepts that item"
                     if not wants_service else "I have no such facility on the map"))
            reply.slots = slots
            reply.facility_result = FacilityResult("no_match", raw)
            return reply

        # Nobody has said where they are: distances come from the map reference, and the
        # reply says so rather than letting "from Core P" imply the user is at Core P.
        slots.unknown_origin = not self.state.origin and self.state.last_route is None
        reference = self._distance_reference
        if slots.facility_origin:
            origins = self._preferred_matches(slots.facility_origin)
            if len(origins) != 1:
                return Reply(f'Which place do you mean by "{slots.facility_origin}"?', slots=slots,
                             facility_result=FacilityResult("unknown_reference", slots.facility_origin))
            reference = origins[0].id
        if reference:
            matching.sort(key=self.graph.node(reference).distance_to)
        cards = [self._card(n, reference) for n in matching[:5]]
        capability = facility_capability(raw)
        label = human_category(capability) if capability != raw else raw
        if named and len(matching) == 1:
            # The question named this place; it is not a search result to be counted.
            head = f'{cards[0]["name"]}:'
        elif by_meaning:
            head = f'Closest places for "{raw or slots.subject}":'
        else:
            head = f"{len(matching)} place(s) for {label}:"
        lines = [head]
        for card in cards:
            bits = [card["name"]]
            where = ", ".join(b for b in (card["building"], card["floor"]) if b)
            if where:
                bits.append(where)
            if card["metres"] is not None:
                bits.append(f'{card["metres"]} m away')
            openness = ("open now" if card["open_now"] else
                        "closed now" if card["open_now"] is False else "")
            hours = card["hours_today"]
            if hours and openness:
                bits.append(f"{openness} ({hours})")
            lines.append("  - " + " - ".join(bits))
        if len(matching) > 5:
            lines.append(f"  ... and {len(matching) - 5} more.")
        if reference:
            lines.append(f"(Straight-line distances are measured from {self.graph.node(reference).name}.)")
        lines.append("Name one and I will draw the walking route.")
        best = cards[0]
        where = ", ".join(p for p in (best["building"], best["floor"]) if p)
        spoken = f'The nearest one is {best["name"]}' + (f" at {where}" if where else "")
        if reference and best["metres"] is not None:
            origin = self.graph.node(reference)
            if (origin.lat, origin.lng) == (best["lat"], best["lng"]):
                spoken += f", mapped at the same location as {origin.name}"
            elif best["metres"] == 0:
                spoken += f", less than one metre in a straight line from {origin.name}"
            else:
                spoken += f", about {best['metres']} metres in a straight line from {origin.name}"
        spoken += "."
        if len(matching) > 1:
            spoken += f" There are {len(matching)} on the map; the others are on screen."
        info = FacilityResult("list", raw, len(matching), reference or "", rows=cards,
                              route_origin=reference if slots.facility_origin else "")
        reply = Reply("\n".join(lines), slots=slots, facilities=cards, spoken=spoken,
                      facility_result=info)
        # The model phrases the answer, the engine supplies the facts -- the same division as
        # for routes. Facility replies used to be the only answers spoken by a template, which
        # is why they read like a database dump next to the narrated routes.
        narration = self._narrate(Facts(
            kind="places",
            origin="",
            destination="",
            reachable=True,
            need=raw or slots.subject or "",
            match_count=len(matching),
            user_location_known=not slots.unknown_origin,
            reference=self.graph.node(reference).name if reference else "",
            best={
                "name": best["name"],
                "building": best["building"],
                "floor": best["floor"],
                "metres": best["metres"],
                "hours_today": best["hours_today"],
                "open_now": best["open_now"],
            },
            language=self.display_language,
        ))
        self._attach_narration(reply, narration)
        if narration.ok:
            reply.spoken = narration.text
        # The engine's guarantee that the measurement is stated: always for the catalogue's
        # own sentence, and for a narrated one only when the model did not mention the
        # reference place itself (then repeating it would say the same thing twice).
        narrated = bool(getattr(reply, "narrated", False))
        needs_note = (not narrated) or (
            self.graph.node(reference).name not in (reply.spoken or ""))
        if slots.unknown_origin and reference and needs_note:
            # Not in the model's facts on purpose: this is an instruction about the
            # conversation, so it is appended rather than left to the prose. Worded as a
            # statement about the *measurement*, not about the user's location -- the model
            # reading "mapped at the same location as Core P" as "right where you are" made
            # the two halves contradict each other. Said once, and in the language the
            # answer is owed in: this sentence used to exist twice in English, once inside
            # the template and once here, so a Chinese answer ended in English.
            reply.spoken = (reply.spoken + " " + copy.t(
                "facility.reference_note", self.display_language,
                reference=self.graph.node(reference).name)).strip()
        return reply

    def _facility_choice(self, slots):
        last = self.state.last_facilities
        index = slots.facility_choice
        if not last or index >= len(last.rows):
            kind = "ask_category" if (index == 0 and "最近" in slots.subject and not (last and last.rows)) else "choice_missing"
            return Reply("Which facility do you mean?", slots=slots,
                         facility_result=FacilityResult(kind))
        # "the nearest from here": where the user now says they are re-ranks the list. The
        # distances were measured from the map reference (Core P) or from where the previous
        # walk started, and neither is where they are standing now.
        reference = self.state.origin or last.route_origin
        here = self._preferred_matches(slots.origin_text) if slots.origin_text else []
        if len(here) == 1:
            reference = here[0].id
            self.state.origin = reference
            if index == 0:
                def distance(row: dict) -> float:
                    node = self.graph.nodes.get(row.get("node_id") or "")
                    return self.graph.node(reference).distance_to(node) if node else float("inf")

                last.rows.sort(key=distance)
                for row in last.rows:
                    node = self.graph.nodes.get(row.get("node_id") or "")
                    metres = self._distance_from(node, reference) if node else None
                    if metres is not None:
                        row["metres"] = round(metres)
        card = last.rows[index] if index < len(last.rows) else last.rows[0]
        node_id = card.get("node_id")
        same_place = (
            bool(node_id) and bool(reference)
            and node_id in self.graph.nodes and reference in self.graph.nodes
            and (node_id == reference
                 or self.graph.node(reference).distance_to(self.graph.node(node_id)) < 5.0)
        )
        if slots.intent == "route" and same_place:
            # "the nearest from here" when the nearest IS here: a zero-metre route is not an
            # answer, and "you are already there" is.
            where = ", ".join(p for p in (card.get("building"), card.get("floor")) if p)
            return Reply(
                f'{card["name"]} is in this building -- you are already there'
                + (f" ({where})." if where else "."),
                slots=slots, facilities=[card],
                facility_result=FacilityResult("selected", last.subject, rows=[card]),
            )
        if slots.intent == "route" and node_id in self.graph.nodes:
            self.state.last_facilities = None
            slots.origin_text = ""
            slots.destination_text = node_id
            if reference:
                self.state.origin = reference
            if slots.avoid_stairs is not None:
                self.state.avoid_stairs = slots.avoid_stairs
            return self._route_reply(slots)
        if slots.intent == "route":
            return Reply("There is no mapped route to this facility.", slots=slots,
                         facility_result=FacilityResult("no_route", rows=[card]))
        kind = "hours" if slots.intent == "hours_query" or last.kind == "hours_options" else "selected"
        info = FacilityResult(kind, last.subject, reference=last.reference, rows=[card],
                              day=requested_day(slots.when or slots.subject) or last.day, route_origin=last.route_origin)
        self.state.last_facilities = info
        return Reply(card.get("hours_summary") or card["name"], slots=slots, facility_result=info,
                     facilities=[card] if node_id in self.graph.nodes else [])
    def _hours(self, slots: Slots) -> Reply:
        return hours.ask(self, slots)

    def _dont_know(self, subject: str, why: str = "") -> Reply:
        """The honest answer. Names the gap and what the bot does cover."""
        coverage = self.knowledge.coverage() if self.knowledge else {}
        documents = coverage.get("documents", 0)
        with_hours = coverage.get("with_opening_hours", 0)
        lines = [
            f'I do not have enough information to answer that'
            + (f' ("{subject}")' if subject else "") + ".",
        ]
        if why:
            lines.append(f"Reason: {why}.")
        # A refusal should still leave the user somewhere to go: the nearest names the
        # resolver can find for what they said are the most useful thing to offer.
        near = self._near_names(subject)
        if near:
            lines.append("Closest on the map: " + ", ".join(near) + ".")
        # The capabilities (water, toilets, AEDs) are only part of what this map carries;
        # listing just those made the refusal sound narrower than the bot is. The service
        # labels are the actual vocabulary: clinics, canteens, bookstores, banks …
        services = sorted({
            n.service for n in self.graph.nodes.values()
            if n.kind == "facility" and n.service
        })
        if services:
            lines.append(
                "What I do cover: walking routes between campus places, and these facilities "
                f"on the map: {', '.join(services)}."
            )
        if documents:
            lines.append(
                f"My knowledge base holds {documents} records from the PolyU campus map, "
                f"but only {with_hours} of them publish opening hours."
            )
        lines.append("I would rather say so than invent an answer.")
        return Reply("\n".join(lines))
