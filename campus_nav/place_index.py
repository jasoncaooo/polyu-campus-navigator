"""Place-name resolution: which node does a phrase name?

Two resolvers, one contract
--------------------------
    PlaceIndex.resolve(text) -> list[Node]      best first, empty when nothing matches

1. **dense** (primary): embed the phrase, read the nearest place names back from the
   vector index. Semantic, so "somewhere to fill a bottle" finds a Water Station.
   Costs two network round-trips -- one embeddings call, one vector query.
2. **lexical** (fallback): word-boundary scoring over the graph's own names, in
   `campus_nav/lexical.py`. Needs nothing but the graph.

The mode is chosen once, at startup, by check(): if the keys are missing or the index
is empty, the index starts lexical and says so. If a *later* call fails (quota, expired
key, Wi-Fi gone), scored() degrades to lexical for the rest of the session instead of
raising into the dialogue layer. A chatbot that cannot find a place because a vector
database blinked is a worse product than one that finds it slightly differently.

The index is built once by `python tools/build.py embed`; at runtime the graph is still
the source of truth, and a metadata id that is not in the graph is dropped rather than
trusted.
"""
from __future__ import annotations

import re
import sys
from collections import OrderedDict

from . import lexical
from .config import Settings, get_settings
from .embeddings import EmbeddingClient, EmbeddingError
from .graph import CampusGraph, Node
from .pinecone import PineconeError, PineconeStore

CACHE_SIZE = 256

MODE_DENSE = "dense"
MODE_LEXICAL = "lexical"

# The lexical scorer returns coarse confidence tiers (0.40 / 0.44 / 0.70 / 1.00), so a
# cosine-sized band would keep one candidate out of a family that belongs together: at
# 0.04, "canteen" answered with a single canteen. Fifteen points keeps a tier together.
LEXICAL_BAND = 0.15

# Chinese is written without spaces, and its place names are head-final: the category word
# comes last ("包玉剛圖書館") while a qualifier comes first ("圖書館咖啡廳", "(去顫器) 圖書館").
# Cosine similarity does not know that, so "圖書館" came back as the café and the AED
# inside the library and not as the library. A published name that *ends* with the phrase
# is a name match -- the same argument as NAME_EVIDENCE, in a language where a word
# boundary is not available.
_CJK = re.compile(r"[\u3400-\u9fff]")

# How much weaker than the best match a description's neighbour may be and still be the
# same answer to a person. "I want to have a swim" scores 0.53 for the pool and 0.39 for the
# nearest water fountain: both clear the "names something" floor, and only the first is an
# answer. A tenth is wide enough to keep genuinely similar places ("sit outside and study"
# returns a bookshop, a plaza and a study centre within 0.03 of each other) and narrow
# enough to drop a different kind of place.
MEANING_BAND = 0.10
# A published name is often a code plus a bracketed name: "L座(包玉剛圖書館)". Brackets and
# separators end a segment, so the head-final test below looks inside them too.
_SEGMENTS = re.compile(r"[()（）\[\]【】\s·,，、]+")


# The campus names its buildings with a kind and a code: "Core P", "Block VA", "Wing FJ".
# One letter is all that separates any two of them, which is precisely the difference an
# embedding cannot see -- and the difference a regex cannot miss.
_STRUCTURAL = re.compile(r"(core|block|wing|tower)\s+([a-z]{1,3}\d?)", re.I)


def shares_a_name_word(node: Node, words: list[str]) -> bool:
    """Does one of these words appear in a name this place is published under?"""
    if not words:
        return False
    names = [node.name or "", node.name_zh or "", node.name_zh_cn or ""]
    for name in names:
        candidates = lexical.normalise(name)
        for word in words:
            for candidate in candidates:
                if word == candidate:
                    return True
                # plurals and endings: "toilets" is "Toilet (Block V)"
                if len(word) >= 4 and len(candidate) >= 4 and (
                    word.startswith(candidate) or candidate.startswith(word)
                ):
                    return True
            # Chinese is written without spaces, so a name is one run and the test is
            # containment
            if _CJK.search(word) and word in name:
                return True
    return False


def evidenced(hits: list[tuple[float, Node]], text: str) -> list[tuple[float, Node]]:
    """Drop matches that nothing in the phrase accounts for.

    Dense scoring always answers with neighbours, so a single unknown word lands on
    whatever is nearest: "want" scored highest for "Block W (Industrial Centre)" and "Hi"
    for the HTI department. Both became "which place did you mean?" -- a greeting and a
    verb turned into place questions.

    The rule that fixes it is the one the rest of this project already uses: a name has to
    be *in* the query. A single word therefore has to appear in a published name; a longer
    phrase is allowed to match on meaning, because describing a place in your own words is
    the entire point of the embeddings ("somewhere to fill my bottle").
    """
    words = [word for word in lexical.normalise(text) if len(word) >= 2]
    if len(words) >= 2:
        return hits
    return [(score, node) for score, node in hits if shares_a_name_word(node, words)]


def cjk_named(hits: list[tuple[float, Node]], probe: str) -> list[tuple[float, Node]]:
    """The candidates a Chinese phrase actually names, place before facility.

    Cosine similarity cannot see that Chinese names are head-final in a segment: "圖書館"
    scored 0.75 for "圖書館咖啡廳" (the café), 0.70 for "(去顫器) 圖書館" (the AED inside it)
    and 0.69 for "L座(包玉剛圖書館)" — the library itself, third. A segment that *ends* with
    the phrase is a name match, which is the same argument NAME_EVIDENCE makes in English.

    Then, when the phrase names both a building and something standing in it, the building
    is the answer: the follower drop in resolve() applies the same rule, but it cannot fire
    here: these facilities carry no building link, so nothing else can make the call.
    """
    def ends_with(name: str) -> bool:
        return any(part.strip().endswith(probe)
                   for part in _SEGMENTS.split(name or "") if part.strip())

    named = [
        (score, node) for score, node in hits
        if ends_with(node.name_zh) or ends_with(node.name_zh_cn)
    ]
    if not named:
        return hits
    places = [pair for pair in named if pair[1].kind != "facility"]
    return places or named


class PlaceIndexError(RuntimeError):
    """The place index is not configured, empty, or does not match the graph."""


def indexable_nodes(graph: CampusGraph) -> list[Node]:
    """Every node a person could name. Waypoints are bends in a walkway, not places."""
    return [
        node for node in graph.nodes.values()
        if node.kind != "waypoint" and node.name
    ]


def document_text(node: Node) -> str:
    """The string embedded for one place: every name it is published under, plus where.

    Both scripts and the block code are included because a user may type any of them;
    the category and service words give the vector something to match a paraphrase
    against ("somewhere to fill a bottle" -> "Water Station").
    """
    parts = [node.name]
    for name in (node.name_zh, node.name_zh_cn):
        if name and name not in parts:
            parts.append(name)
    if node.id and node.id != node.name:
        parts.append(node.id)
    for value in (node.category, node.service):
        if value:
            parts.append(value)
    if node.building:
        parts.append(f"in {node.building}")
    if node.floor:
        parts.append(f"floor {node.floor}")
    return " | ".join(parts)


class PlaceIndex:
    def __init__(self, graph: CampusGraph, settings: Settings | None = None,
                 *, embedder: EmbeddingClient | None = None,
                 store: PineconeStore | None = None) -> None:
        self.graph = graph
        self.settings = settings or get_settings()
        self.embedder = embedder or EmbeddingClient(self.settings)
        self.store = store or PineconeStore(self.settings)
        self.mode = MODE_LEXICAL          # until check() proves the dense index usable
        self.warning = ""                 # why it is lexical, for the operator
        self._cache: OrderedDict[str, list[tuple[float, Node]]] = OrderedDict()

    # -- readiness ---------------------------------------------------------
    def check(self) -> dict:
        """Probe the dense index once. NEVER fatal: report, then fall back.

        Returns the line the server prints at startup and exposes through /api/status.
        """
        info: dict = {
            "index": "-", "vectors": 0, "dimension": None,
            "mode": MODE_LEXICAL, "warning": "",
        }
        if not self.settings.vectors_available:
            self.warning = (
                "no EMBEDDINGS_API_KEY and PINECONE_API_KEY -- using the built-in "
                "name scorer"
            )
        else:
            try:
                stats = self.store.stats()
                namespaces = stats.get("namespaces") or {}
                total = (
                    sum(int(ns.get("vectorCount") or 0) for ns in namespaces.values())
                    if namespaces else int(stats.get("totalVectorCount") or 0)
                )
                if total > 0:
                    self.mode = MODE_DENSE
                    self.warning = ""
                    info.update(
                        index=self.settings.pinecone_index,
                        vectors=total,
                        dimension=stats.get("dimension"),
                        mode=MODE_DENSE,
                    )
                else:
                    self.warning = (
                        f"vector index {self.settings.pinecone_index!r} is empty -- run "
                        "python tools/build.py embed; using the built-in name scorer"
                    )
            except (PineconeError, EmbeddingError) as exc:
                self.warning = (
                    f"vector store unreachable ({exc}) -- using the built-in name scorer"
                )
        info["mode"] = self.mode
        info["warning"] = self.warning
        return info

    def reconfigure(self, settings: Settings) -> dict:
        """Adopt new keys (from the Settings dialog) and re-probe the index.

        The embedder and the store captured the old settings when they were built, so
        they are rebuilt here. Never fatal: a key that does not work leaves the index in
        lexical mode with the reason stated, exactly as a cold start would.
        """
        self.settings = settings
        self.embedder = EmbeddingClient(settings)
        self.store = PineconeStore(settings)
        self.mode = MODE_LEXICAL
        self.warning = ""
        self._cache.clear()
        return self.check()

    def degrade(self, reason: object) -> None:
        """Give up on the dense index for this session and say why, once."""
        if self.mode == MODE_LEXICAL:
            return
        self.mode = MODE_LEXICAL
        self.warning = f"fell back to the built-in name scorer: {reason}"
        self._cache.clear()
        print(f"  places : {self.warning}", file=sys.stderr)

    # -- retrieval ---------------------------------------------------------
    def scored(self, text: str, k: int | None = None) -> list[tuple[float, Node]]:
        """Candidate nodes for the phrase, best first, scores in 0.0-1.0."""
        phrase = " ".join((text or "").split())
        if not phrase:
            return []
        key = phrase.casefold()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        if self.mode == MODE_DENSE:
            try:
                found = self._dense_scored(phrase, k)
            except (EmbeddingError, PineconeError, PlaceIndexError) as exc:
                self.degrade(exc)
                found = self._lexical_scored(phrase, k)
        else:
            found = self._lexical_scored(phrase, k)

        self._cache[key] = found
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return found

    def _dense_scored(self, phrase: str, k: int | None) -> list[tuple[float, Node]]:
        vector = self.embedder.embed_one(phrase)
        hits = self.store.query(vector, k or self.settings.place_top_k)
        found: list[tuple[float, Node]] = []
        seen: set[str] = set()
        for score, meta in hits:
            node_id = str(meta.get("node_id") or "")
            node = self.graph.nodes.get(node_id)
            if node is None or node.kind == "waypoint" or node.id in seen:
                continue
            if score < self.settings.place_min_similarity:
                continue
            seen.add(node.id)
            found.append((score, node))
        return found

    def _lexical_scored(self, phrase: str, k: int | None) -> list[tuple[float, Node]]:
        found = lexical.resolve_scored(self.graph, phrase)
        if found and k:
            found = found[:k]
        return found

    # A whole-word match on a published name is stronger evidence than embedding
    # proximity: "library" is a word in "Block L (Pao Yue-Kong Library)", while its
    # nearest neighbours by cosine are a bookshop, a café and an AED -- related places,
    # not the one that was named. Above this tier the lexical evidence wins; below it
    # ("somewhere to fill my bottle") the embeddings are the only evidence there is.
    NAME_EVIDENCE = 0.60

    def structural_places(self, text: str) -> list[Node] | None:
        """A building named by the campus's own code convention, or None.

        "Core Y" is not a place on this campus, and the embeddings answered it with Core T,
        Core C, Core J and Core Q -- four confident wrong answers, because every core is one
        letter from every other. Where the convention is deterministic the answer is too:
        the phrase must be *only* the code ("Core P", not "Block X Sports Centre"), and the
        name must be exactly it or begin with it and qualify it in brackets. A code that
        does not exist therefore resolves to nothing, and the reply can say so.
        """
        phrase = " ".join(lexical.normalise(text))
        match = _STRUCTURAL.fullmatch(phrase)
        if not match:
            return None
        wanted = f"{match.group(1).title()} {match.group(2).upper()}"
        exact: list[Node] = []
        qualified: list[Node] = []
        for node in self.graph.nodes.values():
            if node.kind == "waypoint" or not node.name:
                continue
            name = node.name.strip()
            if name.casefold() == wanted.casefold():
                exact.append(node)
            elif name.casefold().startswith(wanted.casefold() + " ("):
                qualified.append(node)
        return exact or qualified

    def semantic_places(self, text: str, floor: float | None = None,
                        limit: int = 5) -> list[Node]:
        """Places that *mean* what the phrase means, strongest first.

        The facility layer knows the services this map publishes -- AEDs, toilets, water,
        catering, clinics. A wish that names none of them ("I want to have a swim",
        "somewhere to sit outside") is not an unknown question, it is a question about a
        *place*, and every place here is embedded. So the same nearest-neighbour search
        that resolves names is asked by meaning, with a floor high enough that a swim
        answers the pool rather than the sports centre next to it.

        The floor is the configured one (place_min_similarity) rather than a number chosen
        here: "somewhere to sit outside and study" scores 0.42 against the plaza, while a
        name scores 0.9, and one threshold for "this names something at all" is the whole
        point of having a setting.
        """
        if not (text or "").strip():
            return []
        if floor is None:
            floor = self.settings.place_min_similarity
        ranked = [(score, node) for score, node in self.scored(text)
                  if score >= floor and node.kind != "waypoint"]
        if not ranked:
            return []
        best = ranked[0][0]
        found: list[Node] = []
        for score, node in ranked:
            if score < best - MEANING_BAND:
                break
            if node not in found:
                found.append(node)
            if len(found) >= limit:
                break
        return found

    def coded_siblings(self, text: str) -> list[Node]:
        """Places that do exist under the code the phrase used: "Core Y" -> Block Y.

        A code that names nothing is usually the right letter on the wrong kind of
        building, so the useful thing to offer is that letter under another head -- not
        "Core T, Core C, Core J", which is what the embeddings propose.
        """
        phrase = " ".join(lexical.normalise(text))
        match = _STRUCTURAL.fullmatch(phrase)
        if not match:
            return []
        code = match.group(2).upper()
        found: list[Node] = []
        for node in self.graph.nodes.values():
            if node.kind == "waypoint" or not node.name:
                continue
            head = _STRUCTURAL.match(" ".join(lexical.normalise(node.name)))
            if head and head.group(2).upper() == code:
                found.append(node)
        return found

    def resolve(self, text: str) -> list[Node]:
        """The place(s) the phrase means, best first; empty when it means nothing."""
        structural = self.structural_places(text)
        if structural is not None:
            return structural
        hits = evidenced(self.scored(text), text)
        if not hits:
            return []

        if self.mode == MODE_DENSE:
            if _CJK.search(text):
                hits = cjk_named(hits, text.strip().strip("?？。！!，, "))
            named = {node.id for score, node in lexical.resolve_scored(self.graph, text)
                     if score >= self.NAME_EVIDENCE}
            if named:
                restricted = [(score, node) for score, node in hits if node.id in named]
                if restricted:
                    hits = restricted

        # A name whose every word appears in the query is the name that was said: "Block X
        # Sports Centre" is a building of its own, and answering it with "Block X" sends
        # somebody to the block next door. Where such a name exists, the band narrows to it
        # -- the most specific of them wins ("Block X Sports Centre" over "Block X").
        # Nothing is narrowed when no candidate is fully covered ("library" is inside
        # "Block L (Pao Yue-Kong Library)" but not the other way round), which is the case
        # NAME_EVIDENCE and the lexical tiers already handle.
        said = set(lexical.normalise(text))
        covering = [
            pair for pair in hits
            if said and set(lexical.normalise(pair[1].name)) <= said
        ]
        if covering:
            longest = max(len(lexical.normalise(node.name)) for _, node in covering)
            hits = [pair for pair in covering
                    if len(lexical.normalise(pair[1].name)) == longest]

        # A facility whose name is "X (Parent)" is a feature *of* Parent, so a phrase that
        # names the parent names Parent. The English names get this right on their own --
        # "library" scores the building 0.70 against the defibrillator's 0.39, because the
        # building's own name carries the word -- but the Chinese names do not: 圖書館 is
        # the whole published name of neither candidate, and LibCafé, AED (Library) and
        # Block L (Pao Yue-Kong Library) come back as three equal offers of "which one?".
        # The map says which is which: a facility whose own head words the phrase does not
        # contain is a feature standing inside the building that was named, not an
        # alternative to it. Facilities asked for by their own words ("AED", "診所", the
        # twenty-five toilets) keep their own answers.
        buildings = [pair for pair in hits if pair[1].kind != "facility"]
        if buildings and len(hits) > len(buildings):
            features = {pair[1].id for pair in hits if pair[1].kind == "facility"
                        and not (set(lexical.normalise(pair[1].name.split("(")[0])) & said)}
            if features:
                hits = [pair for pair in hits if pair[1].id not in features]

        # Only the top band is an answer. A weaker neighbour is a worse guess, not an
        # alternative: "library" must not also offer "Block M". Dense scores are cosines
        # and keep the published floor; lexical scores are tiers, so they get a tier.
        best = hits[0][0]
        if self.mode == MODE_DENSE:
            floor = max(self.settings.place_min_similarity,
                        best - self.settings.place_ambiguity_band)
        else:
            floor = best - LEXICAL_BAND
        top = [(score, node) for score, node in hits if score >= floor]

        # One answer per name. "Pao Yue-Kong Library" (a place) and "Pao Yue-kong Library
        # (LIB)" (a facility published under the same name) are one destination written
        # twice, and asking "which one did you mean?" is a worse answer than picking the
        # place. This is about the published names, so it applies to both resolvers.
        # Facilities that merely share a service name -- the twenty-five toilets -- are
        # distinct places and stay.
        owned = {
            lexical.name_parts(node.name)[0]
            for _, node in top if node.kind != "facility"
        }
        if owned:
            top = [
                pair for pair in top
                if pair[1].kind != "facility"
                or lexical.name_parts(pair[1].name)[0] not in owned
            ] or top

        # In lexical mode, prefer matches on what a place IS over the same words inside a
        # bracket: "library" must not also offer "Block L (Pao Yue-Kong Library)". Dense
        # mode is left alone -- there the words need not appear in any name at all
        # ("somewhere to fill a bottle"), so the same test would drop valid answers.
        if self.mode == MODE_LEXICAL and len(top) > 1:
            words = lexical.content_words(text)
            if words:
                by_head = [
                    pair for pair in top
                    if all(re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])",
                                     lexical.name_parts(pair[1].name)[0])
                           for w in words)
                ]
                if by_head:
                    top = by_head

        return [node for _, node in top]
