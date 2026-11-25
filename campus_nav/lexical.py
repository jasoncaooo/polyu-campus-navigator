"""Name resolution without a model: word-boundary matching over the graph's own names.

Why this file exists
--------------------
The primary resolver embeds the query and asks a vector index for nearest neighbours
(`place_index.PlaceIndex`). That needs two network round-trips, an embeddings key, a
vector-store key and a populated index. When any of those is missing -- no key, no
quota, no Wi-Fi, an empty index -- this module answers instead, so "the network is
down" never means "I cannot find the library".

It is not a fallback written in a hurry: it is the resolver this project shipped
before the vector index existed, kept because it needs nothing but the graph. The
confidence tiers below are the lessons of that period.

Confidence levels, highest first
--------------------------------
  100  the user typed the name exactly
   96  the user typed the whole name, brackets and all
 60-90  the user's phrase is one of the node's names, by whole-word coverage
 40-65  a significant word inside the user's phrase appears in a node name --
        this is what makes "how do I get to the library" find
        "Block L (Pao Yue-Kong Library)" with no alias table
 20-39  the phrase names only WHERE a thing is, not what it is ("block v" for
        "Toilet (Block V)"); always outranked by a match on what the place IS

Guards learned the hard way: a short alias such as "N" must never match inside
another word ("e-N-trance"), so substring matching requires 3+ characters on the
candidate and 4+ on the user's word. A phrase that matches nothing must score zero:
an empty score staying zero is what stops "the moon" from returning 222 nodes.

Scores are returned normalised to 0.0-1.0 so the caller can treat this and the
embedding index on the same scale.
"""
from __future__ import annotations

import re
import unicodedata

from .graph import CampusGraph, Node

_BRACKETED = re.compile(r"[（(]([^()（）]+)[)）]")

STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "am", "was", "were", "be", "to", "of", "at", "in",
    "on", "for", "from", "near", "next", "by", "between", "with", "and", "or", "but",
    "me", "my",
    "i", "you", "it", "its", "please", "can", "could", "would", "how", "do", "does",
    "get", "walk", "go", "going", "where", "which", "what", "when", "today", "now",
    "closed", "open", "there", "here", "this", "that",
})


_BLOCK_WORDS = {"core", "block", "wing", "tower"}


def drop_filler(text: str) -> str:
    """The phrase with filler words removed, keeping the caller's capitalisation.

    One implementation, because two did not agree: the parser's own cleaner kept "closed"
    ("the library is closed" -> "library closed") while this list drops it, and that
    disagreement is what made a closure sentence fail to resolve its own place.
    """
    words = [w for w in re.split(r"[\s,.?!;:]+", (text or "").strip()) if w]
    kept: list[str] = []
    for position, word in enumerate(words):
        bare = word.strip(".,;:!?()[]{}\"'").lower()
        if bare in STOPWORDS and not (position and words[position - 1].lower() in _BLOCK_WORDS):
            continue
        kept.append(word)
    return " ".join(kept or words).strip()


def singular(word: str) -> str:
    """"One word, with an English plural folded to its singular.

    Names are published in the singular ("Toilet (Block V)") and people ask in the plural
    ("where are the toilets"), so both sides of every comparison go through this. The
    guards matter: "physics", "campus", "analysis" and "VS" are not plurals, and
    "libraries" is "library", not "librarie".
    """
    if len(word) < 4 or not word.endswith("s"):
        return word
    if word.endswith(("ss", "us", "is", "as", "os")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    return word[:-1]


def fold(text: str) -> str:
    """Strip diacritics, because nobody types them: "LibCafe" is how the café is asked for.

    Decomposing and dropping the combining marks is the whole of it; Chinese has no
    combining marks, so this is a no-op for the names in the other script.
    """
    return "".join(char for char in unicodedata.normalize("NFKD", text or "")
                   if not unicodedata.combining(char))


def normalise(text: str) -> list[str]:
    """"The words a phrase is about": lowercased, filler dropped, block letters kept.

    "the canteen" and "canteen" must score the same: filler words push every candidate
    into the loose tier (40 + word length), which turns a confident answer into a list of
    near-ties. The exception is the letter in "Core A" / "Block Y", where the letter IS
    the place.
    """
    words = " ".join(fold(text).lower().replace("_", " ").split()).split()
    # Sentence punctuation is not part of a word: "closed today." must still be read as
    # "closed" + "today", or the filler word survives and poisons the phrase.
    words = [word.strip(".,;:!?()[]{}\"'") for word in words]
    words = [singular(word) for word in words if word]
    return [
        word for position, word in enumerate(words)
        if word not in STOPWORDS
        or (position and words[position - 1] in _BLOCK_WORDS)
    ]


def content_words(text: str) -> list[str]:
    """The words long enough to identify a place, from a phrase or a whole sentence."""
    found = re.findall(r"[a-z0-9\u3400-\u9fff]{4,}", " ".join(normalise(text)))
    return list(dict.fromkeys(found))


def name_parts(name: str) -> tuple[str, str]:
    """Split a name into what a place IS and where it IS.

    "Toilet (Block Y)" -> ("toilet", "block y"). The bracket is a location qualifier,
    so a match on it must never outrank a match on the part that says what the thing
    is.
    """
    tail = " ".join(_BRACKETED.findall(name or "")).lower()
    head = " ".join(_BRACKETED.sub(" ", name or "").split()).lower()
    return head, tail


def resolve_scored(graph: CampusGraph, text: str) -> list[tuple[float, Node]]:
    """Every node the phrase could mean, best first, score normalised to 0.0-1.0."""
    if not text:
        return []
    needle = " ".join(normalise(text)) or " ".join(text.lower().split())
    if not needle:
        return []

    def has_words(short: str, long: str) -> bool:
        return re.search(rf"(?<![a-z0-9]){re.escape(short)}(?![a-z0-9])", long) is not None

    needle_words = [w for w in re.findall(r"[a-z0-9\u3400-\u9fff]{4,}", needle)]
    short_tokens = [
        token for token in re.findall(r"[a-z0-9]+", needle)
        if len(token) < 4 and token not in STOPWORDS
    ]

    def score(cand: str) -> int:
        """100 exact, 60-90 whole words by coverage, 40-65 loose single words."""
        if not cand:
            return 0
        if needle == cand:
            return 100
        if len(cand) >= 3 and (has_words(cand, needle) or has_words(needle, cand)):
            return 60 + round(30 * min(len(needle), len(cand)) / max(len(needle), len(cand)))
        if any(not has_words(token, cand) for token in short_tokens):
            return 0
        return max((40 + min(len(word), 25) for word in needle_words
                    if has_words(word, cand)), default=0)

    scored: list[tuple[int, Node]] = []
    for node in graph.nodes.values():
        # A waypoint is a bend in a walkway, not a place: no name worth saying, and
        # never a destination.
        if node.kind == "waypoint" or not node.name:
            continue
        head, tail = name_parts(node.name)
        best = max(score(node.id.lower()), score(head))
        # The bracket after a block code names the building itself. In a facility
        # label it only gives an address, e.g. Toilet (Block V).
        if tail and node.kind == "place" and re.fullmatch(r"(?:block|wing|core) [a-z]+", head):
            best = max(best, score(tail))
        for alias in (node.name_zh, node.name_zh_cn):
            if not alias or alias.strip().casefold() == node.name.strip().casefold():
                # Some records repeat the English name as the Chinese one. Scoring that copy
                # whole is the one way a bracketed address can reach the name tier, and it
                # made "forest" score "Z Café (Halal by The Forest)" at 0.66 while the places
                # actually called "The Forest" scored 0.39 -- a phrase that names WHERE a
                # café stands outranking the places it is named after.
                continue
            best = max(best, score(alias.lower()))
        if needle == node.name.lower():
            best = max(best, 96)          # typed out in full, brackets and all
        if best == 0 and tail:
            # The phrase only names WHERE this thing is, not what it is: "the library"
            # must find the Library, not the AED standing in it. Anything that matches
            # on what the place IS -- even loosely -- outranks this. No match in the
            # brackets means no match at all: an empty score must stay zero.
            bracketed = score(tail)
            if bracketed:
                best = 20 + min(19, bracketed // 4)
        if best > 0:
            scored.append((best, node))
    # A shared name at equal confidence (e.g. 圖書館) should prefer the named place
    # over a facility situated inside that place.
    scored.sort(key=lambda pair: (-pair[0], pair[1].kind == "facility", pair[1].name))
    return [(value / 100.0, node) for value, node in scored]


def resolve(graph: CampusGraph, text: str) -> list[Node]:
    """The nodes the phrase names, best first."""
    return [node for _, node in resolve_scored(graph, text)]
