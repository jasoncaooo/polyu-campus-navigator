"""Place-name resolution: banding, grouping, and the two-rung fallback contract.

The network clients are replaced by stubs that return chosen similarities, so these
tests pin the policy -- what counts as an answer, what counts as "I do not know", and
what happens when the vector index is unusable -- without embedding a single string.
"""
from __future__ import annotations

import unittest

from campus_nav.config import Settings
from campus_nav.graph import CampusGraph, Node
from campus_nav.pinecone import PineconeError
from campus_nav.place_index import (
    MODE_DENSE,
    MODE_LEXICAL,
    PlaceIndex,
    document_text,
)

# A dense-capable configuration: keys present, so check() trusts the store's stats.
DENSE = Settings(
    api_key="", model="test", llm_enabled=False, timeout_s=1,
    place_min_similarity=0.30, place_ambiguity_band=0.05, place_top_k=5,
    embeddings_api_key="test-key", pinecone_api_key="test-key",
)
# No vector keys at all: the mode the shipped demo runs in with an empty .env.
OFFLINE = Settings(api_key="", model="test", llm_enabled=False, timeout_s=1)


class StubEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        self.texts.append(text)
        return [0.0]


class StubStore:
    def __init__(self, matches: list[tuple[float, str]], vectors: int = 3,
                 fail: bool = False) -> None:
        self.matches = matches
        self.vectors = vectors
        self.fail = fail

    def query(self, vector: list[float], top_k: int) -> list[tuple[float, dict]]:
        if self.fail:
            raise PineconeError("vector store is down")
        return [(score, {"node_id": node_id}) for score, node_id in self.matches[:top_k]]

    def stats(self) -> dict:
        return {"totalVectorCount": self.vectors, "dimension": 1}


def make_graph() -> CampusGraph:
    nodes = {
        "L": Node(id="L", name="Pao Yue-Kong Library", lat=0.0, lng=0.0, kind="place"),
        "AED": Node(id="AED", name="AED (Library)", lat=0.0, lng=0.0, kind="facility"),
        "X": Node(id="X", name="Elsewhere", lat=0.0, lng=0.0, kind="place"),
        "W": Node(id="W", name="", lat=0.0, lng=0.0, kind="waypoint"),
    }
    return CampusGraph(nodes=nodes, edges=[], meta={})


def make_index(matches: list[tuple[float, str]], *, settings: Settings = DENSE,
               vectors: int = 3, fail: bool = False) -> tuple[PlaceIndex, StubEmbedder]:
    embedder = StubEmbedder()
    index = PlaceIndex(
        make_graph(), settings, embedder=embedder,
        store=StubStore(matches, vectors=vectors, fail=fail),
    )
    index.check()          # select the mode from the stubbed store, as the app does
    return index, embedder


class DensePlaceIndexTests(unittest.TestCase):
    """The embedding path: nearest neighbours, and only the top band is an answer."""

    def test_only_the_top_band_is_an_answer(self) -> None:
        index, _ = make_index([(0.90, "L"), (0.50, "X")])
        self.assertEqual(index.mode, MODE_DENSE)
        self.assertEqual([node.id for node in index.resolve("the library")], ["L"])

    def test_close_scores_are_reported_as_ambiguous(self) -> None:
        # No name evidence either way ("a quiet spot" is in no published name), so the
        # embeddings are the only evidence and two near neighbours are a real choice.
        index, _ = make_index([(0.90, "L"), (0.88, "X")])
        self.assertEqual([node.id for node in index.resolve("a quiet spot")], ["L", "X"])

    def test_a_published_name_beats_embedding_neighbours(self) -> None:
        # "library" is a word in the library's published name, so the place that merely
        # sits closest in the embedding space is a neighbour, not an alternative.
        index, _ = make_index([(0.90, "X"), (0.88, "L")])
        self.assertEqual([node.id for node in index.resolve("library")], ["L"])

    def test_nothing_above_the_floor_is_no_match(self) -> None:
        index, _ = make_index([(0.20, "L")])
        self.assertEqual(index.resolve("library"), [])

    def test_unknown_and_waypoint_ids_are_ignored(self) -> None:
        index, _ = make_index([(0.99, "missing"), (0.98, "W"), (0.50, "L")])
        self.assertEqual([node.id for node in index.resolve("library")], ["L"])

    def test_repeated_phrase_embeds_once(self) -> None:
        index, embedder = make_index([(0.90, "L")])
        index.resolve("library")
        index.resolve("library")
        self.assertEqual(len(embedder.texts), 1)

    def test_a_failing_store_degrades_instead_of_raising(self) -> None:
        # A vector store that dies mid-session must not break the conversation.
        index, _ = make_index([], fail=True)
        self.assertEqual([node.id for node in index.resolve("library")], ["L"])
        self.assertEqual(index.mode, MODE_LEXICAL)
        self.assertIn("fell back", index.warning)


class FallbackTests(unittest.TestCase):
    """The lexical path: the app must work with no keys and no network."""

    def test_missing_keys_fall_back_to_the_builtin_scorer(self) -> None:
        index = PlaceIndex(make_graph(), OFFLINE, embedder=StubEmbedder(),
                           store=StubStore([]))
        info = index.check()
        self.assertEqual(info["mode"], MODE_LEXICAL)
        self.assertIn("EMBEDDINGS_API_KEY", info["warning"])
        self.assertEqual([node.id for node in index.resolve("library")], ["L"])

    def test_empty_index_falls_back_and_says_how_to_fill_it(self) -> None:
        index, _ = make_index([], vectors=0)
        self.assertEqual(index.mode, MODE_LEXICAL)
        self.assertIn("build.py embed", index.warning)

    def test_lexical_scorer_keeps_block_letters(self) -> None:
        # "Core A" must not soften into "Core", which matches every core on the estate.
        graph = CampusGraph(nodes={
            "CA": Node(id="CA", name="Core A", lat=0.0, lng=0.0, kind="place"),
            "CB": Node(id="CB", name="Core B", lat=0.0, lng=0.0, kind="place"),
        }, edges=[], meta={})
        index = PlaceIndex(graph, OFFLINE, embedder=StubEmbedder(), store=StubStore([]))
        index.check()
        self.assertEqual([node.id for node in index.resolve("Core A")], ["CA"])

    def test_lexical_scorer_answers_nothing_for_nonsense(self) -> None:
        index = PlaceIndex(make_graph(), OFFLINE, embedder=StubEmbedder(),
                           store=StubStore([]))
        index.check()
        self.assertEqual(index.resolve("zzzz"), [])

    def test_document_text_carries_both_scripts(self) -> None:
        node = Node(id="L", name="Pao Yue-Kong Library", lat=0.0, lng=0.0,
                    name_zh="包玉剛圖書館", name_zh_cn="包玉刚图书馆", kind="place")
        text = document_text(node)
        self.assertIn("Pao Yue-Kong Library", text)
        self.assertIn("包玉剛圖書館", text)
        self.assertIn("包玉刚图书馆", text)


if __name__ == "__main__":
    unittest.main()
