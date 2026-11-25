"""Properties that must hold for phrasings nobody wrote a rule for.

Every bug in this project's history was a rule that was true for the sentence that prompted
it and false for the next one: a bare "to" that matched every "to" in the language, a
resolver that answered "Hi" with a department, a pair rule that split a fifteen-word clause
at its first preposition, a plural that matched no published name. So this file does not
re-test the sentences that were fixed. It tests the *properties* those fixes established,
over phrasings deliberately unlike them.

A failure here means the fix was a patch. Fix the property, not the sentence.
"""
from __future__ import annotations

import unittest

from campus_nav.config import Settings, graph_path
from campus_nav.graph import CampusGraph
from campus_nav.nlu import parse_with_grammar
from campus_nav.place_index import MEANING_BAND, PlaceIndex
from campus_nav.places import amenity_in_text, facility_capability

OFFLINE = Settings(api_key="", model="test", llm_enabled=False, timeout_s=1)


class ResolverPropertyTests(unittest.TestCase):
    """A place is only named when the words say so."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.places = PlaceIndex(cls.graph, OFFLINE)

    def test_conversation_names_no_place(self) -> None:
        # How people talk around a request. Each of these landed on a real campus place
        # before the evidence rule, because dense matching always returns neighbours.
        for phrase in ("hi", "hello there", "hey", "thanks", "ok", "sure", "want",
                       "I want", "on the way", "make sure", "please", "sorry",
                       "what's up", "good morning", "yes", "no thanks"):
            self.assertEqual(self.places.resolve(phrase), [], phrase)

    def test_one_unknown_word_names_nothing(self) -> None:
        for phrase in ("frobnicator", "xyzzy", "blorp", "wanting", "librarium"):
            self.assertEqual(self.places.resolve(phrase), [], phrase)

    def test_one_word_that_is_a_name_still_resolves(self) -> None:
        # The rule narrows nothing it should not: a single word IS allowed to name a place
        # when it appears in one.
        for phrase in ("library", "bookshop", "canteen", "aed", "water", "clinic"):
            self.assertTrue(self.places.resolve(phrase), phrase)

    def test_a_plural_finds_the_published_singular(self) -> None:
        # Names are published in the singular and people ask in the plural.
        for plural, singular in (("toilets", "toilet"), ("clinics", "clinic"),
                                 ("libraries", "library"), ("bookshops", "bookshop")):
            names = [node.name.casefold() for node in self.places.resolve(plural)]
            self.assertTrue(any(singular in name for name in names), plural)

    def test_a_campus_code_resolves_exactly_or_not_at_all(self) -> None:
        # One letter separates any two cores, which is exactly what an embedding cannot
        # see, so the convention is matched deterministically.
        self.assertEqual([n.name for n in self.places.resolve("core p")], ["Core P"])
        self.assertEqual([n.name for n in self.places.resolve("Block VA")],
                         ["Block VA (Shaw Amenities Building)"])
        self.assertEqual([n.name for n in self.places.resolve("wing fj")],
                         ["Wing FJ (Chan Tai Ho Building)"])
        for missing in ("core y", "block q", "core aa", "tower a", "clore p"):
            self.assertEqual(self.places.resolve(missing), [], missing)

    def test_a_longer_name_is_not_reduced_to_the_block_it_starts_with(self) -> None:
        self.assertEqual([n.name for n in self.places.resolve("Block X Sports Centre")],
                         ["Block X Sports Centre"])


class AmenityWordingTests(unittest.TestCase):
    """The engine owns the amenity vocabulary, and says so in one function."""

    def test_wording_that_names_an_amenity(self) -> None:
        # Asserted as the capability, not the wording: which of two matching words is found
        # is an implementation detail, what it means is not.
        for text, expected in (("where is the nearest AED", "defibrillator"),
                               ("any washroom near Core P", "toilet"),
                               ("fill my bottle at a water station", "water_refill"),
                               ("is the canteen open", "canteen"),
                               ("I need cash from an ATM", "bank"),
                               ("I need to print something", "bookstore"),
                               ("where do I go if I feel sick", "clinic"),
                               ("somewhere to eat lunch", "canteen")):
            self.assertEqual(facility_capability(amenity_in_text(text)), expected, text)

    def test_wording_that_names_none(self) -> None:
        # Activities and descriptions are not amenities, and saying so here is what lets the
        # dialogue search places by meaning instead of picking a wrong neighbour.
        for text in ("I want to have a swim", "where can I swim",
                     "somewhere to sit outside and study", "a quiet place to read",
                     "I broke my phone screen, can someone fix it", ""):
            self.assertEqual(amenity_in_text(text), "", text)


class MeaningSearchTests(unittest.TestCase):
    """A need is answered by a place, and only by places that mean the same thing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())

    def index(self, hits):
        class Stub(PlaceIndex):
            def __init__(self, graph):
                super().__init__(graph, OFFLINE)

            def scored(self, _text, _k=None):
                return hits

        return Stub(self.graph)

    def node(self, name: str):
        for candidate in self.graph.nodes.values():
            if candidate.name == name:
                return candidate
        raise AssertionError(name)

    def test_a_weaker_place_of_another_kind_is_not_an_answer(self) -> None:
        # Observed live: "I want to have a swim" ranked the pool first (0.53) and a water
        # fountain second (0.39), and both cleared the floor -- then distance sorting put
        # the fountain on top, which is a confident wrong answer.
        pool = self.node("Michael Clinton Swimming Pool")
        fountain = self.node("Water Station (Core P)")
        index = self.index([(0.53, pool), (0.39, fountain)])
        self.assertEqual([n.name for n in index.semantic_places("have a swim")], [pool.name])

    def test_near_equal_places_are_all_offered(self) -> None:
        # "somewhere to sit outside and study" has several answers within a hundredth.
        plaza = self.node("Stephen Cheong Kam Chuen Memorial Plaza")
        centre = self.node("Student Computer Centre (SCC)")
        index = self.index([(0.42, plaza), (0.41, centre)])
        self.assertEqual({n.name for n in index.semantic_places("sit outside")},
                         {plaza.name, centre.name})

    def test_the_band_is_what_decides(self) -> None:
        pool = self.node("Michael Clinton Swimming Pool")
        fountain = self.node("Water Station (Core P)")
        edge = [(1.0, pool), (1.0 - MEANING_BAND - 0.01, fountain)]
        self.assertEqual(len(self.index(edge).semantic_places("x")), 1)

    def test_nothing_scores_nothing(self) -> None:
        self.assertEqual(self.index([]).semantic_places("anything"), [])
        self.assertEqual(self.index([(0.9, self.node("Core A"))]).semantic_places("   "), [])


class GrammarPropertyTests(unittest.TestCase):
    """A sentence is understood as the kind of sentence it is."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.places = PlaceIndex(cls.graph, OFFLINE)

    def slots(self, text: str):
        return parse_with_grammar(text, self.places)

    def test_a_preposition_is_not_a_route(self) -> None:
        # "to" appears in most English sentences. Each of these contains one and none of
        # them is a journey to the words that follow it.
        for text in ("but I want to make sure I pass through Core P on the way",
                     "I want to tell you that the library is lovely",
                     "it is hard to find the canteen",
                     "good to know"):
            slots = self.slots(text)
            for junk in ("want", "make", "sure", "pass", "tell", "hard", "know"):
                self.assertNotIn(junk, slots.destination_text.lower(), text)
                self.assertNotIn(junk, slots.origin_text.lower(), text)

    def test_a_trailing_stop_clause_becomes_the_stop(self) -> None:
        for text, origin, destination, via in (
            ("take me from Block Y to Core A but stop at Core P", "Block Y", "Core A", "Core P"),
            ("from Core D to Block Z via Core P", "Core D", "Block Z", "Core P"),
            ("walk me from the library to Block X through Core P", "library", "Block X", "Core P"),
            ("pass by Core P on the way from Core D to Block Z", "Core D", "Block Z", "Core P"),
        ):
            slots = self.slots(text)
            self.assertEqual(
                (slots.origin_text, slots.destination_text, ", ".join(slots.vias)),
                (origin, destination, via), text)
            self.assertEqual(slots.intent, "route", text)

    def test_a_stop_alone_is_still_a_route_to_re_plan(self) -> None:
        # No endpoints: the session already has them, and the sentence is about the stop.
        slots = self.slots("make sure I pass through Core P on the way")
        self.assertEqual(slots.intent, "route")
        self.assertEqual(slots.vias, ["Core P"])
        self.assertEqual(slots.destination_text, "")

    def test_a_greeting_in_front_does_not_swallow_the_question(self) -> None:
        slots = self.slots("hi, how do I get to the library?")
        self.assertEqual(slots.intent, "route")
        self.assertEqual(slots.destination_text, "library")

    def test_a_need_only_the_engine_vocabulary_knows_is_understood(self) -> None:
        # The grammar's own word list is short. These are needs that only places.
        # FACILITY_WORDS knows, and they used to fall through to "I can find walking
        # routes, point you to facilities..." -- or worse, route to a department called
        # "Food Science and Nutrition" because the sentence mentioned food.
        for text in ("i want to grab lunch, where can i head to?",
                     "where can I get food",
                     "I feel sick",
                     "where can I get cash",
                     "I need to print something",
                     "is there a doctor on campus"):
            self.assertEqual(self.slots(text).intent, "facility_query", text)

    def test_the_engine_vocabulary_does_not_outrank_a_route_or_a_closure(self) -> None:
        # "lunch" must not turn a route into a facility list, or a closure into a search.
        self.assertEqual(self.slots("how do I get to the canteen?").intent, "route")
        self.assertEqual(self.slots("the canteen is closed").intent, "close")
        self.assertEqual(self.slots("is the canteen open?").intent, "hours_query")

    def test_an_overview_question_is_not_a_route(self) -> None:
        # "what facilities are there" mentions a place called "UMF - Cleanroom Facility";
        # a request for the list of amenities is not a journey to a cleanroom.
        for text in ("what facilities are there", "what amenities can I find",
                     "what services are available"):
            self.assertEqual(self.slots(text).intent, "facility_query", text)

    def test_a_closure_is_recognised_in_both_scripts(self) -> None:
        for text in ("the bookshop is shut", "Core A is closed", "圖書館今日關閉",
                     "P座裝修"):
            slots = self.slots(text)
            self.assertEqual(slots.intent, "close", text)
            self.assertTrue(slots.closed_text, text)

    def test_a_timetable_question_is_not_a_closure(self) -> None:
        for text in ("is the canteen open now?", "what time does the bookshop close?"):
            slots = self.slots(text)
            self.assertEqual(slots.intent, "hours_query", text)


if __name__ == "__main__":
    unittest.main()
