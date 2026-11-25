"""The dialogue layer's own decisions: no model, no network, no server.

These are the behaviours the demo leans on and that no other test touches -- the walk time
on a receipt, a closure that names its candidates instead of guessing, and the escape hatch
that re-opens everything.
"""
from __future__ import annotations

import unittest

from datetime import datetime, timedelta

from campus_nav.config import Settings, graph_path
from campus_nav.format import DAYS
from campus_nav.kb import HKT
from campus_nav import dialogue as dialogue_module
from campus_nav.dialogue import HISTORY_TURNS, Dialogue, place_legs, walk_minutes
from campus_nav.format import FacilityResult
from campus_nav.nlu import Slots
from campus_nav.graph import CampusGraph
from campus_nav.place_index import PlaceIndex

# No vector keys and no model: the index answers from the graph and the grammar parses.
OFFLINE = Settings(api_key="", model="test", llm_enabled=False, timeout_s=1)


class WalkTimeTests(unittest.TestCase):
    def test_distance_is_divided_by_the_pace(self) -> None:
        # 1.25 m/s is the easy walking pace this project quotes: 75 m a minute.
        self.assertEqual(walk_minutes(75.0), 1)
        self.assertEqual(walk_minutes(206.0), 3)
        self.assertEqual(walk_minutes(750.0), 10)

    def test_a_walk_is_never_zero_minutes(self) -> None:
        self.assertEqual(walk_minutes(0.0), 1)
        self.assertEqual(walk_minutes(4.0), 1)


class _NothingResolves:
    """A resolver that accepts nothing, so the zero-match path runs offline.

    The dense index nearly always returns *some* neighbour, so the "nothing matched"
    branch is hard to reach through the real one — but it is the branch a user hits with a
    typo or a place that is not on this campus, and it must suggest rather than shrug.
    """

    def __init__(self, places: PlaceIndex) -> None:
        self._places = places

    def resolve(self, _text: str) -> list:
        return []

    def coded_siblings(self, _text: str) -> list:
        return []

    def scored(self, text: str):
        return self._places.scored(text)


class DialogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.places = PlaceIndex(cls.graph, OFFLINE)

    def dialogue(self) -> Dialogue:
        return Dialogue(self.graph, client=None, narrator=None, places=self.places)

    def test_a_route_receipt_states_the_walk_time(self) -> None:
        reply = self.dialogue().handle("how do I get from Block V to the library?")
        self.assertIn("easy pace", reply.text)

    def test_a_code_that_does_not_exist_suggests_the_same_letter_elsewhere(self) -> None:
        # "Core Y" is not a place; "Block Y" is. The scorer offers Core T, Core C, Core J.
        talk = self.dialogue()
        reply = talk.handle("from Core Y to Core A")
        self.assertIn("I do not know", reply.text)
        self.assertIn("Block Y", reply.text)

    def test_a_greeting_is_answered_not_guessed_at(self) -> None:
        # "Hi" reached the resolver, which found the HTI department and asked which of
        # three places the user meant.
        for word in ("Hi", "hello", "hey!", "thanks"):
            reply = self.dialogue().handle(word)
            self.assertIn("greeting", reply.notes, word)
            self.assertNotIn("could be several places", reply.text, word)

    def test_a_misspelt_place_gets_a_suggestion(self) -> None:
        # Offline the scorer scores an unknown word zero -- by design, so that a phrase
        # matching nothing returns nothing. "Did you mean" then has to come from spelling.
        talk = Dialogue(self.graph, places=_NothingResolves(self.places))
        reply = talk.handle("how do I get to the libarry?")
        self.assertIn("Did you mean", reply.text)
        self.assertIn("Library", reply.text)

    def test_an_ambiguous_closure_names_the_candidates_and_changes_nothing(self) -> None:
        # This campus has three canteens: closing "the canteen" would be a guess, and a
        # wrong closure silently invalidates every later route.
        talk = self.dialogue()
        reply = talk.handle("the canteen is closed")
        self.assertIn("could be several places", reply.text)
        self.assertEqual(len(talk.state.closed_nodes), 0)

    def test_closing_the_link_between_two_places_closes_the_stretch(self) -> None:
        # Core C and Core D are five links and 55 m apart. "The link between them" is that
        # stretch; requiring one drawn link meant the most natural way to describe a
        # closure in this app closed nothing at all.
        talk = self.dialogue()
        reply = talk.handle("the link between Core C and Core D is closed")
        self.assertTrue(talk.state.closed_edges, reply.text)
        self.assertIn("stretch", reply.text)

    def test_a_named_closure_is_closed_and_can_be_re_opened(self) -> None:
        talk = self.dialogue()
        closed = talk.handle("the bookshop is closed")
        self.assertIn("Closed", closed.text)
        self.assertEqual(len(talk.state.closed_nodes), 1)
        reopened = talk.handle("re-open it")
        self.assertIn("Re-opened", reopened.text)
        self.assertEqual(len(talk.state.closed_nodes), 0)


class PlaceNamingTests(unittest.TestCase):
    """What a question is about: the phrase, and the building it names."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.places = PlaceIndex(cls.graph, OFFLINE)

    def test_the_asking_frame_is_not_part_of_the_name(self) -> None:
        # A whole sentence is not a name: handed to the resolver, its own words matched
        # every building on the map and "the library" was answered with a defibrillator.
        from campus_nav.nlu import topic_phrase
        cases = {
            "when does the library open on Tuesday?": "library",
            "what are the canteen's opening hours?": "canteen",
            "is the canteen open?": "canteen",
            "what time does the clinic close?": "clinic",
            "when does the bookshop close?": "bookshop",
            "Pao Yue-Kong Library": "Pao Yue-Kong Library",
        }
        for text, phrase in cases.items():
            self.assertEqual(topic_phrase(text), phrase, text)

    def test_the_english_name_answers_the_building(self) -> None:
        # True before this rule as well: the building's own English name carries the word.
        names = [node.name for node in self.places.resolve("library")]
        self.assertEqual(names, ["Block L (Pao Yue-Kong Library)"])

    def test_the_chinese_name_answers_the_building(self) -> None:
        # 圖書館 is nobody's whole published name, so the library, the café inside it and
        # the defibrillator inside it all came back as three equal offers. The building is
        # what was named; the café and the AED are what stands in it.
        for query in ("圖書館", "图书馆"):
            names = [node.name for node in self.places.resolve(query)]
            self.assertEqual(names, ["Block L (Pao Yue-Kong Library)"], query)

    def test_a_feature_still_answers_to_its_own_words(self) -> None:
        names = [node.name for node in self.places.resolve("AED")]
        self.assertTrue(names, "AED must still resolve")
        self.assertTrue(all("AED" in name for name in names), names)

    def test_a_generic_category_still_lists_its_members(self) -> None:
        # The building-over-feature rule must not swallow a question about a category.
        names = [node.name for node in self.places.resolve("clinic")]
        self.assertGreater(len(names), 1)
        self.assertFalse(any(name.startswith("Block") for name in names), names)


class _NamesAPlace:
    """A resolver that names one place where the semantic sweep would offer four.

    The dense index does this: asked to *describe* "the library" it returns the AED mounted
    inside it, the café inside it, a bookshop, and the building -- defibrillator nearest.
    Asked what the words *name*, it returns the building.
    """

    def __init__(self, places: PlaceIndex, named: list, described: list) -> None:
        self._places = places
        self._named = named
        self._described = described

    def resolve(self, _text: str) -> list:
        return list(self._named)

    def semantic_places(self, _text: str) -> list:
        return list(self._described)

    def scored(self, text: str):
        return self._places.scored(text)


class NamedPlaceTests(unittest.TestCase):
    """A phrase that names a place is answered with that place, not a search result."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.places = PlaceIndex(cls.graph, OFFLINE)
        cls.library = [n for n in cls.graph.nodes.values()
                       if n.name == "Block L (Pao Yue-Kong Library)"]
        cls.others = [n for n in cls.graph.nodes.values()
                      if n.name in ("AED (Library)", "LibCafé", "Bookshop")]

    def test_the_named_place_answers_instead_of_the_meaning_sweep(self) -> None:
        talk = Dialogue(self.graph, places=_NamesAPlace(self.places, self.library, self.others))
        reply = talk._facilities(Slots(intent="facility_query", subject="library",
                                      language="en"))
        self.assertEqual([c["name"] for c in reply.facility_result.rows],
                         ["Block L (Pao Yue-Kong Library)"])
        self.assertEqual(reply.facility_result.kind, "list")
        self.assertNotIn("place(s) for", reply.text)

    def test_a_description_still_uses_the_meaning_sweep(self) -> None:
        # "somewhere to eat" names nothing, so the sweep is the only thing that can answer.
        talk = Dialogue(self.graph, places=_NamesAPlace(self.places, [], self.others))
        reply = talk._facilities(Slots(intent="facility_query", subject="somewhere to eat",
                                      language="en"))
        self.assertEqual(len(reply.facility_result.rows), len(self.others))
        self.assertIn("Closest places for", reply.text)


class DayTests(unittest.TestCase):
    """The day survives the frame strip, in both scripts."""

    def test_the_day_is_read_before_the_phrase_is_cleaned(self) -> None:
        from campus_nav.format import requested_day
        from campus_nav.nlu import day_phrase, topic_phrase
        # The phrase loses its frame; the day it asked about is kept beside it, because
        # "on Tuesday" is part of the asking and the timetable still owes an answer for it.
        self.assertEqual(topic_phrase("when does the library open on Tuesday?"), "library")
        self.assertEqual(requested_day(day_phrase("when does the library open on Tuesday?")),
                         "Tue")
        # The day is asking frame in Chinese too: the name is 圖書館, and 聽日 is the day.
        self.assertEqual(topic_phrase("圖書館聽日幾點開"), "圖書館")
        self.assertEqual(topic_phrase("明天圖書館幾點開"), "圖書館")
        self.assertEqual(requested_day(day_phrase("圖書館聽日幾點開")),
                         DAYS[(datetime.now(HKT) + timedelta(days=1)).weekday()])
        # A question that names no place at all ("what are the hours on Friday?") leaves the
        # phrase empty -- "hours" is asking wording, not a name -- and still knows the day.
        self.assertEqual(topic_phrase("what are the hours on Friday?"), "")
        self.assertEqual(requested_day(day_phrase("what are the hours on Friday?")), "Fri")

    def test_a_named_weekday_answers_that_weekday(self) -> None:
        # English weekdays used to be read as no day at all, so "on Tuesday" was answered
        # with today's hours -- a wrong answer, not an imprecise one.
        from campus_nav.format import requested_day
        self.assertEqual(requested_day("on Tuesday"), "Tue")
        self.assertEqual(requested_day("sunday"), "Sun")
        self.assertEqual(requested_day("公眾假期"), "PublicHoliday")


class StopRoutingTests(unittest.TestCase):
    """A walk through stops, with alternatives that are real searches."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.places = PlaceIndex(cls.graph, OFFLINE)

    def dialogue(self) -> Dialogue:
        return Dialogue(self.graph, client=None, narrator=None, places=self.places)

    def node(self, name: str) -> str:
        for candidate in self.graph.nodes.values():
            if candidate.name == name:
                return candidate.id
        raise AssertionError(name)

    def test_one_stop_still_offers_alternatives(self) -> None:
        # It used to return exactly one route: "the second best way through Core P is not a
        # question anybody asked". It is, as soon as somebody wants to choose.
        routes = self.dialogue()._routed_via(
            self.node("Block Y (Lee Shau Kee Building)"),
            [self.node("Core P")],
            self.node("Core A"),
        )
        self.assertGreater(len(routes), 1)

    def test_every_option_goes_through_every_stop_in_order(self) -> None:
        stops = [self.node("Core P"), self.node("Block L (Pao Yue-Kong Library)")]
        routes = self.dialogue()._routed_via(
            self.node("Block Y (Lee Shau Kee Building)"), stops, self.node("Core A")
        )
        for route in routes:
            positions = [route.path.index(stop) for stop in stops]
            self.assertEqual(positions, sorted(positions), route.path)

    def test_a_stop_walk_is_measured_as_one_walk(self) -> None:
        # The total is the sum of the legs, and the legs in the reply come from the joined
        # path -- three named places means two legs, not one.
        stop = self.node("Core P")
        routes = self.dialogue()._routed_via(
            self.node("Block Y (Lee Shau Kee Building)"), [stop], self.node("Core A")
        )
        walk = routes[0]
        self.assertEqual(walk.path.count(stop), 1)
        self.assertEqual(len(place_legs(self.graph, walk)), 2)
        self.assertAlmostEqual(
            walk.total_metres, sum(segment.metres for segment in walk.segments), places=3
        )

    def test_a_blocked_leg_returns_one_answer_that_explains_it(self) -> None:
        talk = self.dialogue()
        stop = self.node("Core P")
        talk.state.closed_nodes.add(stop)
        routes = talk._routed_via(
            self.node("Block Y (Lee Shau Kee Building)"), [stop], self.node("Core A")
        )
        self.assertEqual(len(routes), 1)
        self.assertFalse(routes[0].reachable)

class MemoryLifecycleTests(unittest.TestCase):
    """What the conversation remembers, for how long, and what forgets it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.places = PlaceIndex(cls.graph, OFFLINE)

    def dialogue(self) -> Dialogue:
        return Dialogue(self.graph, client=None, narrator=None, places=self.places)

    def test_the_history_is_trimmed_to_its_cap(self) -> None:
        talk = self.dialogue()
        for _ in range(6):
            talk.handle("how do I get from Block V to the library?")
        self.assertEqual(len(talk.history), HISTORY_TURNS)
        self.assertEqual(len(talk.history) % 2, 0, "whole exchanges, not half of one")

    def test_reset_forgets_the_history_as_well_as_the_state(self) -> None:
        # Clearing one left the parser able to read a conversation the user had erased.
        talk = self.dialogue()
        talk.handle("how do I get from Block V to the library?")
        self.assertTrue(talk.history)
        talk.reset()
        self.assertEqual(talk.history, [])
        self.assertIsNone(talk.state.origin)

    def test_the_journey_survives_as_a_state_line_not_a_transcript(self) -> None:
        # Six messages cannot carry a journey through a long aside; the state can, and it is
        # a fact rather than prose, so it is handed over separately.
        talk = self.dialogue()
        talk.handle("how do I get from Block V to the library?")
        digest = talk._context_digest()
        self.assertIsNotNone(digest)
        self.assertIn("CURRENT JOURNEY", digest["content"])
        self.assertIn("Block V", digest["content"])
        self.assertEqual(digest["role"], "system")

    def test_a_fresh_conversation_has_no_state_line(self) -> None:
        self.assertIsNone(self.dialogue()._context_digest())


class JourneyStopsTests(unittest.TestCase):
    """Stops that persist across turns, and a stop found by description."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.places = PlaceIndex(cls.graph, OFFLINE)

    def dialogue(self, origin: str, destination: str) -> Dialogue:
        talk = Dialogue(self.graph, client=None, narrator=None, places=self.places)
        talk.state.origin = self.node("Block L (Pao Yue-Kong Library)")
        talk.state.destination = self.node("Block Y (Lee Shau Kee Building)")
        return talk

    def node(self, name: str) -> str:
        for candidate in self.graph.nodes.values():
            if candidate.name == name:
                return candidate.id
        raise AssertionError(name)

    def test_a_stop_said_later_extends_the_journey(self) -> None:
        # "Along the way, I want to pass through Block X" used to apply for one reply and
        # then vanish; the journey is the thing being edited, so it keeps its stops.
        talk = self.dialogue("library", "Block Y")
        talk.handle("how do I get from the library to Block Y?")
        talk._route_reply(Slots(intent="route", vias=["Block X"]))
        self.assertEqual(talk.state.stops, [self.node("Block X")])
        self.assertEqual(len(place_legs(self.graph, talk.state.last_route)), 2)

    def test_naming_a_new_journey_forgets_the_old_stops(self) -> None:
        talk = self.dialogue("library", "Block Y")
        talk._route_reply(Slots(intent="route", vias=["Block X"]))
        talk._route_reply(Slots(intent="route", origin_text="Block L (Pao Yue-Kong Library)",
                               destination_text="Core A"))
        self.assertEqual(talk.state.stops, [])

    def test_no_stops_clears_them(self) -> None:
        talk = self.dialogue("library", "Block Y")
        talk._route_reply(Slots(intent="route", vias=["Block X"]))
        self.assertTrue(talk.state.stops)
        talk.handle("no stops, the direct route")
        self.assertEqual(talk.state.stops, [])

    def test_a_described_stop_is_inserted_with_its_detour(self) -> None:
        # "I also want to get a quick breakfast along the way": the engine decides that
        # breakfast is catering, and which canteen costs the least extra walking.
        talk = self.dialogue("library", "Block Y")
        reply = talk._stop_for_need(Slots(intent="facility_query", need="a quick breakfast"))
        self.assertIsNotNone(reply, "a breakfast stop should be found near this walk")
        self.assertIn("Added", reply.text)
        self.assertEqual(len(talk.state.stops), 1, talk.state.stops)
        self.assertIn(talk.state.stops[0], talk.state.last_route.path)

    def test_a_stop_that_costs_too_much_is_offered_not_inserted(self) -> None:
        talk = self.dialogue("library", "Block Y")
        talk_module = dialogue_module
        original = talk_module.STOP_DETOUR_M
        talk_module.STOP_DETOUR_M = 0.0       # every detour is now too expensive
        try:
            reply = talk._stop_for_need(Slots(intent="facility_query", need="a quick breakfast"))
        finally:
            talk_module.STOP_DETOUR_M = original
        self.assertIsNotNone(reply)
        self.assertIn("would add", reply.text)
        self.assertEqual(talk.state.stops, [], "nothing is inserted without consent")

    def test_answering_a_question_keeps_the_walk_on_the_map(self) -> None:
        # Turn 3 of a journey is often a question ("where is the nearest toilet?"). It used
        # to clear the drawn route, which reads as the app forgetting the walk.
        talk = self.dialogue("library", "Block Y")
        talk.handle("how do I get from the library to Block Y?")
        reply = talk.handle("where is the nearest AED?")
        self.assertIsNone(reply.route, "a facility answer is not itself a route")
        self.assertTrue(reply.routes, "but the walk stays on the map")

class ListFollowUpTests(unittest.TestCase):
    """Choosing from the list we just showed, and ids that are ids."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.places = PlaceIndex(cls.graph, OFFLINE)

    def dialogue(self) -> Dialogue:
        return Dialogue(self.graph, client=None, narrator=None, places=self.places)

    def node(self, name: str) -> str:
        for candidate in self.graph.nodes.values():
            if candidate.name == name:
                return candidate.id
        raise AssertionError(name)

    def listing(self, *names: str) -> FacilityResult:
        rows = [{"node_id": self.node(name), "name": name, "metres": 120} for name in names]
        return FacilityResult("list", "lunch", len(rows), "", rows=rows)

    # The "pointing at the list" heuristic that used to live here was removed with the
    # stateless parser: the model resolves "the nearest one" from the conversation now, and
    # keeping a phrase list for it was one of the patches that made this layer grow.

    def test_the_choice_re_ranks_from_where_the_user_says_they_are(self) -> None:
        # "the nearest from here": the list was measured from Core P, which is not here.
        talk = self.dialogue()
        talk.state.last_facilities = self.listing("U Green (Staff Club)", "LibCafé")
        slots = Slots(intent="route", facility_choice=0,
                      origin_text="Block L (Pao Yue-Kong Library)")
        reply = talk._facility_choice(slots)
        self.assertIsNotNone(reply.route)
        self.assertEqual(self.graph.node(reply.route.path[-1]).name, "LibCafé")
        self.assertEqual(reply.route.path[0], self.node("Block L (Pao Yue-Kong Library)"))

    def test_a_node_id_is_never_resolved_as_if_it_were_a_name(self) -> None:
        # The id went through the embeddings and came back as Core B: an id is not a name.
        talk = self.dialogue()
        cafe = self.node("LibCafé")
        talk._absorb_places(Slots(intent="route", destination_text=cafe))
        self.assertEqual(talk.state.destination, cafe)

    def test_a_place_you_are_standing_in_is_already_there(self) -> None:
        # A facility published at the same coordinate as the place you named: a zero-metre
        # "route" is not an answer. (LibCafé, by contrast, is a real 35 m walk from the
        # library door and keeps its route -- see the test above.)
        talk = self.dialogue()
        talk.state.last_facilities = self.listing("Toilet (Core P)")
        talk.state.origin = self.node("Core P")
        reply = talk._facility_choice(Slots(intent="route", facility_choice=0))
        self.assertIsNone(reply.route)
        self.assertIn("already", reply.text)

if __name__ == "__main__":
    unittest.main()
