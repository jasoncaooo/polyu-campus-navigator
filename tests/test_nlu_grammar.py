"""The deterministic grammar: what each phrasing has to produce.

This is the fallback that answers when there is no LLM key or no network, so its
behaviour is a contract, not an implementation detail. Commit c6c647f deleted the
previous test file for this module; these cases were captured from the implementation
*after* the parser was restructured, so they pin what it does today and fail loudly if
a future refactor changes a parse.

Expected values are exact: (intent, origin_text, destination_text, item, avoid_stairs).
Place phrases stay as free text -- they are resolved later, against the map.
"""
from __future__ import annotations

import unittest

from campus_nav.config import Settings, graph_path
from campus_nav.graph import CampusGraph
from campus_nav.nlu import parse_with_grammar
from campus_nav.place_index import PlaceIndex

CASES = [
    ('how do I get from the library to Core A?', 'route', 'library', 'Core A', '', None),
    ('how do I get to the library from Core P?', 'route', 'Core P', 'library', '', None),
    ('how far is it from Block V to the library?', 'distance', 'Block V', 'library', '', None),
    ('distance between Core A and Block Z', 'distance', 'Core A', 'Block Z', '', None),
    ('Core D to Block Z', 'route', 'Core D', 'Block Z', '', None),
    ('take me to the library', 'route', '', 'library', '', None),
    ('to the library', 'route', '', 'library', '', None),
    ("I'm at the library, go to Core A", 'route', 'library', 'Core A', '', None),
    ('is the canteen open now?', 'hours_query', '', '', 'canteen', None),
    ('what time does the bookshop close?', 'hours_query', '', '', 'bookshop', None),
    ('the link between Core C and Core D is closed', 'close', '', '', '', None),
    # A closure owns the sentence: the place is resolved from the closure text inside
    # _close(), so it is not also recorded as a destination.
    ('Core A is closed', 'close', '', '', '', None),
    ('re-open everything', 'close', '', '', '', None),
    ('reset', 'reset', '', '', '', None),
    ('help', 'help', '', '', '', None),
    ('where is the nearest AED?', 'facility_query', '', '', 'aed', None),
    # "what facilities are there" mentions "UMF - Cleanroom Facility". Asking for the list
    # of amenities beats routing to a cleanroom nobody asked about.
    ('what facilities are there', 'facility_query', '', '', '', None),
    ('no stairs, from Block V to the library', 'route', 'Block V', 'library', '', True),
    ('從圖書館到賽馬會創新樓點去?', 'route', '圖書館', '賽馬會創新樓', '', None),
    ('由圖書館去飯堂', 'route', '圖書館', '飯堂', '', None),
    ('最近的飲水機喺邊?', 'facility_query', '', '', 'water station', None),
    ('飯堂幾點開門?', 'hours_query', '', '', 'canteen', None),
]


class GrammarFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        # No keys: the index answers with the built-in lexical scorer, which is exactly
        # the mode these cases were recorded in.
        cls.places = PlaceIndex(
            cls.graph,
            Settings(api_key="", model="test", llm_enabled=False, timeout_s=1),
        )

    def test_each_phrasing_parses_as_recorded(self) -> None:
        for text, intent, origin, destination, item, avoid in CASES:
            with self.subTest(text=text):
                slots = parse_with_grammar(text, self.places)
                self.assertEqual(
                    (slots.intent, slots.origin_text, slots.destination_text,
                     slots.item, slots.avoid_stairs),
                    (intent, origin, destination, item, avoid),
                )

    def test_a_stair_preference_is_read_out_of_the_sentence(self) -> None:
        slots = parse_with_grammar("avoid the stairs from the library to Core A", self.places)
        self.assertIs(slots.avoid_stairs, True)
        self.assertEqual(slots.intent, "route")

    def test_nonsense_has_no_intent(self) -> None:
        for text in ("zzzz", "asdf qwer put"):
            with self.subTest(text=text):
                self.assertEqual(parse_with_grammar(text, self.places).intent,
                                 "out_of_scope")

    def test_language_is_detected(self) -> None:
        zh = "圖書館到賽馬會"
        self.assertEqual(parse_with_grammar(zh, self.places).language, "zh")
        self.assertEqual(parse_with_grammar("the library", self.places).language, "en")


if __name__ == "__main__":
    unittest.main()
