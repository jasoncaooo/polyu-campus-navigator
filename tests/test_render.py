"""The rendering layer: one catalogue, one renderer.

These two modules exist to stop the same sentence existing in three places, so the tests are
about that: a missing language falls back rather than crashing, a missing key is still a bug,
and every arm of the renderer says something derived from the reply it was given.
"""
from __future__ import annotations

import unittest

from campus_nav import copy, render
from campus_nav.config import Settings, graph_path
from campus_nav.dialogue import Dialogue
from campus_nav.graph import CampusGraph
from campus_nav.place_index import PlaceIndex

OFFLINE = Settings(api_key="", model="test", llm_enabled=False, timeout_s=1)


class CatalogueTests(unittest.TestCase):
    def test_every_message_has_english(self) -> None:
        # English is the fallback, so a message without it is unusable everywhere.
        for key, row in copy.COPY.items():
            self.assertIn("en-US", row, key)

    def test_a_missing_language_falls_back_to_english(self) -> None:
        # A sentence in the wrong script beats no answer.
        self.assertEqual(copy.t("speech.route", "xx-YY", origin="A", destination="B",
                                metres=12),
                         copy.COPY["speech.route"]["en-US"].format(
                             origin="A", destination="B", metres=12))

    def test_an_unknown_key_is_a_bug_not_a_language_problem(self) -> None:
        with self.assertRaises(KeyError):
            copy.t("speech.nope", "en-US")

    def test_fields_are_filled(self) -> None:
        text = copy.t("speech.route", "en-US", origin="A", destination="B", metres=12)
        self.assertIn("A", text)
        self.assertIn("12", text)


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.places = PlaceIndex(cls.graph, OFFLINE)

    def reply(self, text: str, lang: str = "en-US"):
        return Dialogue(self.graph, client=None, narrator=None,
                        places=self.places).handle(text)

    def test_an_explicit_sentence_wins(self) -> None:
        # The handler said what to read out; a summary would only contradict it.
        class Fake:
            spoken = "explicit words"
            route = None
            facilities: list = []
            text = "ignored"
        self.assertEqual(render.speech(Fake(), self.graph, "en-US"), "explicit words")

    def test_a_route_is_said_in_the_language_asked_for(self) -> None:
        english = render.speech(self.reply("from Block V to the library"), self.graph, "en-US")
        # English names inside a Chinese sentence resolve unambiguously; "圖書館" alone is
        # three places and rightly asks a question instead of answering.
        chinese = render.speech(self.reply("由 Block V 去 Block L 點行？"), self.graph, "zh-HK")
        self.assertIn("metres", english)
        self.assertIn("米", chinese)

    def test_a_reply_with_nothing_structured_still_says_its_text(self) -> None:
        class Fake:
            spoken = ""
            route = None
            facilities: list = []
            text = "Places I can guide you to on this map:\n  - Office (45)\n  - Aed (25)"
        said = render.speech(Fake(), self.graph, "en-US")
        self.assertIn("Office", said)
        self.assertNotIn("\n", said)


class ClockTests(unittest.TestCase):
    """One seam for "now", so a net can be deterministic and a test can pin a Tuesday."""

    def test_a_pinned_clock_is_used_for_open_state(self) -> None:
        import os
        from unittest import mock
        from campus_nav.kb import clock, open_state
        with mock.patch.dict(os.environ, {"CAMPUS_NAV_NOW": "2025-06-11T10:00:00"}):
            self.assertEqual(clock().strftime("%a %H:%M"), "Wed 10:00")
            self.assertEqual(open_state({"Wed": ["07:30 - 20:30"]})[0], True)
            self.assertEqual(open_state({"Wed": ["11:00 - 20:00"]})[0], False)

    def test_an_explicit_time_outranks_the_pin(self) -> None:
        import os
        from datetime import datetime
        from unittest import mock
        from campus_nav.kb import HKT, clock
        with mock.patch.dict(os.environ, {"CAMPUS_NAV_NOW": "2025-06-11T10:00:00"}):
            self.assertEqual(clock(datetime(2025, 6, 12, 9, tzinfo=HKT)).day, 12)


class EmptyTranslationTests(unittest.TestCase):
    """An empty string is a translation; only an absent one falls back."""

    def test_an_empty_translation_is_not_replaced_with_english(self) -> None:
        from campus_nav import copy
        self.assertEqual(copy.t("hours.day.joiner", "zh-HK"), "")
        self.assertEqual(copy.t("hours.day.joiner", "en-US"), " ")
        self.assertEqual(copy.t("hours.day.joiner", "fr-FR"), " ")   # absent -> English

    def test_a_day_is_a_name_in_every_language(self) -> None:
        from campus_nav.format import day_label
        self.assertEqual(day_label("Tue", "en-US"), "Tuesday")
        self.assertEqual(day_label("Tue", "zh-HK"), "星期二")


if __name__ == "__main__":
    unittest.main()
