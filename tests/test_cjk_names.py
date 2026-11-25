"""Chinese name evidence: head-final matching, and place before facility."""
from __future__ import annotations

import types
import unittest

from campus_nav.place_index import cjk_named


def node(name: str, kind: str, zh: str, zh_cn: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(name=name, kind=kind, name_zh=zh, name_zh_cn=zh_cn)


# The three places the score put in this order for "圖書館", as published.
LIBRARY = node("Block L (Pao Yue-Kong Library)", "place", "L座(包玉剛圖書館)", "L座(包玉刚图书馆)")
CAFE = node("LibCafé", "facility", "圖書館咖啡廳", "图书馆咖啡厅")
AED = node("AED (Library)", "facility", "(去顫器) 圖書館", "(去颤器) 图书馆")


class CjkNameTests(unittest.TestCase):
    def test_the_library_itself_wins_over_the_cafe_and_the_aed(self) -> None:
        hits = [(0.75, CAFE), (0.70, AED), (0.69, LIBRARY)]
        for query in ("圖書館", "图书馆"):
            named = cjk_named(hits, query)
            self.assertEqual([n.name for _, n in named], [LIBRARY.name], query)

    def test_a_name_ending_in_the_phrase_is_a_match_even_inside_brackets(self) -> None:
        # "L座(包玉剛圖書館)" ends with the phrase only after the bracket is treated as a
        # separator; without that the library would score as a non-match.
        self.assertEqual(cjk_named([(0.69, LIBRARY)], "圖書館"), [(0.69, LIBRARY)])

    def test_a_facility_only_phrase_keeps_its_facilities(self) -> None:
        # Nothing here is a building, so the place-before-facility rule must not empty it.
        water = node("Water Station (Core P)", "facility", "飲水機(P座)", "饮水机(P座)")
        named = cjk_named([(0.9, water)], "飲水機")
        self.assertEqual([n.name for _, n in named], [water.name])

    def test_a_phrase_no_name_ends_with_changes_nothing(self) -> None:
        # The embeddings are the only evidence for a paraphrased Chinese question.
        hits = [(0.8, CAFE), (0.7, LIBRARY)]
        self.assertEqual(cjk_named(hits, "口渴"), hits)


if __name__ == "__main__":
    unittest.main()
