"""Landmarks along a walk: the ordering, the distance bound, and the exclusions.

The route line runs on the walkway network, which has no names, so these are the only
thing a person can hold on to while walking. The tests pin the three properties that make
that honest: a landmark is passed in the order the walk meets it, it is genuinely close
to the line, and it is neither an endpoint nor a waypoint dressed up as a place.
"""
from __future__ import annotations

import unittest

from campus_nav.config import graph_path
from campus_nav.graph import CampusGraph
from campus_nav.landmarks import GUIDE_CATEGORY, MAX_LANDMARKS, NEAR_M, landmarks_along
from campus_nav.search import search


class LandmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.start = cls._one("Block V (Jockey Club Innovation Tower)")
        cls.goal = cls._one("Wing AG (Chung Sze Yuen Building)")
        cls.route = search(cls.graph, cls.start, cls.goal, algorithm="astar")

    @classmethod
    def _one(cls, name: str) -> str:
        for node in cls.graph.nodes.values():
            if node.name == name:
                return node.id
        raise AssertionError(f"no node named {name!r} in the graph")

    def test_the_route_itself_is_reachable(self) -> None:
        self.assertTrue(self.route.reachable, "fixture route must exist for these tests")

    def test_landmarks_are_passed_in_order(self) -> None:
        marks = landmarks_along(self.graph, self.route)
        self.assertTrue(marks, "a 275 m walk across the campus passes something")
        marks_at = [mark["at_m"] for mark in marks]
        self.assertEqual(marks_at, sorted(marks_at))

    def test_every_landmark_is_genuinely_close_to_the_line(self) -> None:
        for mark in landmarks_along(self.graph, self.route):
            self.assertLessEqual(mark["distance_m"], NEAR_M, mark["name"])

    def test_the_landmark_is_a_published_place_not_a_waypoint(self) -> None:
        for mark in landmarks_along(self.graph, self.route):
            node = self.graph.node(mark["id"])
            self.assertNotEqual(node.kind, "waypoint", node.name)
            self.assertEqual(node.category, GUIDE_CATEGORY, node.name)

    def test_only_buildings_are_landmarks_never_squares_or_galleries(self) -> None:
        # "Main Locations" is squares, galleries, pools and sports centres. Including it
        # put "Suen Chi Sun Memorial Square" and "Block X Sports Centre" beside the blocks
        # the walk passes, so the annotation read as everything the street offered.
        for mark in landmarks_along(self.graph, self.route):
            self.assertEqual(self.graph.node(mark["id"]).category,
                             "Buildings / Cores / Wings / Blocks", mark["name"])
        crossed = {
            node.name
            for node in self.graph.nodes.values()
            if node.category == "Main Locations"
        }
        listed = {mark["name"] for mark in landmarks_along(self.graph, self.route)}
        self.assertEqual(crossed & listed, set())

    def test_the_endpoints_are_not_listed_as_landmarks(self) -> None:
        ids = {mark["id"] for mark in landmarks_along(self.graph, self.route)}
        self.assertNotIn(self.start, ids)
        self.assertNotIn(self.goal, ids)

    def test_the_list_is_capped(self) -> None:
        marks = landmarks_along(self.graph, self.route, limit=2)
        self.assertLessEqual(len(marks), 2)
        self.assertLessEqual(len(landmarks_along(self.graph, self.route)), MAX_LANDMARKS)

    def test_a_failed_route_has_no_landmarks(self) -> None:
        broken = search(self.graph, self.start, self.goal, algorithm="astar",
                        closed_nodes={self.goal})
        self.assertFalse(broken.reachable)
        self.assertEqual(landmarks_along(self.graph, broken), [])


if __name__ == "__main__":
    unittest.main()
