"""Places PolyU publishes that no drawn walkway reaches.

These are real campus locations -- halls across the road, the West Kowloon campus, a
school on another site. They are in the graph so the map shows them and a lookup finds
them, but they have no edges, so a route must fail -- and the reply has to say *why*
rather than "I cannot find a route", which reads like a bug.
"""
from __future__ import annotations

import unittest

from campus_nav.config import graph_path
from campus_nav.graph import CampusGraph
from campus_nav.search import search

# A place beyond this distance from every walkway vertex cannot be "attached" to it.
ATTACH_M = 120.0


class OffNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = CampusGraph.load(graph_path())
        cls.off = [n for n in cls.graph.nodes.values() if n.off_network]

    def test_the_published_off_site_places_are_in_the_graph(self) -> None:
        names = {n.name for n in self.off}
        self.assertTrue(self.off, "the build should keep them, not drop them")
        self.assertTrue(any("Homantin" in name for name in names), sorted(names)[:5])
        self.assertTrue(any("West Kowloon" in name for name in names), sorted(names)[:5])

    def test_the_other_campuses_are_hundreds_of_metres_away(self) -> None:
        # The halls and the West Kowloon campus are genuinely off this map's network.
        far = [node for node in self.off if node.nearest_m > 400]
        self.assertGreaterEqual(len(far), 30, [n.name for n in self.off])

    def test_a_place_that_merely_misses_a_vertex_is_kept_too(self) -> None:
        # The build joins on an exact vertex coordinate ("a line has no vertex 4 cm from
        # the place"), so a bus stop a few metres off a path is off-network as well. It
        # used to be dropped silently; now it is drawn and the reply says why.
        near = [node for node in self.off if node.nearest_m < ATTACH_M]
        self.assertTrue(near, "expected some places that miss a vertex by a few metres")

    def test_they_have_no_edges(self) -> None:
        connected = {edge.u for edge in self.graph.edges} | {edge.v for edge in self.graph.edges}
        for node in self.off:
            self.assertNotIn(node.id, connected, node.name)

    def test_a_route_to_one_is_unreachable(self) -> None:
        target = self.off[0]
        somewhere = next(n for n in self.graph.nodes.values()
                         if n.kind == "place" and not n.off_network)
        route = search(self.graph, somewhere.id, target.id, algorithm="astar")
        self.assertFalse(route.reachable)
        self.assertEqual(route.path, [])

    def test_the_attached_graph_is_unchanged_by_keeping_them(self) -> None:
        # Keeping the strays must not have attached anything by an invented walkway: an
        # edge into a place is never longer than the attachment limit.
        attached = {edge.v for edge in self.graph.edges if edge.v in self.graph.nodes}
        for edge in self.graph.edges:
            if self.graph.node(edge.v).kind in ("place", "facility"):
                self.assertLessEqual(edge.metres, ATTACH_M + 1, edge.id)


if __name__ == "__main__":
    unittest.main()
