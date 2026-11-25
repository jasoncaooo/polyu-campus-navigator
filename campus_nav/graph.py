"""Campus graph: nodes are places, edges are walkable links with access attributes.

The JSON file is the single source of truth for the demo map:

    {
      "meta": {"demo": true, "note": "..."},
      "nodes": [{"id": "VA", "name": "Shaw Amenities Building", "lat": 22.30, "lng": 114.17,
                 "kind": "building"}],
      "edges": [{"id": "E01", "u": "VA", "v": "Z", "metres": 62.0,
                 "covered": true, "open": true}]
    }

Two invariants are enforced at load time, and they are the reason A* is allowed to
use straight-line distance as its heuristic:

  1. every edge length >= the straight-line distance between its endpoints
     -> h(n) never overestimates -> admissible -> A* is optimal
  2. h(goal) == 0

If either fails we raise instead of silently producing a wrong route.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    lat: float
    lng: float
    # The role this node plays in the graph -- NOT how the campus map files it, which
    # is the category field below:
    #   waypoint  a junction of the walkway geometry: never named, never a destination
    #   place     a destination: a block, a main location
    #   facility  somewhere inside a building: a toilet, an office, an AED
    kind: str = "place"
    # The campus map's own category, as published: "Amenities", "Main Locations",
    # "Faculties, Schools & Departments", "Administrative & Supporting Office",
    # "Public Transport", "Buildings / Cores / Wings / Blocks". The legend groups on
    # this, and -- under Amenities -- by the service label.
    category: str = ""
    building: str = ""
    floor: str = ""
    name_zh: str = ""            # the Chinese name in traditional characters
    name_zh_cn: str = ""         # the same name in simplified characters
    service: str = ""            # e.g. "Accessible Toilet"
    accepts: tuple[str, ...] = ()  # canonical capabilities this point accepts
    source: str = ""             # "campus-map" -- provenance of the place
    ref: str = ""                # that source's own record id (poi_id)
    # True when PolyU publishes this place but no drawn walkway reaches it (another
    # campus, a hall across the road). It is drawn on the map and can be looked up, but
    # it has no edges, so no route can end there -- and the reply says why.
    off_network: bool = False
    nearest_m: float = 0.0       # metres to the nearest walkway vertex, when off_network

    def distance_to(self, other: "Node") -> float:
        return haversine_m(self.lat, self.lng, other.lat, other.lng)


@dataclass(frozen=True)
class Edge:
    id: str
    u: str
    v: str
    metres: float
    covered: bool = False
    open: bool = True
    stairs: bool = False         # a flight of steps: drawn as rungs, not a plain line
    # The real shape of the walkway, as (lat, lng) points between u and v. Empty
    # means "we only know the endpoints", and the renderer falls back to a straight
    # line. When it IS present, metros is the length measured along it, so the
    # straight-line heuristic stays admissible (a curve is never shorter).
    path: tuple[tuple[float, float], ...] = ()

    def other(self, node_id: str) -> str:
        return self.v if node_id == self.u else self.u


@dataclass
class GraphViolation:
    edge_id: str
    detail: str


@dataclass
class CampusGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "CampusGraph":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        nodes = {
            n["id"]: Node(
                id=n["id"],
                name=n.get("name", n["id"]),
                lat=float(n["lat"]),
                lng=float(n["lng"]),
                kind=n.get("kind", "place"),
                category=n.get("category", ""),
                building=n.get("building", ""),
                floor=n.get("floor", ""),
                    name_zh=str(n.get("name_zh", "")),
                name_zh_cn=str(n.get("name_zh_cn", "")),
                service=n.get("service", ""),
                accepts=tuple(n.get("accepts", ())),
                source=n.get("source", ""),
                ref=str(n.get("ref", "")),
                off_network=bool(n.get("off_network")),
                nearest_m=float(n.get("nearest_m") or 0.0),
            )
            for n in raw["nodes"]
        }
        edges = [
            Edge(
                id=e["id"],
                u=e["u"],
                v=e["v"],
                metres=float(e["metres"]),
                covered=bool(e.get("covered", False)),
                open=bool(e.get("open", True)),
                stairs=bool(e.get("stairs", False)),
                path=tuple(
                    (float(point[0]), float(point[1])) for point in e.get("path", ())
                ),
            )
            for e in raw["edges"]
        ]
        graph = cls(nodes=nodes, edges=edges, meta=raw.get("meta", {}))
        graph.validate()
        return graph

    # -- validation --------------------------------------------------------
    def validate(self, tolerance_m: float = 0.5) -> list[GraphViolation]:
        """Check referential integrity and the admissibility invariants."""
        problems: list[GraphViolation] = []
        seen: set[str] = set()
        for e in self.edges:
            if e.id in seen:
                problems.append(GraphViolation(e.id, "duplicate edge id"))
            seen.add(e.id)
            for endpoint in (e.u, e.v):
                if endpoint not in self.nodes:
                    problems.append(GraphViolation(e.id, f"unknown node {endpoint!r}"))
            if e.u not in self.nodes or e.v not in self.nodes:
                continue
            straight = self.nodes[e.u].distance_to(self.nodes[e.v])
            if e.metres + tolerance_m < straight:
                problems.append(
                    GraphViolation(
                        e.id,
                        f"length {e.metres:.1f} m < straight-line {straight:.1f} m "
                        "-> heuristic would overestimate",
                    )
                )
            if e.metres <= 0:
                problems.append(GraphViolation(e.id, "non-positive length"))
            if len(e.path) >= 2:
                walked = sum(
                    haversine_m(a[0], a[1], b[0], b[1])
                    for a, b in zip(e.path, e.path[1:])
                )
                if e.metres + tolerance_m < walked:
                    problems.append(
                        GraphViolation(
                            e.id,
                            f"length {e.metres:.1f} m < its own walkway geometry "
                            f"{walked:.1f} m",
                        )
                    )
        if problems:
            detail = "\n".join(f"  [{p.edge_id}] {p.detail}" for p in problems)
            raise ValueError(f"graph failed validation ({len(problems)} problems):\n{detail}")
        return problems

    # -- queries -----------------------------------------------------------
    def node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def heuristic(self, node_id: str, goal_id: str) -> float:
        """Straight-line metres. Admissible by construction (see validate)."""
        return self.nodes[node_id].distance_to(self.nodes[goal_id])

    def usable_edges(
        self,
        closed_edges: set[str] | None = None,
        closed_nodes: set[str] | None = None,
        *, avoid_stairs: bool = False,
    ) -> list[Edge]:
        """Exclude closures and, when requested, all links marked as stairs."""
        closed_edges = closed_edges or set()
        closed_nodes = closed_nodes or set()
        out = []
        for e in self.edges:
            if not e.open or e.id in closed_edges or (avoid_stairs and e.stairs):
                continue
            if (e.u in closed_nodes) or (e.v in closed_nodes):
                continue
            out.append(e)
        return out

    def adjacency(
        self,
        closed_edges: set[str] | None = None,
        closed_nodes: set[str] | None = None,
        *, avoid_stairs: bool = False,
    ) -> dict[str, list[tuple[str, float, Edge]]]:
        adj: dict[str, list[tuple[str, float, Edge]]] = {n: [] for n in self.nodes}
        for e in self.usable_edges(closed_edges, closed_nodes, avoid_stairs=avoid_stairs):
            adj[e.u].append((e.v, e.metres, e))
            adj[e.v].append((e.u, e.metres, e))
        return adj

    def summary(self) -> str:
        closed = sum(1 for e in self.edges if not e.open)
        return (
            f"{len(self.nodes)} nodes, {len(self.edges)} edges"
            + (f", {closed} closed" if closed else "")
        )
