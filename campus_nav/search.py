"""A* and Uniform-Cost Search over the campus graph.

Both algorithms share one code path so the comparison is fair: A* is UCS plus a
heuristic term. Node counting is defined once -- **a node counts as expanded when
it is popped for the first time** -- and the same counter is reported for both,
which is what makes the demo table defensible.

Reference point for the theory (Lecture 2, slides 11-22):
    f(n) = g(n) + h(n);  h admissible (h(n) <= h*(n)) -> A* is optimal;
    h consistent (h(n) <= c(n,n') + h(n')) -> no node is re-expanded.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

from .graph import CampusGraph, Edge


@dataclass
class Stats:
    algorithm: str
    nodes_expanded: int = 0
    nodes_generated: int = 0
    frontier_peak: int = 0
    elapsed_ms: float = 0.0
    heuristic: str = "euclidean"


@dataclass
class Segment:
    edge_id: str
    u: str
    v: str
    metres: float
    path: tuple[tuple[float, float], ...] = ()
    stairs: bool = False          # this leg is a flight of steps
    covered: bool = False         # this leg is under cover


@dataclass
class RouteResult:
    reachable: bool
    algorithm: str
    path: list[str] = field(default_factory=list)
    total_metres: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    stats: Stats | None = None
    explain: list[dict] = field(default_factory=list)
    blocking: dict = field(default_factory=dict)


def _reconstruct(parent: dict[str, tuple[str, Edge]], start: str, goal: str) -> list[tuple[str, Edge]]:
    steps: list[tuple[str, Edge]] = []
    cur = goal
    while cur != start:
        prev, edge = parent[cur]
        steps.append((cur, edge))
        cur = prev
    steps.reverse()
    return steps


def search(
    graph: CampusGraph,
    start: str,
    goal: str,
    *,
    algorithm: str = "astar",
    closed_edges: set[str] | None = None,
    closed_nodes: set[str] | None = None,
    avoid_stairs: bool = False,
) -> RouteResult:
    """Run A* or UCS. `algorithm` is "astar" or "ucs"."""
    if start not in graph.nodes:
        raise KeyError(f"unknown start node {start!r}")
    if goal not in graph.nodes:
        raise KeyError(f"unknown goal node {goal!r}")

    use_heuristic = algorithm == "astar"
    adj = graph.adjacency(closed_edges, closed_nodes, avoid_stairs=avoid_stairs)
    blocked = {
        "closed_edges": sorted(closed_edges or ()),
        "closed_nodes": sorted(closed_nodes or ()),
    }

    active = {"avoid_stairs": True} if avoid_stairs else {}
    blocked.update(active)
    started = time.perf_counter()
    stats = Stats(algorithm=algorithm, heuristic="euclidean" if use_heuristic else "zero")

    if start == goal:
        stats.elapsed_ms = (time.perf_counter() - started) * 1000
        return RouteResult(True, algorithm, [start], 0.0, [], stats, [], active)

    g_cost: dict[str, float] = {start: 0.0}
    parent: dict[str, tuple[str, Edge]] = {}
    closed: set[str] = set()
    explain: list[dict] = []

    h0 = graph.heuristic(start, goal) if use_heuristic else 0.0
    frontier: list[tuple[float, int, str, float]] = [(h0, 0, start, 0.0)]
    tie = 1

    while frontier:
        stats.frontier_peak = max(stats.frontier_peak, len(frontier))
        f_val, _, node, g_val = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        stats.nodes_expanded += 1

        h_val = graph.heuristic(node, goal) if use_heuristic else 0.0
        explain.append(
            {
                "node": node,
                "name": graph.node(node).name,
                "g": round(g_val, 1),
                "h": round(h_val, 1) if use_heuristic else None,
                "f": round(g_val + h_val, 1) if use_heuristic else round(g_val, 1),
            }
        )

        if node == goal:
            steps = _reconstruct(parent, start, goal)
            segments, total = [], 0.0
            cur = start
            for nxt, edge in steps:
                segments.append(
                    Segment(
                        edge_id=edge.id,
                        u=cur,
                        v=nxt,
                        metres=edge.metres,
                        path=edge.path if cur == edge.u else edge.path[::-1],
                        stairs=edge.stairs,
                        covered=edge.covered,
                    )
                )
                total += edge.metres
                cur = nxt
            stats.elapsed_ms = (time.perf_counter() - started) * 1000
            return RouteResult(True, algorithm, [start] + [n for n, _ in steps], total, segments, stats, explain, active)

        for neighbour, weight, edge in adj.get(node, ()):
            if neighbour in closed:
                continue
            candidate = g_val + weight
            if candidate + 1e-9 < g_cost.get(neighbour, float("inf")):
                g_cost[neighbour] = candidate
                parent[neighbour] = (node, edge)
                hn = graph.heuristic(neighbour, goal) if use_heuristic else 0.0
                tie += 1
                heapq.heappush(frontier, (candidate + hn, tie, neighbour, candidate))
                stats.nodes_generated += 1

    stats.elapsed_ms = (time.perf_counter() - started) * 1000
    return RouteResult(False, algorithm, [], 0.0, [], stats, explain, blocked)


def astar(graph: CampusGraph, start: str, goal: str, **kwargs) -> RouteResult:
    return search(graph, start, goal, algorithm="astar", **kwargs)


def ucs(graph: CampusGraph, start: str, goal: str, **kwargs) -> RouteResult:
    return search(graph, start, goal, algorithm="ucs", **kwargs)


def compare(graph: CampusGraph, start: str, goal: str, **kwargs) -> dict:
    """A* vs UCS on identical inputs -- the numbers for the demo slide."""
    a = astar(graph, start, goal, **kwargs)
    u = ucs(graph, start, goal, **kwargs)
    return {
        "start": start,
        "goal": goal,
        "constraints": {
            "closed_edges": sorted(kwargs.get("closed_edges") or ()),
            "closed_nodes": sorted(kwargs.get("closed_nodes") or ()),
            "avoid_stairs": bool(kwargs.get("avoid_stairs")),
        },
        "astar": {
            "reachable": a.reachable,
            "metres": round(a.total_metres, 1),
            "expanded": a.stats.nodes_expanded if a.stats else 0,
            "generated": a.stats.nodes_generated if a.stats else 0,
            "ms": round(a.stats.elapsed_ms, 3) if a.stats else 0.0,
            "path": a.path,
        },
        "ucs": {
            "reachable": u.reachable,
            "metres": round(u.total_metres, 1),
            "expanded": u.stats.nodes_expanded if u.stats else 0,
            "generated": u.stats.nodes_generated if u.stats else 0,
            "ms": round(u.stats.elapsed_ms, 3) if u.stats else 0.0,
            "path": u.path,
        },
        "same_cost": a.reachable == u.reachable and abs(a.total_metres - u.total_metres) < 0.5,
        "saving": (
            (u.stats.nodes_expanded - a.stats.nodes_expanded)
            if a.stats and u.stats
            else 0
        ),
    }
