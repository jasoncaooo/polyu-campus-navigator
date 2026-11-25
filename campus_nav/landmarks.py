"""Landmarks along a walk: the blocks a route passes, in the order you meet them.

The route line is drawn from the walkway network, which carries no names -- a waypoint is
a bend, not a place. The buildings are what a person actually recognises ("walk past
Block M, then core P"), so this module answers one question about a finished route: which
published buildings/wings/cores does the walked line pass close to, and at what metre
mark?

Two deliberate choices:

  * "Passes" is geometry, not topology. The walkway does not run through the buildings
    (they are pictures, not corridors), so proximity is the only honest test. A building
    whose footprint edge is within `within_m` of the line counts.
  * The endpoints are left out. The reply already names the start and the goal; repeating
    them as "landmarks" would be noise.
  * Only the campus map's "Buildings / Cores / Wings / Blocks" count. Not the squares and
    galleries it files under "Main Locations", and nothing inside a building: the point is
    the short list of blocks a person looks up and sees.

Nothing here is invented: a landmark is a node from the graph with its own published name
and coordinates, and its metre mark comes from the walked geometry the engine measured.
"""
from __future__ import annotations

import math

from .graph import CampusGraph, Node

# A walkway is a few metres wide and a podium walkway sits under a building edge, so 20 m
# is "you are walking past it", not "it is vaguely nearby".
NEAR_M = 20.0

# What fits beside a route without turning the map into a list.
MAX_LANDMARKS = 6

# The first and last few metres belong to the start and the goal: the building you are
# standing at is not something you "pass", and naming it twice crowds out a real landmark.
EDGE_M = 12.0

# The only category that counts as a landmark: the blocks themselves. A toilet inside a
# building is not a landmark, the building is -- and neither is a square, a gallery, a
# swimming pool or a sports centre, which is what "Main Locations" is made of. That
# category used to be included and it put "Suen Chi Sun Memorial Square" and "Block X
# Sports Centre" on the route line beside the buildings, so the walk read as a list of
# everything it passed rather than the blocks somebody navigates by.
GUIDE_CATEGORY = "Buildings / Cores / Wings / Blocks"

_METRES_PER_DEG_LAT = 111_320.0


def _point_segment(px: float, py: float, ax: float, ay: float,
                   bx: float, by: float) -> tuple[float, float]:
    """Distance from a point to a segment, and how far along it the closest point is."""
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 0.0:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / length2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def _walked_points(graph: CampusGraph, route) -> list[tuple[float, float]]:
    """The walked line as [lat, lng] points, taken from the measured segment geometry."""
    points: list[tuple[float, float]] = []
    for segment in route.segments:
        if segment.path and len(segment.path) >= 2:
            points.extend(segment.path)
        else:
            points.append((graph.node(segment.u).lat, graph.node(segment.u).lng))
            points.append((graph.node(segment.v).lat, graph.node(segment.v).lng))
    if not points:
        points = [(graph.node(node_id).lat, graph.node(node_id).lng)
                  for node_id in route.path]
    return points


def landmarks_along(graph: CampusGraph, route, *, within_m: float = NEAR_M,
                    limit: int = MAX_LANDMARKS) -> list[dict]:
    """Buildings the walk passes, in order, with the metre mark where you meet them."""
    if route is None or not getattr(route, "reachable", False):
        return []
    points = _walked_points(graph, route)
    if len(points) < 2:
        return []

    # A local metric plane is accurate enough over one campus and keeps the arithmetic
    # readable: metres east per degree of longitude shrinks with latitude.
    centre_lat = sum(lat for lat, _ in points) / len(points)
    metres_per_deg_lng = _METRES_PER_DEG_LAT * math.cos(math.radians(centre_lat))

    def to_xy(lat: float, lng: float) -> tuple[float, float]:
        return lng * metres_per_deg_lng, lat * _METRES_PER_DEG_LAT

    plane = [to_xy(lat, lng) for lat, lng in points]
    walked = [0.0]
    for index in range(1, len(plane)):
        walked.append(walked[-1] + math.hypot(
            plane[index][0] - plane[index - 1][0],
            plane[index][1] - plane[index - 1][1],
        ))

    ends = {route.path[0], route.path[-1]} if route.path else set()
    found: list[tuple[float, float, Node]] = []
    for node in graph.nodes.values():
        if node.id in ends or node.kind == "waypoint":
            continue
        if (node.category or "") != GUIDE_CATEGORY:
            continue
        px, py = to_xy(node.lat, node.lng)
        best_distance = within_m + 1.0
        best_mark = 0.0
        for index in range(1, len(plane)):
            distance, along = _point_segment(
                px, py, plane[index - 1][0], plane[index - 1][1],
                plane[index][0], plane[index][1],
            )
            if distance < best_distance:
                best_distance = distance
                best_mark = walked[index - 1] + along * (walked[index] - walked[index - 1])
        if best_distance > within_m:
            continue
        if best_mark < EDGE_M or best_mark > walked[-1] - EDGE_M:
            continue          # the start and the goal already have their own labels
        found.append((best_mark, best_distance, node))

    found.sort(key=lambda item: item[0])
    chosen: list[dict] = []
    for mark, distance, node in found:
        if len(chosen) >= limit:
            break
        chosen.append({
            "id": node.id,
            "name": node.name,
            "at_m": round(mark),
            "distance_m": round(distance, 1),
            "lat": node.lat,
            "lng": node.lng,
        })
    return chosen

