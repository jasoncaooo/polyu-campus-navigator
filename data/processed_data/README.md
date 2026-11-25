# processed_data

One file: `campus_graph.json`. The app reads it and nothing else.

It is **generated**, not edited. Change one of the inputs in `../authored/` and run:

```
python tools/build.py graph
```

Hand-editing this file works until the next build overwrites it, and it cannot fix
a drawing that is wrong -- the next build will simply reproduce your mistake.

## What is in it

| Node kind | Count | Drawn on the map? |
|---|---|---|
| `place` (building, entrance, service) | 191 | yes |
| `waypoint` | 327 | no -- the end of a link, not a stop |

Each edge carries `metres` (measured along the walkway), `path` (the polyline),
`stairs` and `covered`. The `meta` block records where every number came from.

## The invariant that matters

```
every edge length >= the straight line between its two endpoints
```

That is what makes straight-line distance an admissible heuristic, and therefore
what makes A* return an optimal route. `CampusGraph.validate()` raises on load if it
is ever violated, so a graph that breaks it cannot be served -- which is the point.

`docs/PIPELINE.md` describes the whole route from an Overpass download to this file.
