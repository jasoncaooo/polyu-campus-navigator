"""Replay a fixed set of prompts and diff the *facts* of the answers.

What this is for
----------------
Refactoring the dialogue layer means moving sentences and branching around, and "it looks
fine" is not evidence. This tool asks the app the same questions twice -- before and after --
and reports any prompt whose answer changed in a way a person would notice: the intent, the
route, the metres, the places offered, the words spoken.

It runs offline (grammar parser, built-in name scorer, no network, no keys) because a golden
diff needs determinism, not model prose. Pass --live to use the configured model instead, and
read the differences with that in mind: the model phrases sentences differently every run.

The clock is pinned (see campus_nav.kb.clock): answers say "open now" or "closed now", and a
canteen crossing 10:30 between two runs once produced a diff that no code change had caused.

Usage
-----
    python tools/replay.py                       # print what the app answers today
    python tools/replay.py --save before.json    # record it
    ...refactor...
    python tools/replay.py --diff before.json    # what changed? (timings ignored)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 10:00 on a Wednesday: mid-morning, mid-week, nothing opening or closing across the run.
# --live keeps the real clock, because a live answer is about now by definition.
if "--live" not in sys.argv:
    os.environ.setdefault("CAMPUS_NAV_NOW", "2025-06-11T10:00:00")

from campus_nav.config import Settings, get_settings, graph_path   # noqa: E402
from campus_nav.dialogue import Dialogue                          # noqa: E402
from campus_nav.graph import CampusGraph                          # noqa: E402
from campus_nav.narrate import Narrator                           # noqa: E402
from campus_nav.llm import DeepSeekClient                         # noqa: E402
from campus_nav.place_index import PlaceIndex                     # noqa: E402

# A timing is not behaviour: "expanded 51 nodes, 1.03 ms" and "... 0.98 ms" are the same
# answer. Everything else in the receipt is compared exactly.
TIMING = re.compile(r"\d+(?:\.\d+)? ms")

# One case per behaviour the app claims, in all three languages. Multi-turn cases are lists
# of messages, because context is now part of the contract.
CASES: list[dict] = [
    {"lang": "en-US", "turns": ["how do I get from Block V to the library?"]},
    {"lang": "en-US", "turns": ["how do I get from Block Y to Core A?"]},
    {"lang": "en-US", "turns": ["how far is it from Core B to Core E?"]},
    {"lang": "en-US", "turns": ["from Core D to Block Z"]},
    {"lang": "en-US", "turns": ["how do I get to the toilet in Block M?"]},
    {"lang": "en-US", "turns": ["from Core A to UMF - Atomic Electron Microscopy Laboratory (AEML)"]},
    {"lang": "en-US", "turns": ["where is the nearest AED?"]},
    {"lang": "en-US", "turns": ["I want to grab lunch, where can I go?"]},
    {"lang": "en-US", "turns": ["is the canteen open?"]},
    {"lang": "en-US", "turns": ["what time does the bookshop close?"]},
    {"lang": "en-US", "turns": ["what facilities are there"]},
    {"lang": "en-US", "turns": ["I want to have a swim"]},
    {"lang": "en-US", "turns": ["the library is closed"]},
    {"lang": "en-US", "turns": ["re-open everything"]},
    {"lang": "en-US", "turns": ["no stairs, from Block V to the library"]},
    {"lang": "en-US", "turns": ["from Block Y to Core A via Core P"]},
    {"lang": "en-US", "turns": ["from Core Y to Core A"]},
    {"lang": "en-US", "turns": ["hi"]},
    {"lang": "en-US", "turns": ["how do I get from Block V to the library?",
                               "how far is it?",
                               "no stops, the direct route"]},
    {"lang": "en-US", "turns": ["I want to walk from block L to block Y.",
                               "I also want a quick breakfast along the way."]},
    {"lang": "zh-HK", "turns": ["由圖書館去賽馬會創新樓點行？"]},
    {"lang": "zh-HK", "turns": ["最近的飲水機喺邊？"]},
    {"lang": "zh-HK", "turns": ["飯堂幾點開門？"]},
    {"lang": "zh-CN", "turns": ["从图书馆到赛马会创新楼"]},
    {"lang": "zh-CN", "turns": ["最近的厕所在哪里？"]},
]


def build(live: bool) -> tuple[CampusGraph, PlaceIndex, DeepSeekClient | None]:
    graph = CampusGraph.load(graph_path())
    settings = get_settings()
    if live:
        index = PlaceIndex(graph, settings)
        index.check()
        client = DeepSeekClient(settings)
        return graph, index, client
    offline = Settings(api_key="", model="test", llm_enabled=False, timeout_s=1)
    return graph, PlaceIndex(graph, offline), None


def record(graph: CampusGraph, index: PlaceIndex, client: DeepSeekClient | None,
           live: bool) -> dict:
    narrator = Narrator(client) if client is not None else None
    out: dict = {}
    for case in CASES:
        talk = Dialogue(graph, client=client, narrator=narrator, places=index)
        turns = []
        for message in case["turns"]:
            # The request's language, exactly as the browser sends it. Without it the engine
            # can only guess from the script, and a zh-CN case was answered with zh-HK wording.
            reply = talk.handle(message, language=case["lang"])
            route = reply.route
            turns.append({
                "ask": message,
                "intent": getattr(getattr(reply, "slots", None), "intent", ""),
                "parser": getattr(getattr(reply, "slots", None), "source", ""),
                "text": TIMING.sub("", reply.text or ""),
                "spoken": reply.spoken or "",
                "metres": round(route.total_metres, 1) if route else None,
                "legs": len(route.path) if route else None,
                "reachable": route.reachable if route else None,
                "cards": [card.get("name") for card in (reply.facilities or [])],
                "options": len(reply.routes or []),
            })
        out[" / ".join(case["turns"]) + f"  [{case['lang']}]"] = turns
    return out


def diff(before: dict, after: dict) -> int:
    changed = 0
    for key in before:
        if key not in after:
            print(f"MISSING  {key}")
            changed += 1
            continue
        for index, (old, new) in enumerate(zip(before[key], after[key])):
            for field in old:
                if old[field] != new.get(field):
                    changed += 1
                    print(f"CHANGED  {key}  turn {index + 1}  field {field}")
                    print(f"    before: {json.dumps(old[field], ensure_ascii=False)[:180]}")
                    print(f"    after : {json.dumps(new.get(field), ensure_ascii=False)[:180]}")
    for key in after:
        if key not in before:
            print(f"NEW      {key}")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay prompts and diff the answers")
    parser.add_argument("--live", action="store_true", help="use the configured model")
    parser.add_argument("--save", type=Path, help="write the recording to this file")
    parser.add_argument("--diff", type=Path, help="compare against a recording")
    args = parser.parse_args()

    graph, index, client = build(args.live)
    print(f"mode: {'live (model)' if args.live else 'offline (deterministic)'}; "
          f"{len(CASES)} cases")
    after = record(graph, index, client, args.live)

    if args.diff:
        before = json.loads(args.diff.read_text(encoding="utf-8"))
        changed = diff(before, after)
        print()
        print(f"{changed} difference(s)" if changed else "no differences: behaviour unchanged")
        return 1 if changed else 0

    if args.save:
        args.save.write_text(json.dumps(after, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"recorded {len(after)} cases -> {args.save}")
        return 0

    for key, turns in after.items():
        print(f"\n{key}")
        for turn in turns:
            print(f"    {turn['intent'] or '-':14s} {turn['metres'] or '-':>7} m  "
                  f"{(turn['spoken'] or turn['text']).splitlines()[0][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
