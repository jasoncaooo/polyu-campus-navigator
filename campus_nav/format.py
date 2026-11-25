"""Facility shapes with no sentences in them: which rows, which day, what state.

Split out of `facility_speech.py`, which did three jobs at once -- filter, decide, translate --
in a language that was hard-wired to Chinese. Nothing here branches on language except
through `copy.py`, and nothing here writes prose: it returns the pieces a renderer needs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import copy
from .graph import CampusGraph
from .kb import HKT, clock, parse_slots
from .local_names import building_name, floor_name, has_service_label, place_name, service_label

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# A comfortable campus walking pace: 4.5 km/h. Used to turn measured metres into "about N
# minutes", the number a person actually plans with.
WALK_MPS = 1.25


def walk_minutes(metres: float) -> int:
    """Minutes at a comfortable pace, never less than one."""
    return max(1, round(metres / WALK_MPS / 60))


@dataclass
class FacilityResult:
    kind: str
    subject: str = ""
    total: int = 0
    reference: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    day: str = ""
    route_origin: str = ""


def requested_day(question: str) -> str:
    """The day a question asks about, as a key: "Tue", "PublicHoliday", or "" for today.

    The words that ask it are language data (星期/禮拜/明天/公眾假期), and this is the one
    place that reads them -- the renderers only ever see the key.
    """
    said = question or ""
    if re.search(r"公[眾众]假(?:期|日)|公共假期|public\s+holiday", said, re.I):
        return "PublicHoliday"
    match = re.search(r"(?:星期|禮拜|礼拜|[週周])([一二三四五六日天])", said)
    if match:
        return DAYS["一二三四五六日".index(match.group(1).replace("天", "日"))]
    # English asks the same question with the day's own name, and it used to be read as no
    # day at all: "on Tuesday" was answered with today's hours, which is a wrong answer
    # rather than an imprecise one.
    match = re.search(r"(?<![a-z])(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
                      r"|(?<![a-z])(mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?![a-z])",
                      said, re.I)
    if match:
        word = match.group(0).lower()
        for index, full in enumerate(("monday", "tuesday", "wednesday", "thursday",
                                      "friday", "saturday", "sunday")):
            if full.startswith(word[:3]):
                return DAYS[index]
    if re.search(r"明[天日]|聽日|听日|tomorrow", said, re.I):
        return DAYS[(clock() + timedelta(days=1)).weekday()]
    return ""


def day_label(day: str, language: str) -> str:
    if not day:
        return copy.t("hours.day.today", language)
    if day == "PublicHoliday":
        return copy.t("hours.day.public_holiday", language)
    return copy.t(f"hours.day.{day.lower()}", language)


def local_hours(hours_today: str, language: str) -> str:
    """A published "08:00-23:00" written the way the language writes it."""
    value = hours_today or ""
    if not value or language not in ("zh-HK", "zh-CN"):
        return value
    if value.lower() == "closed":
        return copy.t("hours.closed", language)
    separator = copy.t("hours.list_separator", language)
    ranged = re.sub(r"(\d\d:\d\d)\s*-\s*(\d\d:\d\d)",
                    lambda m: copy.t("hours.range", language, start=m.group(1), end=m.group(2)),
                    value)
    return ranged.replace(", ", separator)


def place_label(card: dict, graph: CampusGraph, language: str) -> str:
    """The place as a person would say it: its name, then where it is."""
    node_id = card.get("node_id")
    name = place_name(graph, node_id, language) if node_id in graph.nodes else ""
    original = card.get("name", "")
    if not name or name == original:
        name = name or original
        if language in ("zh-HK", "zh-CN"):
            # The card carries both scripts; asking for English must not fall into the
            # simplified-Chinese field (it did, and an English answer said 图书馆咖啡厅).
            key = "name_zh" if language == "zh-HK" else "name_zh_cn"
            name = card.get(key) or card.get("name_zh") or name
            # A generic label with a location suffix ("Toilet (Block V)") can be rebuilt from
            # the node's own category, building and floor -- in the language asked for.
            service = card.get("service", "")
            if name == original and service and (" (" in name or " · " in name):
                if has_service_label(service):
                    name = service_label(service, language)
    building = building_name(graph, card.get("building", ""), language) if card.get("building") else ""
    floor = floor_name(card["floor"], language) if card.get("floor") else ""
    location = " ".join(part for part in (building, floor) if part and part not in name)
    return copy.t("place.with_location", language, name=name, location=location) if location else name


def timetable(card: dict, day: str, language: str) -> str:
    """One line of opening hours: the day, then the times (or why there are none)."""
    if not day:
        value = card.get("hours_today") or ""
    else:
        raw = (card.get("opening_hours") or {}).get(day) or []
        closed = any(str(item).upper() == "CLOSED"
                     for item in ([raw] if isinstance(raw, str) else raw))
        value = ", ".join(f"{opens}-{closes}" for opens, closes in parse_slots(raw)) \
            or ("closed" if closed else "")
    label = day_label(day, language)
    if value == "closed":
        return label + copy.t("hours.closed", language)
    if not value:
        if card.get("hours_summary"):
            return copy.t("hours.not_published", language, label=label)
        return copy.t("hours.none", language)
    return label + copy.t("hours.day.joiner", language) + local_hours(value, language)


def rows_of(info: FacilityResult, reply) -> list[dict]:
    """The cards this result is about: its own rows, or whatever the reply carried."""
    return info.rows or getattr(reply, "facilities", None) or []
