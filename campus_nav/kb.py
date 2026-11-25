"""The knowledge base: every fact the bot is allowed to state, and where it came from.

What problem this solves
------------------------
The route planner answers "how do I get to X" from the graph. It cannot answer
"is the canteen open now" or "which floor is the clinic on", and a language model
asked those questions unaided will happily invent an answer. So the facts live
here, in documents built from the published campus-map records, and the reply is assembled
from the documents that were actually retrieved.

The pipeline, in order (this is the "RAG" part, and it is worth drawing on a slide):

    question
      -> campus_nav/nlu.py      which places and which intent does this sentence have?
      -> campus_nav/kb.py       retrieve the documents that mention those places
      -> campus_nav/graph.py    if a route is wanted, plan it over real coordinates
      -> campus_nav/narrate.py  phrase the answer, refusing numbers nobody computed

Three rules this module enforces, because they are the difference between a useful
domain bot and a confident liar:

  1. No document, no claim. If retrieval returns nothing above the threshold,
     KnowledgeBase.search() returns an empty list and the dialogue layer says it
     does not know. It never falls back to "probably".
  2. Opening hours are quoted, not interpreted. "07:30 - 20:30" is printed as the
     source printed it; only the *open or closed right now* verdict is computed,
     and only when the day is actually covered by the published table.
  3. Silence is a fact too. 33 of the 561 campus POIs publish opening hours. For
     the other 528 the honest answer is "the campus map does not publish opening
     hours for this place" -- not a guess.

Retrieval is BM25 (the ranking function from the IR literature, implemented in
about 40 lines of standard library). It is lexical, not a neural embedding index,
and that is a defensible choice for a closed vocabulary: the source is 561 short
documents, and BM25 never needs a model download, a vector store or a
network call, so the demo works with the Wi-Fi off.
"""
from __future__ import annotations

import json
import os
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

HKT = timezone(timedelta(hours=8))

# Dependency-free tokenisation: ASCII words, CJK characters and CJK bigrams. The
# bigrams are what let a two-character Chinese query match a longer document.
_TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]")
_CJK = re.compile(r"[\u3400-\u9fff]")

_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_DAY_NAMES = {
    "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
    "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
    "publicholiday": "public holidays",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, plus CJK single characters and bigrams."""
    if not text:
        return []
    tokens = _TOKEN.findall(text.lower())
    run: list[str] = []
    for tok in tokens:
        if _CJK.match(tok):
            run.append(tok)
        else:
            if len(run) > 1:
                tokens.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
            run = []
    if len(run) > 1:
        tokens.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
    return tokens


# --------------------------------------------------------------------- documents
@dataclass
class Doc:
    """One retrievable unit: a place, or a green facility."""

    id: str
    kind: str                    # "place"
    title: str
    text: str                    # the searchable surface
    source: str                  # "campus-map"
    ref: str                     # the source record id
    fields: dict = field(default_factory=dict)

    @property
    def hours(self) -> dict:
        return self.fields.get("opening_hours") or {}


# ----------------------------------------------------------------- opening hours
def clock(when: datetime | None = None) -> datetime:
    """The time to judge "open now" by: the argument, else the wall clock, else the pin.

    CAMPUS_NAV_NOW exists because the alternative was invisible: the replay net compared
    answers recorded at one time of day with answers produced at another, so a canteen
    crossing 10:30 turned "closed now" into "open now" and the net reported a difference
    that no code change had caused. One seam, and the net is deterministic.
    """
    if when is not None:
        return when.astimezone(HKT)
    pinned = os.environ.get("CAMPUS_NAV_NOW", "").strip()
    if pinned:
        try:
            return datetime.fromisoformat(pinned).replace(tzinfo=HKT)
        except ValueError:
            pass
    return datetime.now(HKT)


def parse_slots(values) -> list[tuple[str, str]]:
    """["07:30 - 20:30"] -> [("07:30", "20:30")]; "CLOSED" -> [].

    The source mixes three notations -- "07:00 - 23:00", "11.00 - 22.30" and the
    clinics' "0900 - 1300" -- so the separator is optional and the hour needs no
    leading zero. A malformed slot is dropped rather than guessed at.
    """
    if isinstance(values, str):
        values = [values]
    out: list[tuple[str, str]] = []
    for raw in values or []:
        text = str(raw).strip().upper()
        if not text or "CLOSED" in text:
            continue
        match = re.match(
            r"^(\d{1,2})[:.]?(\d{2})\s*(?:-|TO|~)\s*(\d{1,2})[:.]?(\d{2})$", text
        )
        if not match:
            continue
        oh, om, ch, cm = match.groups()
        out.append((f"{int(oh):02d}:{om}", f"{int(ch):02d}:{cm}"))
    return out


def _minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


def hours_summary(raw: dict) -> str:
    """A readable summary, quoting the source's own figures."""
    if not raw:
        return ""
    lines = []
    for day in _DAYS + ("PublicHoliday",):
        slots = parse_slots(raw.get(day))
        flag = raw.get(day) or []
        if not slots:
            special = str(flag[0]).strip() if flag else ""
            if special.upper() == "CLOSED":
                value = "closed"
            else:
                continue
        else:
            value = ", ".join(f"{a}-{b}" for a, b in slots)
        lines.append(f"{_DAY_NAMES.get(day.lower(), day)} {value}")
    remark = (raw.get("Remark") or "").strip()
    if remark:
        lines.append(f"Remark: {remark}")
    return "; ".join(lines)


def open_state(raw: dict, when: datetime | None = None) -> tuple[bool | None, str]:
    """(open now?, one-line explanation) for a published opening-hours table.

    Returns (None, reason) whenever the answer would be a guess: no table, an
    unparsable slot, or a day the table does not cover.
    """
    if not raw:
        return None, "no opening hours are published for this place"
    now = clock(when)
    day = _DAYS[now.weekday()]
    slots = parse_slots(raw.get(day))
    if not slots:
        flag = raw.get(day) or []
        if flag and "CLOSED" in str(flag[0]).upper():
            return False, f"it is closed on {_DAY_NAMES.get(day.lower(), day)} according to the source"
        return None, f"the published table has no usable hours for {_DAY_NAMES.get(day.lower(), day)}"
    minute = now.hour * 60 + now.minute
    for start, end in slots:
        if _minutes(start) <= minute <= _minutes(end):
            return True, f"open until {end} ({day})"
    return False, f"closed at {now:%H:%M}; it opens at {slots[0][0]} on {day}"


def today_hours(raw: dict, when: datetime | None = None) -> str:
    """'07:30-20:30' for today, 'closed', or '' when the table does not say."""
    if not raw:
        return ""
    now = clock(when)
    day = _DAYS[now.weekday()]
    slots = parse_slots(raw.get(day))
    if slots:
        return ", ".join(f"{a}-{b}" for a, b in slots)
    flag = raw.get(day) or []
    if flag and "CLOSED" in str(flag[0]).upper():
        return "closed"
    return ""


# --------------------------------------------------------------------- searching
class KnowledgeBase:
    """A tiny BM25 index over documents built from the two official sources."""

    def __init__(self, docs: list[Doc] | None = None) -> None:
        self.docs: list[Doc] = docs or []
        self._by_title: dict[str, Doc] = {}
        self._index: dict[str, list[int]] = {}
        self._lengths: list[int] = []
        self._term_counts: list[dict[str, int]] = []
        self._avgdl = 0.0
        if self.docs:
            self.build()

    # -- construction ------------------------------------------------------
    def build(self) -> None:
        self._by_title = {d.title.lower(): d for d in self.docs}
        self._index = {}
        self._lengths = []
        self._term_counts = []
        for position, doc in enumerate(self.docs):
            tokens = tokenize(doc.title + " " + doc.text)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._term_counts.append(counts)
            self._lengths.append(len(tokens))
            for token in counts:
                self._index.setdefault(token, []).append(position)
        self._avgdl = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

    @classmethod
    def from_raw(cls, raw_dir: str | Path) -> "KnowledgeBase":
        """Read the published records, joined to the map by *title*.

        There used to be a node-id join here: the scraped record's poi_id was matched
        against the graph's ref. It never matched once -- the graph carries short ids
        ("POI_5bcd371552_10") and the scrape carries 24-character hex ones -- so the index it
        built was always empty and every lookup fell through to the title. The title is what
        the two sources genuinely share (326 of 329 node names are published titles), so it
        is now the only join, in one place, by_title().
        """
        raw = Path(raw_dir)
        docs: list[Doc] = []

        campus_path = raw / "campus_map_pois.json"
        if campus_path.exists():
            for rec in json.loads(campus_path.read_text(encoding="utf-8"))["records"]:
                names = rec.get("name") or {}
                addrs = rec.get("addr") or {}
                en = names.get("en_US") or rec.get("category", "")
                zh = " ".join(v for v in (names.get("zh_HK"), names.get("zh_CN")) if v)
                address = addrs.get("en_US", "")
                text = " ".join(
                    part for part in (
                        en, zh, address, rec.get("category", ""), rec.get("loc", ""),
                        rec.get("loc_code", ""), hours_summary(rec.get("opening_hours")),
                    ) if part
                )
                docs.append(
                    Doc(
                        id=f"poi:{rec.get('poi_id', '')}",
                        kind="place",
                        title=en,
                        text=text,
                        source="campus-map",
                        ref=rec.get("poi_id", ""),
                        fields={
                            "category": rec.get("category", ""),
                            "name_zh": names.get("zh_HK", ""),
                            "name_zh_cn": names.get("zh_CN", ""),
                            "address": address,
                            "opening_hours": rec.get("opening_hours") or {},
                            "lat": rec.get("lat"),
                            "lng": rec.get("lng"),
                            "toilet_for_disabled": rec.get("toilet_for_disabled"),
                        },
                    )
                )

        return cls(docs)

    # -- retrieval ---------------------------------------------------------
    def search(self, query: str, k: int = 5, min_score: float = 0.5) -> list[tuple[float, Doc]]:
        """Rank documents against the query with BM25.

        An empty list means "the knowledge base has nothing on this" -- the caller
        must say so rather than answer out of nowhere.
        """
        terms = tokenize(query)
        if not terms or not self.docs:
            return []
        k1, b = 1.5, 0.75
        scores: dict[int, float] = {}
        total = len(self.docs)
        for term in set(terms):
            positions = self._index.get(term)
            if not positions:
                continue
            idf = math.log(1 + (total - len(positions) + 0.5) / (len(positions) + 0.5))
            for position in positions:
                tf = self._term_counts[position].get(term, 0)
                norm = 1 - b + b * (self._lengths[position] / (self._avgdl or 1))
                scores[position] = scores.get(position, 0.0) + idf * (tf * (k1 + 1)) / (tf + k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [(score, self.docs[p]) for p, score in ranked[:k] if score >= min_score]

    # -- lookups -----------------------------------------------------------
    def by_title(self, title: str) -> Doc | None:
        return self._by_title.get((title or "").lower())

    def coverage(self) -> dict:
        """What the base actually contains -- quoted in the honest 'I do not know' reply."""
        with_hours = sum(1 for d in self.docs if d.hours)
        by_source: dict[str, int] = {}
        for doc in self.docs:
            by_source[doc.source] = by_source.get(doc.source, 0) + 1
        return {
            "documents": len(self.docs),
            "with_opening_hours": with_hours,
            "by_source": by_source,
        }
