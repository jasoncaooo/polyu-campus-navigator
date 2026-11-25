"""What gets said about a reply: the one place that decides, and phrases it.

Everything this module needs may be absent -- no narration, no route, no facilities -- so the
arms are ordered from most specific to least, and the last arm truncates whatever the reply
already says. No arm branches on language: the words come from `copy.COPY`, keyed by
language, and only the *structure* lives here.
"""
from __future__ import annotations

from . import copy, format as fmt
from .graph import CampusGraph
from .local_names import place_name, service_label


def route_speech(route, graph: CampusGraph, language: str) -> str:
    """The walk in one sentence, from the numbers the search produced."""
    if not route.reachable:
        key = ("speech.unreachable.stairs" if route.blocking.get("avoid_stairs")
               else "speech.unreachable.other")
        return copy.t(key, language)
    origin = place_name(graph, route.path[0], language)
    destination = place_name(graph, route.path[-1], language)
    # The map rounds its one-decimal metre value with JavaScript Math.round.
    metres = int(round(route.total_metres, 1) + 0.5)
    head = copy.t("speech.route", language, origin=origin, destination=destination,
                  metres=metres)
    if route.blocking.get("avoid_stairs"):
        return head + copy.t("speech.route.avoids_stairs", language)
    stairs = any(segment.stairs for segment in route.segments)
    return head + copy.t("speech.route.stairs" if stairs else "speech.route.no_stairs",
                         language)


def facility_speech(reply, graph: CampusGraph, language: str) -> str | None:
    """What to say for a facility answer: the kind decides the shape, copy.py the words."""
    info = getattr(reply, "facility_result", None)
    if info is None:
        return None
    rows = fmt.rows_of(info, reply)
    separator = copy.t("hours.list_separator", language)
    subject = service_label(info.subject, language)
    if info.kind == "overview":
        top = sorted((key for key in info.counts if key), key=info.counts.get, reverse=True)[:4]
        categories = separator.join(service_label(key, language) for key in top)
        if not categories:
            return copy.t("facility.overview.none", language)
        return copy.t("facility.overview.list", language, categories=categories)
    if info.kind == "no_match":
        return copy.t("facility.no_match", language, subject=subject)
    if info.kind == "unknown_reference":
        return copy.t("facility.unknown_reference", language, subject=subject)
    if info.kind == "choice_missing":
        return copy.t("facility.choice_missing", language)
    if info.kind == "ask_category":
        return copy.t("facility.ask_category", language)
    if not rows:
        return None
    first = rows[0]
    name = fmt.place_label(first, graph, language)
    if info.kind == "no_route":
        return copy.t("facility.no_route", language, name=name)
    if info.kind == "hours":
        schedule = fmt.timetable(first, info.day, language)
        current = ""
        state = first.get("open_now")
        if not info.day and state is not None and first.get("hours_today") != "closed":
            current = copy.t("facility.hours.open_now" if state else "facility.hours.closed_now",
                             language)
        return copy.t("facility.hours.one", language, name=name, schedule=schedule,
                      current=current)
    if info.kind == "hours_options":
        names = separator.join(fmt.place_label(row, graph, language) for row in rows[:2])
        return copy.t("facility.hours_options", language, total=info.total, names=names)
    if info.kind in ("list", "selected"):
        reference = place_name(graph, info.reference, language) \
            if info.reference in graph.nodes else ""
        if reference and first.get("metres") is not None:
            # "the nearest one is X" and "X is" are different sentences, not a prefix: the
            # prefix form produced "the nearest one is X is about 120 metres away".
            chosen = "" if info.kind == "list" else ".selected"
            if first["metres"] == 0:
                origin = graph.node(info.reference)
                if (origin.lat, origin.lng) == (first.get("lat"), first.get("lng")):
                    return copy.t("facility.same_place" + chosen, language, name=name,
                                  reference=reference)
                return copy.t("facility.under_metre", language, name=name, reference=reference)
            return copy.t("facility.from_reference" + chosen, language, reference=reference,
                          name=name, metres=first["metres"])
        return copy.t("facility.found", language, name=name)
    return None


def facility_details(reply, graph: CampusGraph, language: str) -> str | None:
    """The panel under the spoken sentence: the rows, not the prose."""
    summary = facility_speech(reply, graph, language)
    if not summary:
        return None
    info = reply.facility_result
    separator = copy.t("hours.list_separator", language)
    if info.kind == "overview":
        counts = sorted(info.counts.items(), key=lambda pair: -pair[1])
        return "\n".join(copy.t("details.count", language,
                                 label=service_label(key, language), count=value)
                          for key, value in counts)
    rows = fmt.rows_of(info, reply)
    if info.kind == "hours" and rows:
        raw = rows[0].get("opening_hours") or {}
        days = [day for day in (*fmt.DAYS, "PublicHoliday") if raw.get(day)]
        if days:
            return "\n".join([fmt.place_label(rows[0], graph, language)]
                             + [fmt.timetable(rows[0], day, language) for day in days])
    if info.kind in ("list", "hours_options"):
        lines = []
        for index, card in enumerate(rows, 1):
            label = fmt.place_label(card, graph, language)
            if info.kind == "hours_options":
                label = copy.t("details.with_timetable", language, label=label,
                               times=fmt.timetable(card, info.day, language))
            elif card.get("metres") is not None:
                label = copy.t("details.metres", language, label=label, metres=card["metres"])
            lines.append(copy.t("details.numbered", language, index=index, label=label))
        if info.total > len(rows):
            lines.append(copy.t("details.more", language, shown=len(rows), total=info.total))
        return "\n".join(lines)
    return summary


def speech(reply, graph: CampusGraph, language: str = "en-US") -> str:
    """What the text-to-speech layer says for this reply."""
    explicit = getattr(reply, "spoken", "") or ""
    facility = (facility_speech(reply, graph, language)
                if getattr(reply, "facility_result", None) else None)
    # Chinese facility answers are spoken from the catalogue (that is what the dedicated
    # layer always did); every other language keeps the model's own sentence, and reaches
    # for the catalogue only when there is nothing to say.
    chinese = language in ("zh-HK", "zh-CN")
    # A narratable reply says so, and then its sentence is the answer in *any* language: the
    # model was given the language, the facts, and the guards, and its sentence passed them.
    # Anything it did not write (or wrote in the wrong script) falls through to the catalogue.
    if getattr(reply, "narrated", False) and explicit:
        return explicit
    if facility and (chinese or not explicit):
        return facility
    # In Chinese the structured arms outrank the reply's own sentence, because that sentence
    # is written by the dialogue layer (and by the narrator) in English: preferring it made
    # every Chinese route answer English -- "由Block V去圖書館點行" was answered with
    # "Yes, you can walk from Block V ...". English prefers the narration, which reads better
    # than the receipt.
    route = getattr(reply, "route", None)
    if route is not None and (chinese or not explicit):
        return route_speech(route, graph, language)
    if explicit:
        return explicit
    if route is not None:
        return route_speech(route, graph, language)
    facilities = getattr(reply, "facilities", None) or []
    if facilities:
        best = facilities[0]
        where = ", ".join(part for part in (best.get("building"), best.get("floor")) if part)
        parts = [f"The nearest match is {best.get('name')}"]
        if where:
            parts.append(f"at {where}")
        if best.get("metres") is not None:
            parts.append(f"about {best.get('metres')} metres away")
        spoken = " ".join(parts) + "."
        if len(facilities) > 1:
            spoken += f" There are {len(facilities)} on the map; the others are listed on screen."
        return spoken[:320]
    text = (reply.text or "").strip()
    short = " ".join(line.strip(" -") for line in text.split("\n\n")[0].splitlines() if line.strip())
    return short[:320]
