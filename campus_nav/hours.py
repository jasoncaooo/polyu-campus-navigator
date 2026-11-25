"""Opening hours: quote the published table, or say there is no table.

This was 250 lines inside dialogue.py -- the largest single behaviour in that file, and the
only one that consults the knowledge base rather than the router. It reads a few things from
the dialogue it is handed (graph, knowledge, places, and that dialogue's card and answer
helpers) and writes nothing to the state, which is what makes it a module rather than a
method: the dialogue keeps dispatch and routing, this keeps the timetable.
"""
from __future__ import annotations

import re

from . import lexical
from .format import FacilityResult, requested_day, timetable
from .kb import hours_summary, open_state, today_hours
from .place_index import shares_a_name_word
from .replies import Reply


def document_card(talk, doc):
    # KB documents can publish a timetable even for places that have no
    # routable node. Such a document can be listed, but has no map marker.
    matches = [n for n in talk.graph.nodes.values() if n.name.casefold() == doc.title.casefold()]
    campus = [n for n in matches if n.source == "campus-map"]
    matches = campus or matches
    node = matches[0] if len(matches) == 1 else None
    card = talk._card(node) if node else {"node_id": "", "name": doc.title}
    card.update(name_zh=doc.fields.get("name_zh", ""), name_zh_cn=doc.fields.get("name_zh_cn", ""),
                opening_hours=doc.hours, hours_today=today_hours(doc.hours),
                hours_summary=hours_summary(doc.hours), open_now=open_state(doc.hours)[0])
    return card

def chinese_hours(talk, slots):
    query = slots.subject.casefold()
    day = requested_day(slots.when or query)
    kb = talk.knowledge
    # An explicit official name takes priority over a generic category.
    # Match both published Chinese names and spoken English names within a Chinese sentence.
    docs = kb.docs if kb else []
    named = []
    for doc in docs:
        if doc.source != "campus-map":
            continue
        names = [doc.title, doc.fields.get("name_zh", ""), doc.fields.get("name_zh_cn", "")]
        size = max((len(n) for n in names if len(n) >= 2 and n.casefold() in query), default=0)
        if size:
            named.append((size, doc))
    if named:
        longest = max(size for size, _ in named)
        records = [doc for size, doc in named if size == longest]
    else:
        # A generic question about a category may need a choice. Never
        # substitute a similarly named clinic's timetable for an unknown name.
        generic = re.sub(r"(?:星期|禮拜|礼拜|[週周])[一二三四五六日天]|公[眾众]假[期日]|明[天日]|聽日|听日|"
                         r"今日|今天|現在|现在|而家|依家|[請请][問问]?|[邊边哪]一?[個个家間间]|"
                         r"[開开]放[時間时间]*|[營营][業业][時間时间]*|[時时][間间]|[幾几][點点時时]|"
                         r"什[么麼][時时]候|[開开關关][門门館馆]?|有冇|[仲還还]有|[了咗]?嗎|[呢呀啊吗的嘅？? ，,]", "", query)
        generic_facility = slots.item and (not generic or generic == slots.facility_target.casefold()
                              or bool(re.fullmatch(r"[食飯饭膳餐][堂廳厅]|[診诊]所|洗手[間间]|[飲饮]水[機机]|[書书]店|[銀银]行|AED", generic, re.I)))
        records = [doc for doc in docs if generic_facility and
                   doc.fields.get("category", "").lower() == slots.item] if slots.item else []
        if not records:
            topic = generic or slots.item or slots.subject
            places = talk._preferred_matches(topic)
            if len(places) == 1:
                known = talk._card(places[0])
                info = FacilityResult("hours", rows=[known], day=day)
                return Reply(known["hours_summary"] or known["name"], slots=slots,
                             facility_result=info, facilities=[known])
            if generic_facility:
                matches = talk._service_matches(slots)
                if matches:
                    cards = [talk._card(n) for n in matches[:5]]
                    info = FacilityResult("hours_options" if len(matches) > 1 else "hours",
                                          slots.item, len(matches), rows=cards, day=day)
                    return Reply(cards[0]["name"], slots=slots, facility_result=info, facilities=cards)
            return Reply(f'No published record for "{topic}".', slots=slots,
                         facility_result=FacilityResult("no_match", topic))

    cards = [document_card(talk, doc) for doc in records[:5]]
    info = FacilityResult("hours" if len(records) == 1 else "hours_options", slots.item,
                          len(records), rows=cards, day=day)
    return Reply(cards[0]["hours_summary"] or cards[0]["name"], slots=slots,
                 facilities=[c for c in cards if c.get("node_id")], facility_result=info)

def ask(talk, slots):
    """Opening hours. Quotes the published table, or says there is not one."""
    if slots.language == "zh":
        return chinese_hours(talk, slots)
    kb = talk.knowledge
    if kb is None:
        return talk._dont_know(slots.subject or slots.item, why="no knowledge base is loaded")

    # A named clinic can be in the official directory even if its graph
    # node's service field was left empty. Prefer its own published hours.
    query = slots.subject.casefold()
    # The day the question asked about, if any. It is read from the words the user said,
    # not from the cleaned phrase: taking "on Tuesday" off the question is what makes the
    # name resolvable, and the answer still owes that day's hours.
    day = requested_day(slots.when or slots.subject)
    named_docs = [doc for doc in kb.docs if doc.source == "campus-map" and doc.hours and
                  len(doc.title) > 2 and doc.title.casefold() in query]
    if len(named_docs) == 1:
        card = document_card(talk, named_docs[0])
        _, why = open_state(named_docs[0].hours)
        text = (f'{card["name"]}:\n  {card["hours_summary"]}\n  Right now: {why}.'
                "\nSource: PolyU campus map.")
        line = (timetable(card, day, "en-US") if day
                else f'Today: {card["hours_today"] or "no hours published"}')
        # The "open until ..." fact is about now, not about the day that was asked about.
        right_now = (f'Right now: {why}.' if day
                     else f'{why[:1].upper()}{why[1:]}.')
        spoken = f'{card["name"]}. {line}. {right_now}'
        return Reply(text, slots=slots, facilities=[card] if card.get("node_id") else [],
                     spoken=spoken,
                     facility_result=FacilityResult("hours", rows=[card], day=day))

    # 1) the question names something the map can resolve
    matching = talk._service_matches(slots, slots.subject)
    # 1a) ... and if the question names one specific place ("the dental clinic"), the
    #     resolver picks that one out of the group before anything is printed. "The
    #     canteen" names twenty-four of them, so nothing is narrowed and the list is the
    #     answer. (This narrowed by the record's node id, which never matched a node, and
    #     then by BM25 title score, which happily picked one canteen out of twenty-four.)
    if len(matching) > 1 and (slots.subject or slots.item):
        wanted = {node.id for node in matching}
        named = [node for node in talk.places.resolve(slots.subject or slots.item)
                 if node.id in wanted]
        if len(named) == 1:
            matching = named
    if len(matching) > 1:
        # prefer the candidates that actually publish a table
        documented = [n for n in matching if (kb.by_title(n.name) and kb.by_title(n.name).hours)]
        if documented:
            matching = documented
    if len(matching) > 1:
        matching.sort(key=lambda n: (talk._distance_from(n) or 0.0))
        cards = [talk._card(n) for n in matching[:5]]
        lines = [f"{len(matching)} places match; hours published for today:"]
        for card in cards:
            if card["hours_today"]:
                hours = card["hours_today"]
            elif card["hours_summary"]:
                hours = "no hours published for today (other days are)"
            else:
                hours = "hours not published"
            lines.append(f'  - {card["name"]}: {hours}')
        lines.append("Which one do you mean?")
        first = cards[0]
        spoken = (
            f"{len(matching)} places match. The nearest is {first['name']}, "
            f"today {first['hours_today'] or 'closed'}. The rest are on screen."
        )
        info = FacilityResult("hours_options", slots.item, len(matching), rows=cards)
        return Reply("\n".join(lines), slots=slots, facilities=cards, spoken=spoken, facility_result=info)

    if len(matching) == 1:
        node = matching[0]
        card = talk._card(node)
        if card["hours_summary"]:
            text = (
                f'{card["name"]}'
                + (f' ({card["address"]})' if card["address"] else "")
                + f':\n  {card["hours_summary"]}'
                + f'\n  Right now: {card["open_reason"]}.'
                + "\nSource: PolyU campus map. Hours change on public holidays, which"
                  " I do not have a calendar for."
            )
            reason = card["open_reason"]
            # A question about a named day gets that day quoted, not today's figure with
            # the day thrown away -- which is how "on Tuesday" used to be answered.
            line = (timetable(card, day, "en-US") if day
                    else f'Today: {card["hours_today"] or "closed"}')
            # When a day was asked, the "open until ..." fact is about *now*, not about that
            # day: saying "Sunday 12:00-22:00. open until 23:00" invites a wrong reading.
            right_now = (f'Right now: {reason}.' if day
                         else f'{reason[:1].upper()}{reason[1:]}.')
            spoken = f'{card["name"]}. {line}. {right_now}'
            return Reply(text, slots=slots, facilities=[card], spoken=spoken,
                         facility_result=FacilityResult("hours", rows=[card], day=day))
        return Reply(
            f'The campus map does not publish opening hours for {card["name"]}. '
            f'I can tell you where it is'
            + (f' ({card["address"]})' if card["address"] else "")
            + " and draw a route, but I will not guess its hours.",
            slots=slots, facilities=[card],
            facility_result=FacilityResult("hours", rows=[card]),
            spoken=f'I do not have opening hours for {card["name"]}. '
                   f'I can show you where it is and draw a route.')

    # 2) retrieval: the question may name a place the service index does not cover
    query = slots.subject or slots.item
    hits = kb.search(query, k=6, min_score=1.0) if query else []
    # A document has to be *named* by the question, not merely the closest match: BM25 with a
    # minimum score still returns the nearest record for any words at all, and "when does the
    # library open on Tuesday?" was answered with Z Café's timetable.
    said = set(lexical.normalise(query))
    hits = [(score, doc) for score, doc in hits
            if said & set(lexical.normalise(doc.title))]

    # If the question names a place the map knows, answer about THAT place.
    # "When does the library open?" must not be answered with whichever document
    # happened to score highest -- quoting a table that belongs to somewhere else
    # is worse than admitting the table does not exist.
    # Only a place the question actually *names* may redirect the answer. The embeddings
    # return a neighbour for any phrase at all, and "when does the library open?" was answered
    # with "the campus map does not publish opening hours for Bookshop" -- a true statement
    # about the wrong place.
    named = [node for node in talk._preferred_matches(query)
             if shares_a_name_word(node, lexical.normalise(query))] if query else []
    if len(named) > 1:
        # "the library" names Block L and also the AED inside it: the building is what has a
        # timetable, so a place outranks the facilities standing in it.
        buildings = [node for node in named if node.kind != "facility"]
        named = buildings or named
    if named:
        labels = [named[0].name.lower()]
        inner = named[0].name.split("(")[-1].rstrip(")").strip()
        if inner and inner.lower() != labels[0]:
            labels.append(inner.lower())
        narrowed = [
            (score, doc) for score, doc in hits
            if any(label in doc.title.lower() for label in labels)
        ]
        if narrowed:
            hits = narrowed
        elif hits:
            return Reply(
                f'The campus map does not publish opening hours for "{named[0].name}". '
                "I can tell you where it is or draw a route, but I will not guess its "
                "hours.",
                slots=slots,
                facility_result=FacilityResult("hours", rows=[talk._card(named[0])]),
            )

    for score, doc in hits:
        if doc.hours:
            _is_open, why = open_state(doc.hours)
            text = (
                f'{doc.title}:\n  {hours_summary(doc.hours)}'
                + f'\n  Right now: {why}.'
                + "\nSource: PolyU campus map."
            )
            spoken = f'{doc.title}. Today: {today_hours(doc.hours) or "closed"}. {why[:1].upper()}{why[1:]}.'
            return Reply(text, slots=slots, spoken=spoken,
                         facility_result=FacilityResult("hours", rows=[document_card(talk, doc)]))
    if hits:
        doc = hits[0][1]
        return Reply(
            f'The campus map does not publish opening hours for "{doc.title}". '
            "I can give you its address or draw a route instead.",
            slots=slots,
            facility_result=FacilityResult("hours", rows=[document_card(talk, doc)]),
        )
    reply = talk._dont_know(query, why="nothing in my knowledge base matches that")
    reply.slots, reply.facility_result = slots, FacilityResult("no_match", query)
    return reply
