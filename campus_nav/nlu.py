"""Turn a sentence into slots.

Two parsers share one output type, and `parse` runs them in this order:

  * `parse_with_llm`     -- asks the model for strict JSON, then validates it. This
                           is the primary parser: it reads any phrasing, not only
                           the sentence shapes somebody enumerated. Every failure
                           -- no key, timeout, bad JSON, unknown intent, an
                           endpoint the resolver cannot name -- returns None and
                           the grammar answers instead.
  * `parse_with_grammar` -- deterministic regex rules. The fallback: the mode the
                           app runs in with no key and no network, and the safety
                           net when a model turn is unusable.

The LLM never decides a route and never invents a place: it reports what the user
said, and `_validated()` accepts a place phrase only if the resolver recognises it.
That boundary is enforced in code, not requested in a prompt.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from . import lexical
from .graph import CampusGraph
from .llm import DeepSeekClient
from .places import amenity_in_text

# The resolver every grammar rule consults: PlaceIndex.resolve(text) -> list[Node].
# It is passed in rather than imported so the rules stay testable without a network.
PlaceResolver = Any

INTENTS = (
    "route",          # find a way from A to B
    "distance",       # how far is A from B
    "close",          # this link or place is closed

    "facility_query", # where is the nearest AED / toilet / water station / canteen
    "hours_query",    # is the canteen open now / what are the clinic's hours
    "reset",
    "help",
    "out_of_scope",
)

SYSTEM_PROMPT = """You convert a user's campus-navigation message into JSON slots.

You do not know which places or facilities exist on this campus, and you are not asked to
guess: the app knows its own map and matches what the user said against it. Your job is to
report the user's words, not to classify or translate them.

Return ONLY a JSON object with exactly these keys:
{{
  "intent": one of {intents},
  "need": the words the user used for the thing they want to find, copied verbatim, or null,
  "origin": the place the user starts at, in their own words, or null,
  "destination": the place they want to reach, in their own words, or null,
  "vias": every place to walk through, in the order the user said them, as a list of
          their own words (empty or null when there are none),
  "closed": the place or link the user says is closed, as free text, or null,
  "along_the_way": true when what they want is about the walk already in progress
          ("...on the way", "stop for X en route"), otherwise false,
  "clear_stops": true when they want the stops dropped ("no stops", "the direct route"),
  "language": "en" or "zh"
}}

Rules:
- Copy the user's words into the slot: "destination": "the 7-Eleven", "origin": "Block V".
  Do not translate them, do not shorten them, do not guess an id or a coordinate, and do
  not replace them with a category of your own -- "a swim" stays "a swim", never "sports
  facilities". Never answer null for a place the user actually named: that silently turns
  their question into "where do you want to go?".
- Those examples show the shape of an answer, not its content. Use only the words this
  user wrote: if they did not say where they are starting, "origin" is null, and null is
  the right answer for a place they did not mention.
- intent "facility_query" is any "find me / where can I / I want ..." question that is not
  a route and not a timetable. Put the thing they want in "need", copied verbatim ("a
  swim", "somewhere to eat lunch", "the nearest AED", "to print something"), and leave
  origin and destination null. Whether that is an amenity, a building or an activity is
  decided by the app, which can see the whole campus; you cannot see it, so do not decide.
- intent "hours_query" for opening times: "is the canteen open now", "opening hours of the
  clinic", "what time does the bookshop close" -> put the place or service in "need".
- intent "route" for a journey, "distance" for how far, "close" when the user reports
  something closed, "reset"/"help" for those two commands, "out_of_scope" only when the
  message is not about this campus at all.
- Never copy a need or an activity into "destination": a need in a place slot becomes a
  route to somewhere nobody asked for.
- The conversation so far is included. Resolve references from it: "the nearest one", "that
  one", "the first", "there" mean the place your last answer named; "make it go via X",
  "what about going through X", "on the way" describe the journey already in progress, so
  repeat its origin and destination and set "vias"; "and the toilet?" carries the same kind
  of question. Copy the *name* the conversation used, and describe only the user's LAST
  message. What you may not do is invent a place that appears nowhere in the conversation.
- Stops are a list and keep their order: "from A through C, then passing Block D, to B" ->
  origin A, vias ["C", "Block D"], destination B. "I must pass through X, Y and Z" is three
  stops: vias ["X", "Y", "Z"].
- Output JSON only, no prose, no code fences."""


@dataclass
class Slots:
    intent: str = "out_of_scope"
    origin_text: str = ""
    destination_text: str = ""
    closed_text: str = ""    # the raw words around a closure ("the library is closed")
    # The thing the user wants to find, in their own words: "a swim", "the nearest AED".
    # The model reports it; the engine decides what kind of thing it is (see places.
    # amenity_in_text). This replaced a slot the model had to fill with a label from a list
    # of amenities, which is how "I want to have a swim" came back as "water station".
    need: str = ""
    item: str = ""           # the amenity this map offers, decided by the engine or grammar
    # The places to walk through, in the order the user said them: "through Core P, then
    # Block X" is two. A list rather than one string because that is what the request is --
    # the walk visits each stop in turn, so the order is part of the answer.
    vias: list[str] = field(default_factory=list)
    subject: str = ""      # the phrase the question is *about* (hours/facility queries)
    # The day the question asks about, exactly as it was said ("Tuesday", "聽日"). It is
    # kept apart from "subject" because taking the asking frame off the phrase takes the
    # day with it, and the timetable still has to answer for that day.
    when: str = ""
    language: str = "en"
    source: str = "grammar"
    # Why the grammar answered when a model was configured: "model abstained", "model
    # failed (timeout, quota or bad JSON)", "model slots rejected", "no model configured".
    parse_note: str = ""
    avoid_stairs: bool | None = None  # None keeps the current route preference
    clear_stops: bool = False         # "no stops" / "the direct route"
    along_the_way: bool = False       # the wish is about the walk in progress, not a new one
    unknown_origin: bool = False      # a distance answer measured from the map reference
    facility_origin: str = ""
    facility_target: str = ""
    facility_choice: int | None = None  # zero-based position in the last displayed list

    @property
    def mentions_places(self) -> bool:
        return bool(self.origin_text or self.destination_text)


# Labels that answer "where can I get X". The directory labels are excluded: offered as
# options, a model asked for cash answers "office", which is true of 45 places and useful
# for none. Everything else the map publishes is an amenity somebody would walk to.
_DIRECTORY_LABELS = frozenset({
    "office", "faculty / department", "lecture theatre", "building", "sports facilities",
    "facility", "",
})


def facility_words(graph: CampusGraph) -> list[str]:
    """The amenity labels this map actually carries, for the prompt.

    Read from the graph rather than written into the prompt, so a facility the map does
    not have can never be offered to the model as an option.
    """
    labels = {
        node.service.strip().lower() for node in graph.nodes.values()
        if node.kind == "facility" and node.service
    }
    amenities = labels - _DIRECTORY_LABELS
    return sorted(amenities or labels)

def _as_list(value: Any) -> list[str]:
    """A slot that may be a list, one string, or null, read as a list of strings.

    A prompt is not a type system: the model writes "vias": ["Core P", "Block X"] most of the
    time and "vias": "Core P" the rest, and both mean the same thing.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def parse_with_llm(text: str, graph: CampusGraph, client: DeepSeekClient,
                   context: list[dict] | None = None) -> Slots | None:
    # No place catalogue: 98 lines and ~1,400 tokens per message that the model does not
    # need, because the resolver names places from the graph. The short facility list
    # stays -- that is the vocabulary "item" has to be chosen from.
    # No map vocabulary at all: not the place catalogue (which let the model invent names)
    # and not the amenity list (which let it turn "a swim" into "water station"). Both are
    # things the engine can see and the model cannot, so both are the engine's job.
    #
    # The *conversation*, on the other hand, is language, and reading language is the one
    # thing the model is better at than this codebase. Sending a single message made every
    # referring expression ("the nearest one", "make it go via Core P", "and the toilet?")
    # unreadable, and the dialogue layer grew a rule for each phrasing to compensate. The
    # last few turns are sent instead; the guard in _validated() still decides whether the
    # model may act on them (see _DEICTIC).
    prompt = SYSTEM_PROMPT.format(intents=", ".join(f'"{i}"' for i in INTENTS))
    messages = [{"role": "system", "content": prompt}]
    messages.extend(context or [])
    messages.append({"role": "user", "content": text})
    result = client.chat(messages, max_tokens=220, retries=2)
    payload = result.json()
    if not payload:
        return None

    intent = str(payload.get("intent") or "").strip()
    if intent not in INTENTS:
        return None

    def as_id(value) -> str:
        """Turn what the model reported into the words this app resolves places by.

        A real node id becomes that node's *name*, because the rest of the pipeline
        resolves place names through the index -- and an id is not a name. Embedding
        "POI_5bcd371552_67" lands between unrelated neighbours (it scored 0.59 for Core B
        and 0.55 for Block Z, while the id belonged to Pao Yue-Kong Library), so the
        model's correct answer arrived as "which Core B did you mean?".

        Free text is kept as written: it is resolved by name later, and an id the model
        invented matches no node and stays text. This is the boundary the module
        docstring describes -- the model may report a place, it may not create one.
        """
        if not isinstance(value, str) or not value.strip():
            return ""
        candidate = value.strip()
        node = graph.nodes.get(candidate)
        if node is not None and node.kind != "waypoint":
            return node.name
        return candidate

    closed = payload.get("closed")
    closed_text = closed.strip() if isinstance(closed, str) else ""

    return Slots(
        intent=intent,
        origin_text=as_id(payload.get("origin")),
        destination_text=as_id(payload.get("destination")),
        closed_text=closed_text,
        # "need" is copied, "item" is not: any label the model volunteers is ignored, because
        # choosing from a vocabulary it cannot see is the model's one job it cannot do.
        need=str(payload.get("need") or "").strip(),
        vias=[phrase for phrase in (as_id(value) for value in
                                    _as_list(payload.get("vias", payload.get("via"))))
              if phrase],
        along_the_way=bool(payload.get("along_the_way")),
        clear_stops=bool(payload.get("clear_stops")),
        subject=topic_phrase(text) if intent in ("hours_query", "facility_query") else "",
        when=day_phrase(text, topic_phrase(text)),
        language=str(payload.get("language") or "en"),
        source="llm",
    )


# Only an explicit preference changes the route. Merely asking whether a route
# contains stairs must not remove those edges from the search.
_STAIR_REQUEST = re.compile(
    # These words belong to the preference clause, never to the preceding POI.
    r"(?:(?:\b(?:and|but|while|with|please|on the way|if possible|"
    r"I(?:['’]d like to| would like to| want to| need to)?|could you|can you)\b|"
    r"中途|路上|途中|全程|最好|盡量|尽量|請|请|麻煩|麻烦|但係|但是|但|記得|记得|我想|我)\s*)*"
    r"(?:(?P<allow>\b(?:no need to|don['’]t need to|do not need to) avoid (?:the )?(?:stairs|steps)\b|"
    r"\b(?:remove|cancel|clear) (?:the )?(?:no[- ]stairs|stair|step[- ]free) "
    r"(?:restriction|preference|filter)\b|"
    r"取消(?:不走|唔行|避[開开]|無|无)?(?:樓梯|楼梯)(?:的|嘅)?(?:限制|要求)|"
    r"\b(?:allow|permit|include|can (?:take|use)|(?:do not|don['’]t) avoid) "
    r"(?:the )?(?:stairs|steps)\b|\b(?:stairs|steps) (?:are )?(?:fine|ok(?:ay)?|allowed)\b|"
    r"(?:允許|允许|可以|可用|不必避[开開]|不用避[开開])(?:行|走|用|爬|使用)?(?:樓梯|楼梯|台阶|台階)|"
    r"(?:樓梯|楼梯)(?:都得|也可以))|"
    r"(?P<avoid>\b(?:avoid(?:ing|s)?|without(?: (?:taking|using|climbing))?|no|skip|"
    r"(?:do not|don['’]t|cannot|can['’]t)(?: want to)? (?:take|use|climb|allow)) (?:(?:any|all|the) )?(?:stairs|steps)\b|"
    r"\b(?:stair|step)[ -]free\b|"
    # Observed Mandarin ASR homophone: 不走楼梯 -> 步走楼梯. Accept the
    # full anomalous clause only, so 徒步走楼梯 (walk up stairs) is unaffected.
    r"(?<![\u3400-\u9fff])步走[楼樓]梯(?=$|[，。；！？,.!?;\s])|"
    r"(?:唔(?:好|使|要|想|可以)?|不(?:要|用|想|能)?|别|別|避[开開]|避免|繞[過开開]|绕[过开]|"
    r"無|无|沒有|没有|冇)\s*(?:再|經過|经过)?(?:行|走|用|爬|上落|上下|上|下)?\s*(?:所有|任何)?\s*"
    r"(?:樓梯|楼梯|台阶|台階|梯級)))",
    re.I,
)


def stair_request(text: str) -> tuple[str, bool | None]:
    """Remove preference words before parsing the origin and destination."""
    matches = list(_STAIR_REQUEST.finditer(text))
    if not matches:
        return text.strip(), None
    clean = _STAIR_REQUEST.sub(" ", text)
    clean = re.sub(r"\bplease[.!? ]*$|(?:的|嘅)(?:路线|路線)[。！？ ]*$", "", clean, flags=re.I)
    clean = re.sub(r"\b(?:and|while|with)\s*[.,!?]*$", "", clean, flags=re.I)
    return clean.strip(" ，,。.!?！？"), matches[-1].lastgroup == "avoid"


# --- grammar fallback -----------------------------------------------------
# Every rule below is a *signal*: it answers one yes/no question about the sentence, and
# parse_with_grammar() decides the intent from those answers in a fixed order. Keeping
# the decision in one place is what stops a new phrasing from quietly changing an old
# one -- the order is the specification.

# "re-open it" lifts every closure, so a demo can recover from its own failure.
REOPEN = re.compile(
    r"\b(re-?open|open (?:it|that|them|everything) again|restore|unblock|"
    r"clear the closures?|remove the closures?)\b",
    re.I,
)
# Closure words, in both scripts, in one place. The grammar reads closures with this and
# the model guard checks it too: two lists that must agree is a bug waiting to happen (they
# have already disagreed once in this project's history, over filler words). The Chinese
# half matters because the grammar's English rules cannot see "圖書館今日關閉", which the
# model had to be allowed to read.
_CLOSED_SAID = re.compile(
    r"\b(?:close[ds]?|closing|shut|blocked|out of service|under maintenance|"
    r"no longer (?:open|available))\b|"
    r"關閉|关闭|關門|关门|裝修|装修|維修|维修|唔開|不开|停開|停开|"
    r"暫停開放|暂停开放|閉館|闭馆|封閉|封闭|停用"
)
_FROM_TO = re.compile(r"\bfrom\s+(?P<o>.+?)\s+\bto\s+(?P<d>.+?)(?:[?.!,]|$)", re.I)
# A stop bolted onto the end of a request: "from A to B but stop at C", "route me to A via
# B". Only a *trailing* clause, and only behind an explicit marker -- cutting clauses out of
# the middle of a sentence moved endpoints around, which is why the grammar keeps a
# two-slot contract. Without this, "core a but stop at core p" arrived as one destination.
# The same stop said first: "pass by Core P on the way from A to B". The trailing marker
# after the name ("on the way", "first", "then") is what makes the cut safe -- it is an
# explicit signal that the clause has ended, which is exactly what was missing when cutting
# clauses out of the middle of sentences moved endpoints around.
_VIA_PREFIX = re.compile(
    r"^(?:(?:could you|can you|please|i want to|i'd like to)\s+)?"
    r"(?:via|through|by\s+way\s+of|"
    r"pass(?:ing)?\s+(?:by|through)|"
    r"stop(?:ping)?\s+(?:at|by|off\s+at|in)|"
    r"go(?:ing)?\s+(?:through|via|past))\s+"
    r"(?P<via>[^,.;?!]+?)\s+(?:(?:on|along)\s+the\s+way|first|then)\b",
    re.I,
)
_VIA_CLAUSE = re.compile(
    r"[,\s]+(?:(?:but|and|then|also|please)\s+)?"
    r"(?:via|through|by\s+way\s+of|"
    r"pass(?:ing)?\s+(?:by|through)|"
    r"stop(?:ping)?\s+(?:at|by|off\s+at|in)|"
    r"go(?:ing)?\s+(?:through|via|past))\s+"
    # the idiom after the name is not part of the name: "through Core P on the way"
    r"(?P<via>[^,.;?!]+?)\s*(?:(?:on|along)\s+the\s+way|if\s+possible|please)?"
    r"\s*[.?!]*$",
    re.I,
)
# "How do I get to D from O?" and "How far is D from O?" put D first.
_AT_AND_GO = re.compile(
    r"\bi(?: am|['’]m)\s+(?:currently\s+|now\s+)?at\s+(?P<o>.+?)"
    r"[,.;]?\s+(?:and\s+)?(?:i\s+)?(?:want\s+to\s+|would\s+like\s+to\s+)?"
    r"(?:go|get|walk|head)\s+to\s+(?P<d>.+?)(?:[?!.]|$)", re.I,
)
_TO_FROM = re.compile(
    r"(?:^to|\b(?:get|go|walk|head|navigate|directions|route|take me)\s+to|\bhow far is)\s+(?P<d>.+?)\s*[,.;]?\s+"
    r"(?:(?:starting\s+)?from|i(?: am|['’]m)\s+(?:currently\s+|now\s+)?at)\s+"
    r"(?P<o>.+?)(?:[?!.]|$)", re.I,
)
_DISTANCE_BETWEEN = re.compile(
    r"\bdistance\s+between\s+(?P<o>.+?)\s+and\s+(?P<d>.+?)(?:[?!.]|$)", re.I,
)
# Spoken requests often omit "from": "Core D to Block Z". Keep commands
# such as "take me to the library" as destination-only requests.
_PLACE_TO_PLACE = re.compile(r"^(?P<o>.+?)\s+to\s+(?P<d>.+?)(?:[?.!,]|$)", re.I)
# A word list, not a pattern: the sentence opens with a command or a question. Used as
# a guard only -- "Core D to Block Z" is a route request, "the link to Core D" is not.
_COMMAND_WORDS = frozenset({
    "how", "where", "what", "when", "why", "which", "can", "could", "would", "is",
    "are", "i", "we", "please", "take", "go", "get", "walk", "head", "navigate",
    "direction", "directions", "route", "who", "will", "do", "does", "tell", "show",
    "need", "want", "to", "from", "at", "of", "in", "on", "and",
})
_COMMAND_PHRASES = ("the link",)
# Both scripts are accepted. Keep each campus place intact, including English
# names embedded in a Chinese utterance ("由 Core D 去 Block Z 點行？").
_ZH_PAIR = re.compile(
    r"(?:[从從由]|我(?:而家|依家|現在|现在)?(?:喺|在))\s*(?P<o>.+?)"
    r"\s*[,，。;；]?\s*(?:出[发發]\s*)?(?:我)?(?:想|要)?(?:前往|去|到|往)\s*(?P<d>.+)"
)
_ZH_TO_FROM = re.compile(
    r"(?:去|到|前往)\s*(?P<d>.+?)\s*[,，。;；]?\s*"
    r"(?:我(?:而家|依家|現在|现在)?(?:喺|在)|[从從由])\s*(?P<o>.+)"
)
_ZH_DISTANCE_FROM = re.compile(r"^(?P<d>.+?)(?:距[离離]|[离離])(?P<o>.+)")
_ZH_SHORT_PAIR = re.compile(r"^(?P<o>.+?)\s*(?:去|到)\s*(?P<d>.+)")
_ZH_TO = re.compile(r"^(?:[请請](?:問|问)?\s*)?(?:我(?:想|要)|帶我|带我|點樣|怎麼|怎么)?"
                    r"(?:前往|去|到)\s*(?P<d>.+)")
_ZH_ENDING = re.compile(r"[,，\s]*(?:[点點](?:樣|样)?(?:行|走|去)|怎[么麼样樣](?:走|去)|"
                        r"怎樣走|如何(?:走|去)|該怎麼走|该怎么走)[呢啊呀嗎吗]?$")
# "to X" needs a *request* in front of it, not just the preposition. A bare "\bto\s+" matched
# every "to" in the language, so "I want to make sure I pass through Core P on the way" gave
# the destination "make sure I pass through Core P on the way" -- the fragment resolved, so
# every check downstream believed it. The marker is what makes a request: a leading "to", or
# a travel verb immediately before it.
_TO_ONLY = re.compile(
    r"(?:^to\s+|\b(?:get|go|walk|head|navigate|route|take)\s+(?:me\s+)?to\s+|"
    r"\bdirections\s+to\s+)"
    r"(?P<d>[a-z0-9][\w &'\-/().]*?)(?:[?.!,]|$)",
    re.I,
)
# If a travel verb or leading "to" is explicit, preserve an unknown
# destination as written so the dialogue can ask for its name.
_DEST_REQUEST = re.compile(
    r"^to\s+|\b(?:get|go|walk|head|navigate|route|take)\s+(?:me\s+)?to\s+|"
    r"\bdirections\s+to\s+", re.I,
)
_ZH_DISTANCE_END = re.compile(r"(?:大[概約约])?(?:有|要)?(?:多[遠远]|[幾几][遠远]|多少米|[幾几]多米)[呢啊呀嗎吗]?$")
_HOW_FAR = re.compile(r"\bhow\s+far\b|\bdistance\b|多[遠远]|[幾几][遠远]|多少米|[幾几]多米", re.I)
_DISTANCE_SUBJECT = re.compile(r"^how far (?:away )?is (?:it to )?(?P<d>.+?)[?!.]*$", re.I)
_DISTANCE_FOLLOWUP = re.compile(
    r"how far(?: is (?:it|that|this|the route))?|(?:what(?:['’]s| is) )?(?:the )?distance|"
    r"(?:有)?(?:多[遠远]|[幾几][遠远])", re.I,
)

# "Is this a timetable question?" -- one signal, both languages. They were three
# patterns (English words, Chinese 幾點-class words, Chinese 開/關-class words) that were
# only ever tested for truth, and their union is what the decision actually used.
# Checked BEFORE the closure rule: "is the canteen closed on Sunday?" is a timetable
# question, while "the link to Core D is closed" is an instruction.
_HOURS = re.compile(
    r"\b(opening hours?|opening times?|business hours?|what time|which time|"
    r"until when|how late|when (?:is|does|do)\b[^?]*\b(?:open|close)|"
    r"(?:is|are)\b[^?]*\b(?:open|closed)(?![a-z])|still open|open (?:today|now|tonight|"
    r"on (?:mon|tue|wed|thu|fri|sat|sun))|close[sd]? at|hours? of)\b"
    r"|幾點|幾時|開放|營業|開門|關門|時間|關了|開了"
    r"|[開开](?:放|門|门|館|馆|到|咗|了|唔[開开]|不[開开])|[關关][門门館馆]"
    r"|[營营][業业]|[幾几][點点時时]|什[麼么][時时]候|收[幾几][點点]|有冇[開开]"
    r"|(?:有|[仲還还])[開开]|[開开][嗎吗未]",
    re.I,
)

# Where the question points, and how it is phrased. Named so that the decision in
# parse_with_grammar() reads as prose instead of an anonymous re.match().
_PASSAGE = re.compile(r"\b(passage|link|walkway|corridor|connection|route|entrance|exit)\b", re.I)
_ASKS_AS_HOURS = re.compile(r"^(?:what time|when|is|are)\b", re.I)
# A wish attached to the walk in progress, rather than a new journey: "I also want a quick
# breakfast along the way", "stop for a coffee en route". The reading belongs here, with the
# rest of the language work; what to *do* about it (insert a stop, which one, at what cost) is
# the dialogue's decision.
_ALONG_THE_WAY = re.compile(
    r"\b(?:along|on)\s+(?:the|my)\s+way\b|\ben\s+route\b|\bon\s+my\s+route\b|"
    r"\bwhile\s+(?:i\s+)?(?:walk|walking|passing)\b|"
    r"途中|沿途|順路|顺路|路上",
    re.I,
)
# "no stops", "the direct route": drop the places added along the way.
_NO_STOPS = re.compile(
    r"\b(?:no|without|drop|remove|forget|clear|skip)\b[^.!?]{0,24}\bstops?\b|"
    r"\b(?:the\s+)?direct\s+(?:route|way)\b|\bgo\s+direct(?:ly)?\b|"
    r"\bno\s+(?:more\s+)?(?:detours?|stops?)\b|"
    r"唔使(?:再)?(?:經|经)|(?:不|唔)(?:再)?(?:經過|经过|经)|直接去",
    re.I,
)

_SEEKS_NEAREST = re.compile(r"^(?:where|which|what|find|show|locate)\b.*?\b(?:closest|nearest)\b", re.I)
_CLOSED_WORD = re.compile(r"\bclosed\b", re.I)
_HAS_CJK = re.compile(r"[\u3400-\u9fff]")


def has_cjk(text: str) -> bool:
    """Whether these words are written in Chinese -- which decides the display language."""
    return bool(_HAS_CJK.search(text or ""))

# --- facility: which service does the question name, if any --------------------
# One pattern, not two. These are the words a person uses for the things this map
# carries; the captured group is the item the dialogue then looks up.


# --- meta: the two commands that are about the conversation, not the campus ------
_RESET = re.compile(r"^\s*(reset|restart|start over|clear|new route)\s*$", re.I)
_HELP = re.compile(r"\b(help|what can you do|how do you work)\b", re.I)

# --- Chinese: the same eight intents, written differently -----------------------
# The lexicon is data: a service word (or several) per facility. Everything else the
# Chinese rules need is a small named pattern, gathered here rather than interleaved
# with the English ones.
# Vocabulary for the categories actually searched. Keep the user's place or item
# wording separately, so an unknown service never turns into a generic list.
_ZH_SERVICES = {
    "accessible toilet": r"無障礙廁所|无障碍厕所|暢通易達洗手間|畅通易达洗手间|傷健洗手間",
    "toilet": r"洗手[間间]|[廁厕厠]所|[衛卫]生[間间]",
    "water station": r"[飲饮]水(?:[機机站點点]|的?地方)?|[裝装斟添補补]水(?:[機机站]|的?地方)?",
    "aed": r"(?:自[動动](?:體外|体外)?)?(?:去|除)[顫颤]器|心[臟脏]除[顫颤]器|AED",
    "canteen": r"食堂|[飯饭]堂|[餐膳][廳厅堂]",
    "clinic": r"[診诊]所|[醫医][務务]室", "bank": r"[銀银]行|ATM",
    "postbox": r"郵筒|邮筒|郵箱|邮箱", "bookstore": r"[書书]店|[書书]局",
    "lactation room": r"集乳室|哺乳室|母[嬰婴]室", "prayer room": r"祈[禱祷]室",
}
_ZH_SERVICE = re.compile("|".join(f"(?P<s{i}>{words})" for i, words in enumerate(_ZH_SERVICES.values())), re.I)
_ZH_WHERE = re.compile(r"最近|附近|哪[裏里儿]|[哪邊边]度|[邊边]個|有[冇沒没]|在哪|有哪|[搵找]|有[咩乜]|有什[么麼]")
_ZH_OVERVIEW = re.compile(r"(?:有|列[出舉举]|看看|[顯显]示).*(?:[設设]施|[設设][備备])|[設设]施(?:列表|清[單单])")
_ZH_CHOICE = re.compile(r"第\s*([一二兩两三四五六七八九十1-9])\s*(?:[個个處处項项間间家]|座)|最近(?:的|嗰|那)[個个間间家]|[那嗰這这](?:一)?[個个間间家]")
_ZH_ROUTE_ACTION = re.compile(r"[帶带]我|怎[么麼]去|[點点]去|去第|去[那嗰]|去最近|[導导]航|怎[么麼]走|[點点]行")

# The shapes that carry no place of their own: a bare "where is the nearest one", a bare
# "what time", and the three ways a question says where it is measured from.
_ZH_NEAREST_ONLY = re.compile(r"最近(?:的|嘅|嗰個|那[个個])?(?:[喺在]?[哪邊边](?:[裡里裏度儿兒])?)[呀啊呢]?")
_ZH_TIME_ONLY = re.compile(r"(?:它|佢|[那咁]麼?|[那咁])?(?:的|嘅)?(?:[營营][業业]|[開开]放)?[時时][間间][呢呀啊]?")
_ZH_POINTS_AT_PLACE = re.compile(r"^(?:它|佢|嗰度|那里|那裡)")
_ZH_TIME_QUESTION = re.compile(
    r"(?:今日|今天|而家|現在|现在)?(?:[幾几][點点時时]|什[麼么][時时]候)?"
    r"(?:[開开關关][門门]?|[營营][業业]|[開开]放)[吗嗎呢呀啊未]?"
)
_ZH_REFERENCE_PATTERNS = (
    re.compile(r"(?:距[離离]|[離离])\s*(.+?)\s*最近"),
    re.compile(r"^我(?:而家|依家|現在|现在)?[喺在]\s*([^，,。]+?)(?:[，,。]|附近|最近)"),
    re.compile(r"^(.+?)(?:附近|旁[边邊])"),
)
_ZH_IS_HERE = re.compile(
    r"我(?:[這这哪呢]?[裡里裏度邊边兒儿]|所[在處处]|(?:目前|[現现]在)?的位置)?|"
    r"[當当]前位置|[這这呢][裡里裏度兒儿]|身[邊边]"
)

# (There was a second amenity regex here. It listed the same words as
# places.FACILITY_WORDS, in a different form, so the two could drift -- one knew "canteen"
# and the other "catering", and a need understood by one was invisible to the other. The
# table is the single source; amenity_in_text() reads it on word boundaries.)

# "what facilities are there / what can I find here" -> a facilities overview
_FACILITY_ALL = re.compile(
    r"\b(facilit(?:y|ies)|amenities|what places|what can i find|what is here|"
    r"what's here|what is available|what services)\b", re.I)

def asks_for_everything(text: str) -> bool:
    """Is this a request for the catalogue rather than for one place?

    "What facilities are there?" and "where can I sit outside and study?" both arrive with
    no amenity item and no place name. The first wants the list; the second wants a place
    found by meaning. The words are the only thing that separates them.
    """
    return bool(_FACILITY_ALL.search(text or ""))


def _clean_place(text: str) -> str:
    """Drop the verb and the filler, but never the letter in a building name.

    "a", "an", "the" are filler -- except in "Core A", "Block A", "Wing A", where the
    single letter IS the place. Stripping it turned "Core A" into "Core", which matches
    eight buildings and forces a disambiguation question, so no route was ever drawn.

    The filler list itself lives in campus_nav/lexical.py: the resolver and the parser
    have to agree on what a phrase is about, and when they each had their own list a
    closure sentence resolved its place to nothing.
    """
    # "go to Block Y" -> "Block Y": a verb in front of the place is not part of its name
    text = re.sub(r"^(?:get|go|walk|head|take\s+me)\s+(?:to\s+)?", "", text.strip(), flags=re.I)
    return lexical.drop_filler(text)


def refers_to_last_distance(text: str) -> bool:
    """An elliptical follow-up explicitly asks about the current trip."""
    route_text, _ = stair_request(text)
    return bool(_DISTANCE_FOLLOWUP.fullmatch(route_text.rstrip("？！。?!. ")))


# Words that continue a thought rather than start a request. "but I want to make sure I
# pass through Core P on the way" opened with "but", so the bare "A to B" rule read it as
# origin "want", destination "Core P" -- a sentence about a stop became a new journey.
_CONTINUATION = frozenset({
    "but", "and", "so", "also", "then", "actually", "oh", "well", "because", "if", "when",
    "since", "plus", "though", "however", "instead",
})


def reads_as_a_clause(text: str) -> bool:
    """Too long, or continuing a thought, to be a bare pair of place names."""
    words = re.findall(r"[a-z']+", (text or "").lower())
    if len(words) > 6:
        return True
    return bool(words) and words[0] in _CONTINUATION


def opens_with_a_request(text: str) -> bool:
    """Does the sentence start with a command or question word?"""
    lowered = (text or "").lstrip().lower()
    if lowered.startswith(_COMMAND_PHRASES):
        return True
    words = re.findall(r"[a-z]+", lowered)
    return bool(words) and words[0] in _COMMAND_WORDS


# The shapes that name a pair, in priority order. Data, not a chain of ors: the order is
# the tie-break rule when a sentence fits two shapes, so it belongs somewhere readable.
_PAIR_PATTERNS = (_FROM_TO, _AT_AND_GO, _TO_FROM, _DISTANCE_BETWEEN, _ZH_TO_FROM, _ZH_PAIR)


def _route_pair(text: str, places: PlaceResolver) -> re.Match | None:
    pair = next((found for pattern in _PAIR_PATTERNS
                 if (found := pattern.search(text))), None)
    chinese_clause = text.rstrip(" 。！？?!")
    if pair is None and _ZH_DISTANCE_END.search(chinese_clause):
        pair = _ZH_DISTANCE_FROM.match(_ZH_DISTANCE_END.sub("", chinese_clause).strip())
    if pair is None and not _ZH_TO.match(text):
        short = _ZH_SHORT_PAIR.match(text)
        if short and (_ZH_ENDING.search(chinese_clause) or _ZH_DISTANCE_END.search(chinese_clause)
                      or re.match(r"^(?:core|block)\s+\w+", text, re.I)):
            pair = short
        elif short:
            # Bare "图书馆到赛马会创新楼" has no other instruction words. Only
            # accept this shorthand if both sides name a place in the index.
            if all(places.resolve(name) for name in short.group("o", "d")):
                pair = short
    if pair is None and not opens_with_a_request(text) and not reads_as_a_clause(text):
        bare = _PLACE_TO_PLACE.match(text)
        # A bare "A to B" is only a pair if it names two places. "but I want to make sure I
        # pass through Core P on the way" splits at its "to" into "but I want" and "make
        # sure I pass through Core P on the way" -- the shape was the coincidence of a
        # preposition, not a route, and the left side names nothing. The rule that decides
        # is evidence, exactly as it is everywhere else in this module: both sides have to
        # resolve. reads_as_a_clause() above is only the cheap pre-filter that keeps a
        # long sentence from costing two resolver round-trips; this is the authority, and
        # it is the same test the Chinese shorthand below already applied.
        if bare and all(places.resolve(side) for side in bare.group("o", "d")):
            pair = bare
    return pair


def _clean_endpoint(text: str) -> str:
    text = text.strip(" ，,。！？?!")
    text = _ZH_DISTANCE_END.sub("", _ZH_ENDING.sub("", text)).strip(" ，,。！？?!")
    return _clean_place(re.sub(r"出[發发]$", "", text).strip())


def names_destination(text: str, places: PlaceResolver) -> bool:
    """A directed request must not be mistaken for a bare clarification."""
    return bool(_TO_ONLY.search(text) or _ZH_TO.match(text.strip()) or _route_pair(text.strip(), places))


def _chinese_facility(text: str, places: PlaceResolver) -> Slots | None:
    if not _HAS_CJK.search(text):
        return None
    phrase = text.strip(" ?？!！.。，")
    match = _ZH_SERVICE.search(phrase)
    item = list(_ZH_SERVICES)[int(match.lastgroup[1:])] if match else ""
    choice = _ZH_CHOICE.search(phrase)
    hours = _HOURS.search(phrase)
    place_words = re.sub(r"^(?:[請请][問问](?:一下)?|[請请]|想[問问](?:一下)?)", "", phrase).strip()
    slots = Slots(language="zh", subject=phrase, source="grammar", item=item)
    if not item and _ZH_NEAREST_ONLY.fullmatch(phrase):
        slots.intent, slots.facility_choice = "facility_query", 0
        return slots
    if choice and not item:
        index = choice.group(1) or "一"
        slots.facility_choice = ("一二三四五六七八九十".index(index.replace("兩", "二").replace("两", "二"))
                                 if not index.isascii() else int(index) - 1)
        slots.intent = "hours_query" if hours else ("route" if _ZH_ROUTE_ACTION.search(phrase) else "facility_query")
        return slots
    if not item and _ZH_TIME_ONLY.fullmatch(phrase):
        slots.intent, slots.facility_choice = "hours_query", 0
        return slots
    if hours:
        slots.intent = "hours_query"
        if not item and (_ZH_POINTS_AT_PLACE.match(place_words)
                         or _ZH_TIME_QUESTION.fullmatch(place_words)):
            slots.facility_choice = 0
        slots.facility_target = place_words
        return slots
    explicit_route = names_destination(phrase, places)
    if explicit_route and not (_ZH_WHERE.search(phrase) and not _ZH_ROUTE_ACTION.search(phrase)):
        return None
    if item and (_ZH_WHERE.search(phrase) or not explicit_route):
        slots.intent = "facility_query"
        slots.facility_target = match.group()
        reference = next((found for pattern in _ZH_REFERENCE_PATTERNS
                          if (found := pattern.search(place_words))), None)
        if reference:
            slots.facility_origin = reference.group(1).strip(" ，,的嘅")
            if _ZH_IS_HERE.fullmatch(slots.facility_origin):
                slots.facility_origin = ""
        return slots
    if _ZH_OVERVIEW.search(phrase):
        slots.intent = "facility_query"
        return slots
    return None
# A question about a place carries the place inside a frame: "when does the library open",
# "what are the canteen's hours". Everything around the name belongs to the asking, not to
# the name, and handing that frame to the resolver is what once answered "the library" with
# the defibrillator inside it -- the sentence's own words matched every building on the map.
# The engine owns this vocabulary, so it strips the frame here rather than in each caller.
_LEADING_FRAME = re.compile(
    r"^(?:(?:when|what|whats|where|which|how|is|are|was|were|do|does|did|can|could|"
    r"may|should|tell|me|please|time|times|hour|hours)\b|(?:the|a|an)\b)[\s,:]*", re.I)
_TRAILING_FRAME = re.compile(
    r"[\s,:]*(?:\b(?:open|opens|opened|opening|close|closes|closed|closing|start|starts|"
    r"started|starting|end|ends|ending|finish|finishes|shut|shuts|operate|operates|"
    r"available|on|at|today|tomorrow|now|currently|right)\b.*)$", re.I)


# The same frame in Chinese, where there are no word boundaries to rely on and the
# question mark is often a particle. These are asking words, never part of a name: no
# published name on this map ends in 的/呢/呀/嗎/嘅/邊/點 (checked against the graph).
_CJK_LEADING_FRAME = re.compile(
    r"^(?:請問|请问|請|请|我想去|我想|想去|想問|想问|想知|想搵|想找|搵|找|"
    r"帶我去|带我去|邊度有|边度有|哪裡有|哪里有|有冇|有沒有|有無|"
    r"如何去|怎麼去|怎么去|點去|点去|點行|点行|聽日|听日|明日|明早|明天|今日|今天|"
    r"去|到)[\s,，、]*")
_CJK_TRAILING_FRAME = re.compile(
    r"(?:喺邊度|喺边度|喺邊|喺边|喺哪裏|喺哪里|在哪裡|在哪里|在哪|"
    r"幾點開門|几点开门|幾點關門|几点关门|幾點開|几点开|幾點關|几点关|幾點|几点|"
    r"開放時間|开放时间|營業時間|营业时间|開門時間|关门时间|"
    r"點樣去|点样去|點去|点去|點行|点行|怎麼去|怎么去|如何去|"
    r"邊度|边度|哪裡|哪里|位置|地址|嘅|的|呢|呀|嗎|吗|吖|啦|喇|"
    # the day is asking frame too -- it is kept as the day, not inside the name
    r"聽日|听日|明日|明早|明天|今日|今天|今晚|琴日|尋日)"
    r"[\s,，、。？！?!…]*$")


# The words that name a day, in both scripts: one place reads them, for the timetable.
_DAY_WORD = re.compile(
    r"公[眾众]假[期日]|公共假期|(?:星期|禮拜|礼拜|[週周])[一二三四五六日天]|明[天日]|聽日|听日|"
    r"今日|今天|(?<![a-z])(?:mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)(?:day)?"
    r"(?![a-z])|tomorrow|today|public\s+holiday", re.IGNORECASE)


def day_phrase(text: str, subject: str = "") -> str:
    """The day words a question contains, or "" when it asks about no particular day.

    A day word inside the name being asked about is part of the name, not a day:
    "Suen Chi Sun Memorial Square" is not a question about Sunday.
    """
    found = _DAY_WORD.findall(text or "")
    if found and subject:
        found = [word for word in found
                 if not re.search(rf"(?i)(?<![a-z]){re.escape(word)}(?![a-z])", subject)]
    return " ".join(found) if found else ""


def topic_phrase(text: str) -> str:
    """The phrase a question is about: "the library", not the sentence around it.

    Strips the asking frame from both ends, repeatedly, because frames nest
    ("what are the library's opening hours so far today"). A phrase that survives
    unchanged is already the name. Empty means the question named nothing.
    """
    phrase = (text or "").strip().strip("?？。！!，, ")
    for _ in range(4):
        before = phrase
        phrase = _LEADING_FRAME.sub("", phrase).strip()
        phrase = _CJK_LEADING_FRAME.sub("", phrase).strip()
        phrase = _TRAILING_FRAME.sub("", phrase).strip()
        phrase = _CJK_TRAILING_FRAME.sub("", phrase).strip()
        phrase = phrase.strip("?？。！!，, '\"\u2019s")
        if phrase == before:
            break
    return phrase


def parse_with_grammar(text: str, places: PlaceResolver) -> Slots:
    lowered, avoid_stairs = stair_request(text)

    # A stop at the end of the sentence is a stop, not part of the destination. The phrase
    # has to name a place -- the same evidence rule as everywhere else -- so a stray
    # "through the tunnel" is left in the sentence rather than turned into a waypoint.
    vias: list[str] = []
    trailing_via = _VIA_CLAUSE.search(lowered)
    if trailing_via:
        candidate = _clean_endpoint(trailing_via.group("via"))
        if candidate and places.resolve(candidate):
            vias = [candidate]
            lowered = lowered[:trailing_via.start()].strip(" ,.;:")
    if not vias:
        leading_via = _VIA_PREFIX.match(lowered)
        if leading_via:
            candidate = _clean_endpoint(leading_via.group("via"))
            if candidate and places.resolve(candidate):
                vias = [candidate]
                lowered = lowered[leading_via.end():].strip(" ,.;:")

    facility = _chinese_facility(lowered, places)
    if facility:
        facility.avoid_stairs = avoid_stairs
        return facility

    if _RESET.match(lowered):
        return Slots(intent="reset", source="grammar")
    if _HELP.search(lowered) and not names_destination(lowered, places):
        return Slots(intent="help", source="grammar")
    if REOPEN.search(lowered):
        return Slots(intent="close", closed_text=lowered.strip(), source="grammar")
    if refers_to_last_distance(lowered):
        return Slots(intent="distance", avoid_stairs=avoid_stairs, source="grammar",
                     language="zh" if _HAS_CJK.search(lowered) else "en")

    origin_text = destination_text = ""
    # A stop ("pass by Core P") is read by the model, not by these patterns: cutting a
    # clause out of the sentence moved the endpoints ("Block X" became the destination)
    # far more often than it helped. The grammar keeps its two-slot contract.
    # "explicit route" means the sentence really said from/to. A bare place name
    # ("is the 7-Eleven open") must not be read as a route request.
    explicit_route = False
    m = _route_pair(lowered, places)
    if m:
        origin_text = _clean_endpoint(m.group("o"))
        destination_text = _clean_endpoint(m.group("d"))
        explicit_route = True
    else:
        m2 = _ZH_TO.match(lowered) or _TO_ONLY.search(lowered)
        if m2:
            cand = _clean_endpoint(m2.group("d"))
            if m2.re is _ZH_TO or _DEST_REQUEST.search(lowered) or places.resolve(cand):
                destination_text = cand
                explicit_route = True

    # "How far is JO?" still names a new destination, even if unknown.
    if not m and not destination_text:
        subject = _DISTANCE_SUBJECT.match(lowered)
        if subject:
            destination_text = _clean_endpoint(subject.group("d"))

    # A place mentioned without "to/from" ("the library?") still counts -- but held here,
    # not assigned, because a sentence that mentions a place is not always *about* that
    # place. "What facilities are there?" mentions "UMF - Cleanroom Facility", and
    # answering a request for the list of amenities with a route to a cleanroom is the
    # wrong kind of right. The more specific rules below get first refusal.
    mentioned = ""
    if not destination_text:
        hits = places.resolve(lowered)
        if len(hits) == 1:
            mentioned = hits[0].name

    closed_text = lowered.strip() if _CLOSED_SAID.search(lowered) else ""

    item = amenity_in_text(lowered)
    if not item:
        # The engine owns the amenity vocabulary (places.FACILITY_WORDS): "lunch" is
        # catering, "sick" is a clinic, "cash" is a bank. The grammar used to keep its own
        # short list of these words, so a need outside it fell straight through -- "I want to
        # grab lunch, where can I head to?" was answered with "I can find walking routes,
        # point you to facilities...". One table, read by both parsers.
        item = amenity_in_text(lowered)
    wants_hours = bool(_HOURS.search(lowered))
    # "the link to Core D is closed" and "the library is closed today" are reports
    # about the map; "is the canteen closed on Sunday?" and "what time does the
    # bookshop close" are questions about a timetable. Two signals separate them:
    # naming a physical link, and whether the sentence is phrased as a question.
    asked_as_question = text.strip().endswith("?")
    closed_adjective = _CLOSED_WORD.search(lowered) is not None
    if closed_adjective and (_PASSAGE.search(lowered) or not asked_as_question):
        wants_hours = False
    # "how do I get to X" outranks "is X open": a route request is never a timetable
    # request, even when the sentence happens to contain the word "open".
    asks_for_route = explicit_route

    asks_for_hours = _ASKS_AS_HOURS.match(lowered)
    seeks_nearest = _SEEKS_NEAREST.match(lowered)
    if wants_hours and (not asks_for_route or asks_for_hours):
        intent = "hours_query"
        origin_text = destination_text = ""   # a timetable question is not a route
    elif closed_text and not wants_hours:
        intent = "close"
    elif asks_for_route and not (item and seeks_nearest):
        intent = "distance" if _HOW_FAR.search(lowered) else "route"
    elif item:
        intent = "facility_query"
    elif _FACILITY_ALL.search(lowered) and not (origin_text or destination_text):
        intent = "facility_query"   # no subject -> the overview reply lists what exists
    elif mentioned and not (origin_text or destination_text):
        destination_text = mentioned        # a bare place name is the thing asked about
        intent = "route"
    elif _HOW_FAR.search(lowered):
        intent = "distance"
    elif origin_text or destination_text or vias or avoid_stairs is not None:
        intent = "route"
    else:
        intent = "out_of_scope"

    subject = ""
    if intent in ("facility_query", "hours_query"):
        # The phrase, not the sentence: the knowledge base and the resolver both take a
        # name, and a name is what is left when the asking frame is taken away.
        subject = topic_phrase(text)

    return Slots(
        intent=intent,
        origin_text=origin_text,
        destination_text=destination_text,
        vias=vias,
        along_the_way=bool(_ALONG_THE_WAY.search(text)),
        clear_stops=bool(_NO_STOPS.search(text)) and intent in ("route", "out_of_scope"),
        closed_text=closed_text,
        item=item,
        subject=subject,
        when=day_phrase(text, subject),
        language="zh" if _HAS_CJK.search(text) else "en",
        avoid_stairs=avoid_stairs,
        source="grammar",
    )


# Referring expressions: a message that contains one is *about* something the conversation
# already said, so a slot may be filled from the recent turns. A message without one must be
# self-contained -- which is what keeps "where is the library?" from inheriting the origin of
# a route two turns ago. This is a permission gate, not a resolver: the model still has to
# read the sentence and the engine still owns every place it names.
_DEICTIC = re.compile(
    r"\b(?:nearest|closest|that|this|these|those|there|here|it|its|them|they|the\s+one|"
    r"first|second|third|last|other|another|same|via|through|way|again|instead|"
    r"the\s+route|the\s+path|directions)\b|"
    r"最近|那裡|那里|這裡|这里|它|嗰個|那個|呢個|第[一二三]|同一|附近|沿途|路上|經過|经过|改為|改为",
    re.I,
)


def _named_by_user(phrase: str, text: str, context: str = "") -> bool:
    """Can the user's own message account for this place phrase?

    A prompt shows the *shape* of an answer with a real example ("origin": "Block V"), and
    a model that is unsure hands the example back. Observed: "where is the library?"
    answered as a route from Block V, because "Block V" is in the prompt and the library is
    not. The slot resolved to a real place, so every check downstream believed it.

    So a name has to be traceable to the words the user wrote: a word in common, or a word
    close enough to be the same word misspelt ("libarry" -> "library"). The tolerance is
    for spelling, never for substitution -- "Block V" shares nothing with "how do I get to
    the library?". Chinese carries its own test, because it is written without spaces.
    """
    said = lexical.normalise(text) + lexical.normalise(context)
    for word in lexical.normalise(phrase):
        if word in said:
            return True
        if _HAS_CJK.search(word) and (word in text or word in context):
            return True
        if len(word) >= 4 and any(
            len(other) >= 4 and difflib.SequenceMatcher(None, word, other).ratio() >= 0.8
            for other in said
        ):
            return True
    return False


def _validated(slots: Slots, places: PlaceResolver, text: str = "",
               context: str = "") -> Slots | None:
    """Accept a model parse only if the map can back it up.

    Three cheap checks. The intent must be one this dialogue layer implements
    (parse_with_llm already filters that, so this is belt and braces). Every place phrase
    the model reports must be one the resolver recognises: the model may report what the
    user said, it may never create a place. And every phrase must be traceable to the
    user's own words (see _named_by_user), because resolving is not the same as being
    asked about -- the prompt's own example resolves. An endpoint that fails either test
    is dropped, and a route left with no endpoint is refused outright so the grammar gets
    its turn instead.
    """
    if slots.intent not in INTENTS:
        return None
    # A model may decline to understand something; it may not decide that a campus
    # question is out of scope, and it may not decide that a question is about the
    # conversation rather than the campus. Observed: "re-open it" -- one of this app's own
    # commands, recognised by the grammar's REOPEN rule -- came back as out_of_scope and the
    # recovery path never ran; "what can I find here" came back as "help" and the catalogue
    # listing never ran. These two intents are meta: the grammar holds fixed, inspectable
    # rules for both, so the grammar is what gets to say them.
    if slots.intent in ("out_of_scope", "help"):
        return None
    # The model read the intent but not the place ("the canteen is closed" -> close, no
    # closed text). The grammar hands _close the whole sentence and lets the resolver find
    # the place inside it, so do the same instead of asking the user to repeat themselves.
    if slots.intent == "close" and not slots.closed_text and _CLOSED_SAID.search(text):
        slots.closed_text = text.strip()
    # A closure blocks every route through that place, so it is the most expensive thing a
    # model can make up. Keeping one takes two things: the user has to have said something
    # is closed (naming a place is not saying it is closed -- "how do I get from Block V to
    # the library?" names both ends and reports nothing closed), and the place has to be
    # one of the places they named.
    # A slot may be filled from the recent conversation only when this message refers to it
    # ("the nearest one", "make it go via Core P"). Everything else must be in this message.
    window = context if (context and _DEICTIC.search(text)) else ""

    if slots.closed_text and not (
        _CLOSED_SAID.search(text) and _named_by_user(slots.closed_text, text, window)
    ):
        slots.closed_text = ""
    # A stop is optional: an unnameable or unmentioned one is dropped so the plain route
    # still runs. Each one is checked on its own, so a request with three stops keeps the
    # two that name places.
    slots.vias = [phrase for phrase in (slots.vias or [])
                  if _named_by_user(phrase, text, window) and places.resolve(phrase)]
    for name in ("origin_text", "destination_text"):
        phrase = getattr(slots, name)
        if not phrase:
            continue
        if not _named_by_user(phrase, text, window):
            setattr(slots, name, "")    # the model's own idea, not the user's words
            if slots.intent in ("route", "distance"):
                return None
        # A phrase the user did say that the map cannot find is *kept*: the dialogue then
        # answers it with "I do not know X" and the closest names, instead of quietly
        # turning their sentence into a different question about a place they never named.
        # Dropping it is how "Core Y" became a question about Core A.
    # A route needs somewhere to go -- unless it only adds a stop, which re-plans the
    # journey already in progress ("make sure I pass through Core P on the way"). The
    # dialogue holds those endpoints; refusing here sent the sentence back to the grammar,
    # which read it as a fresh pair of places.
    if slots.intent == "route" and not (
        slots.origin_text or slots.destination_text or slots.vias
    ):
        return None
    return slots


def parse(text: str, graph: CampusGraph, client: DeepSeekClient | None,
          places: PlaceResolver, context: list[dict] | None = None) -> Slots:
    """Ask the model first; fall back to the deterministic grammar.

    The model is the parser whenever it is available, because it reads any phrasing and
    not only the sentence shapes somebody enumerated. Its answer is validated by
    _validated() before anything trusts it, and every failure path -- no key, timeout,
    bad JSON, unknown intent, unnameable endpoint -- ends in the same place: the
    grammar answers. A bad model turn therefore costs coverage, never correctness.
    """
    # Which layer answered, and why. "understood by grammar" alone hides four very
    # different situations -- no key, a dead API, the model declining, the model being
    # wrong -- and they need different fixes. The transcript shows this note.
    note = "no model configured"
    if client is not None and client.settings.llm_available:
        route_text, avoid_stairs = stair_request(text)
        candidate = parse_with_llm(route_text or text, graph, client, context)
        if candidate is None:
            note = "model failed (timeout, quota or bad JSON)"
        elif candidate.intent in ("out_of_scope", "help"):
            note = "model abstained"
        else:
            window = " ".join(turn.get("content", "") for turn in (context or []))
            accepted = _validated(candidate, places, route_text or text, window)
            if accepted is not None:
                if not accepted.item and _HAS_CJK.search(text):
                    # The model answered in Chinese ("飯堂幾點開門" -> hours_query, need
                    # 飯堂), and the engine's amenity table is English wording: the grammar's
                    # Chinese service table is the same vocabulary in the other script, so it
                    # supplies the item the model could not.
                    chinese = _chinese_facility(text, places)
                    if chinese is not None and chinese.item:
                        accepted.item = chinese.item
                accepted.avoid_stairs = (
                    avoid_stairs if avoid_stairs is not None else accepted.avoid_stairs
                )
                return accepted
            note = "model slots rejected"
    slots = parse_with_grammar(text, places)
    slots.parse_note = note
    return slots
