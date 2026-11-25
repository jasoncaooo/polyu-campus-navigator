"""Level B: let the model write the reply, while the numbers stay computed.

The problem this solves
-----------------------
Level A (before this file) printed a template:

    Route: Block Y (Lee Shau Kee Building) -> Core A  Total: 500 m, 1 leg

Correct, but it reads like a receipt and no part of the sentence came from the model.
Level B hands the *finished* route back to the model and asks it to phrase it:

    "It's about 500 m from Block Y to Core A - the quickest way, and the steps come
     at the end."

Every word comes from the API. The numbers still come from A*, because the model is
only describing facts it was given.

Why the model cannot lie about the numbers
------------------------------------------
`grounded_numbers()` compares every number in the model's reply against the set of
numbers that were actually supplied. One invented figure ("about 400 m", "200 steps")
and the reply is thrown away in favour of the deterministic template. This is a real,
demonstrable guard, not a promise in the prompt - prompts can be ignored, code cannot.

Nothing here routes anything. If the model fails, is slow, or invents a number, the
caller keeps the template and the demo continues.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLMResult, DeepSeekClient
from .nlu import has_cjk

NUMBER = re.compile(r"\d+(?:\.\d+)?")

# Phrases that assert proximity to the user. They are only ever legitimate when the user said
# where they are; when nobody did, the engine measured from a map reference and the sentence
# is a claim about a fact that does not exist. The list is deliberately short and literal:
# a false positive costs the model's phrasing (the catalogue speaks instead), never a fact.
_LOCATION_CLAIM = re.compile(
    r"\b(?:right where you are|where you are standing|you(?:'re| are) (?:at|in|near|standing)|"
    r"nearest to you|closest to you|near you|around you|your (?:current )?location|"
    r"from where you (?:are|stand))\b|"
    r"離你最近|离你最近|就在你|你附近|在你附近|你所在|你的位置|你嘅位置|"
    r"距離你|距离你|你而家喺|你現在在|你现在在|由你出發|从你出发",
    re.IGNORECASE,
)

SYSTEM_PROMPTS: dict[str, str] = {
    "en-US": """You are the voice of a campus navigation assistant at the Hong Kong
Polytechnic University. You are given the RESULT OF A ROUTE CALCULATION as JSON.

Write the answer to the user in at most 2 short sentences, plain English, friendly
and concrete.

Hard rules:
- Use ONLY numbers that appear in the JSON. Never estimate, round to a new figure, or
  invent a number. If a number is not in the JSON, do not mention numbers at all.
- Use only the place names that appear in the JSON.
- If "reachable" is false, say clearly that no route exists and mention the single most
  relevant restriction, then stop. Do not promise a route.
- When "kind" is "places", the answer is a *place*, not a journey: "asked_for" is what the
  user wants, "nearest" describes the one place you should name (name, building, floor,
  metres, hours, whether it is open), and "matches" is how many the map carries. Say which
  place it is and how far; mention the count if it is greater than one. Do not describe a
  route and do not list the others.
- If "user_location_known" is false, you do not know where the user is: never say they
  are at, in, or near a place, and never say "you are here". Say the distance is measured
  from "measured_from" instead.
- "walk_minutes", when present, is how long the walk takes at an easy pace; you may say it.
- If a field is absent, do not talk about it.
- No lists, no headings, no markdown. Just the sentence(s).
""",
    # The same instructions in Chinese. They are model instructions, not sentences the app
    # says, so they live beside the guards they describe rather than in copy.py -- but the
    # language they *ask for* is the whole point: this prompt is what makes a Chinese
    # question get a Chinese sentence from the model instead of the catalogue's wording.
    "zh-HK": """你係香港理工大學校園導航助理，負責用自然嘅書面廣東話講出答案。
你收到嘅係路線計算結果（JSON）。

用最多兩句短句回答，語氣親切、具體。

硬性規則：
- 只可以用 JSON 出現過嘅數字。唔准估算、唔准四捨五入成另一個數、唔准自己編。JSON 冇嘅數字就唔好提數字。
- 只可以用 JSON 出現過嘅地方名。
- 如果 reachable 係 false，就清楚講冇路線，講出最相關嘅限制，然後停；唔好承諾有路。
- 如果 kind 係 places，答案係一個地點而唔係行程：asked_for 係用戶搵咩，nearest 係你要講嗰個地點（名、所在建築、樓層、米數、開放時間、開咗未），matches 係地圖上有幾多個。講出嗰個地點同距離，多過一個就提數量；唔好描述路線，亦唔好逐個列出其他。
- 如果 user_location_known 係 false，即係你唔知用戶喺邊：唔可以話佢「就喺」或者「附近」某個地點，亦唔可以講「你而家喺呢度」。要講明距離係由 measured_from 起計。
- 如果有 walk_minutes，可以講步行時間。
- JSON 冇嘅欄位就唔好提。
- 唔用列表、唔用標題、唔用 markdown，只寫句子。
""",
    "zh-CN": """你是香港理工大学校园导航助理，负责用自然的书面中文说出答案。
你收到的是路线计算结果（JSON）。

用最多两句短句回答，语气亲切、具体。

硬性规则：
- 只可以使用 JSON 中出现过的数字。不准估算、不准四舍五入成另一个数、不准自己编。JSON 没有的数字就不要提数字。
- 只可以使用 JSON 中出现过的地名。
- 如果 reachable 是 false，就清楚说明没有路线，指出最相关的限制，然后停止；不要承诺有路。
- 如果 kind 是 places，答案是一个地点而不是行程：asked_for 是用户要找什么，nearest 是你要说的那个地点（名称、所在建筑、楼层、米数、开放时间、是否开放），matches 是地图上有多少个。说出那个地点和距离，多于一个就提数量；不要描述路线，也不要逐个列出其他。
- 如果 user_location_known 是 false，即你不知道用户在哪里：不可以说他“就在”或者“附近”某个地点，也不可以说“你现在在这里”。要说明距离是从 measured_from 起算。
- 如果有 walk_minutes，可以说步行时间。
- JSON 没有的字段就不要提。
- 不用列表、不用标题、不用 markdown，只写句子。
""",
}
SYSTEM_PROMPT = SYSTEM_PROMPTS["en-US"]


@dataclass
class Facts:
    """Everything the model is allowed to talk about. Built from the computed route."""

    kind: str                                  # "route" | "no_route" | "places"
    origin: str
    destination: str
    reachable: bool
    total_metres: float | None = None
    segment_lines: list[str] = field(default_factory=list)
    segment_metres: list[float] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    previous_metres: float | None = None
    walk_minutes: int | None = None       # at an easy pace, computed from total_metres
    blocking: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    option_metres: list[float] = field(default_factory=list)
    # A list of places ("where can I get lunch"): what was asked for, how many the map
    # carries, and the one the engine picked out, with the facts about it.
    need: str = ""
    match_count: int | None = None
    best: dict = field(default_factory=dict)
    reference: str = ""                   # where the distances were measured from
    # The language to answer in. It is part of the facts because it is part of the request:
    # the model cannot pick it (it does not see the browser) and the engine must not guess it.
    language: str = "en-US"
    # False when the user never said where they are, so distances were measured from a map
    # reference. Without this the model read "0 metres from Core P" as "you are at Core P"
    # and said "right where you are" -- a claim about the user that nothing supports.
    user_location_known: bool = True

    # -- what the model sees ------------------------------------------------
    def as_prompt_json(self) -> dict:
        payload = {
            "kind": self.kind,
            "from": self.origin,
            "to": self.destination,
            "reachable": self.reachable,
            "language": self.language,
        }
        if self.reachable and self.total_metres is not None:
            payload["total_metres"] = round(self.total_metres)
            payload["legs"] = len(self.segment_lines)
            if self.walk_minutes is not None:
                payload["walk_minutes"] = self.walk_minutes
            payload["legs_detail"] = self.segment_lines
            if self.previous_metres is not None:
                payload["previous_route_metres"] = round(self.previous_metres)
                payload["difference_metres"] = round(self.total_metres - self.previous_metres)
        else:
            payload["blocking"] = self.blocking
            payload["options"] = self.options
        if self.constraints:
            payload["constraints"] = self.constraints
        if self.kind == "places":
            # No "from"/"to": this answer is about a kind of place, not a journey.
            payload.pop("from", None)
            payload.pop("to", None)
            payload.pop("reachable", None)
            payload["asked_for"] = self.need
            payload["matches"] = self.match_count
            payload["nearest"] = self.best
            payload["user_location_known"] = self.user_location_known
            if self.reference:
                payload["measured_from"] = self.reference
        return payload

    # -- what the model is allowed to say -----------------------------------
    def grounded_numbers(self) -> set[float]:
        """Every number the model may mention: exactly the ones it was given.

        Read out of the payload itself rather than kept in a second list beside it. The list
        drifted the moment the payload grew a field: a correct Chinese sentence quoting the
        canteen's hours and floor was thrown away as "ungrounded number(s): 3, 07, 30, 20"
        because the allowances knew about metres and counts but not about "nearest".
        """
        allowed: set[float] = {0.0, 1.0}
        def add(value) -> None:
            if value is None:
                return
            try:
                allowed.add(round(float(value), 1))
            except (TypeError, ValueError):
                return
        def sweep(value) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, (int, float)):
                add(value)
            elif isinstance(value, str):
                # A numeral inside a published name belongs to the name, not to the answer:
                # "7-Eleven" once made a correct narration look like an invented figure.
                for token in NUMBER.findall(value):
                    add(token)
            elif isinstance(value, dict):
                for item in value.values():
                    sweep(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    sweep(item)
        sweep(self.as_prompt_json())
        # Not in the payload: the raw metre totals and the counts behind its rounded fields.
        for metres in (*self.segment_metres, *self.option_metres, self.total_metres,
                       self.previous_metres, self.walk_minutes, self.match_count):
            add(metres)
        for count in (len(self.segment_lines), len(self.options), len(self.blocking)):
            add(count)
        return allowed


@dataclass
class Narration:
    ok: bool
    text: str = ""
    reason: str = ""
    invented: list[str] = field(default_factory=list)


def claims_unknown_location(text: str) -> bool:
    """Does this sentence say where the user is, or how far something is from them?"""
    return bool(_LOCATION_CLAIM.search(text or ""))


def in_language(text: str, language: str) -> bool:
    """Was this sentence written in the language that was asked for?

    The check is coarse on purpose -- one Chinese character decides Chinese, and the
    absence of any decides English -- because the real content guards are the numbers and
    the place names. This one only stops a fluent sentence in the wrong script.
    """
    chinese = language in ("zh-HK", "zh-CN")
    return has_cjk(text) if chinese else not has_cjk(text)


def grounded_numbers(reply: str, allowed: set[float], tolerance: float = 1.0) -> list[str]:
    """Return the numbers in `reply` that were NOT supplied to the model."""
    offenders = []
    for token in NUMBER.findall(reply or ""):
        try:
            value = float(token)
        except ValueError:
            continue
        if not any(abs(value - candidate) <= tolerance for candidate in allowed):
            offenders.append(token)
    return offenders


class Narrator:
    """Turns a computed route into a sentence. Falls back silently on any problem."""

    def __init__(self, client: DeepSeekClient | None, max_tokens: int = 160) -> None:
        self.client = client
        self.max_tokens = max_tokens

    @property
    def available(self) -> bool:
        return self.client is not None and bool(
            getattr(getattr(self.client, "settings", None), "llm_available", False)
        )

    def narrate(self, facts: Facts) -> Narration:
        if not self.available:
            return Narration(False, reason="no LLM configured")

        prompt = SYSTEM_PROMPTS.get(facts.language) or SYSTEM_PROMPT
        result: LLMResult = self.client.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"RESULT JSON:\n{facts.as_prompt_json()}"},
            ],
            temperature=0.2,          # a little warmth for phrasing, still stable
            max_tokens=self.max_tokens,
            retries=1,                # a demo cannot wait: one try, then the template
        )
        if not result.ok:
            return Narration(False, reason=f"api: {result.error[:80]}")

        text = " ".join(result.text.split())
        if not in_language(text, facts.language):
            # A model that answers in the wrong language is the leak this guard exists for:
            # a Chinese question answered with an English sentence reads as a broken app.
            return Narration(False, reason=f"reply was not written in {facts.language}")
        if len(text) < 12:
            return Narration(False, reason="reply too short")
        if len(text) > 600:
            return Narration(False, reason="reply too long")

        if not facts.user_location_known and claims_unknown_location(text):
            # A prompt rule the model broke in testing: asked for the nearest water station it
            # wrote "离你最近的是 …" although the user never said where they were. Same
            # treatment as an invented number: refuse the wording, keep the answer.
            return Narration(False, reason="claims the user's location, which is unknown")
        invented = grounded_numbers(text, facts.grounded_numbers())
        if invented:
            # the model made up a figure: refuse the wording, keep the calculation
            return Narration(False, reason=f"ungrounded number(s): {', '.join(invented[:4])}",
                             invented=invented)

        return Narration(True, text=text, reason=f"{result.model} {result.latency_ms} ms")
