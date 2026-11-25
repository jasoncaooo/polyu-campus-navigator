"""Every sentence this app says, in one place, keyed by language.

Why a catalogue
---------------
Sentences used to live in `if lang == "zh-HK": … elif "zh-CN": … else:` chains, in three
different modules. Three arms to keep in step is the same hazard as two lists that must agree
-- and this project has already been bitten by that once (`_FACILITY` vs `FACILITY_WORDS`
drifted, so a need understood by one parser was invisible to the other).

So: the *structure* of a sentence (which numbers, which names) stays in code, and the words
live here as data. Adding a language is adding a column, not editing branches. A missing
translation falls back to English rather than raising: a wrong-script sentence is better than
no answer at all.

Placeholders are `{name}` fields filled by `t()`. Note that the leading space in an English
suffix (" This route avoids …") is deliberate: Chinese runs on without a space, and putting
the spacing in the data keeps the concatenation identical in every language.
"""
from __future__ import annotations

COPY: dict[str, dict[str, str]] = {
    # -- replies that are not about a place ----------------------------------
    "speech.greeting": {
        "en-US": "Hi! Tell me where you are and where you want to go -- or ask me to find "
                 "something ({examples}).",
        "zh-HK": "你好！講低你喺邊、想去邊，或者叫我幫你搵嘢（例如{examples}）。",
        "zh-CN": "你好！告诉我你在哪里、想去哪里，或者让我帮你找设施（例如{examples}）。",
    },
    "facility.reference_note": {
        "en-US": "Distances are measured from {reference} -- tell me where you are and I "
                 "will re-rank them.",
        "zh-HK": "距離係由{reference}起計——同我講你喺邊，我會按你嘅位置重新排序。",
        "zh-CN": "距离从{reference}起算——告诉我你在哪里，我会按你的位置重新排序。",
    },
    "speech.thanks": {
        "en-US": "Any time. Where next?",
        "zh-HK": "唔使客氣。跟住想去邊？",
        "zh-CN": "不客气。接下来想去哪里？",
    },
    "speech.reset": {
        "en-US": "Cleared. Where would you like to go?",
        "zh-HK": "已清除。你想去邊度？",
        "zh-CN": "已清除。你想去哪里？",
    },
    "speech.out_of_scope": {
        "en-US": "I can find walking routes on the PolyU campus, point you to facilities "
                 "(AEDs, water, accessible toilets, canteens) and quote their opening hours. "
                 'Try: "how do I get from Block Y to the library?" or '
                 '"where is the nearest AED?"',
        "zh-HK": "我可以幫你搵理大校園嘅步行路線、指你去設施（AED、飲水機、暢通易達洗手間、"
                 "飯堂），同埋查開放時間。試下講：「由Y座去圖書館點行？」或者「最近嘅AED喺邊？」",
        "zh-CN": "我可以帮你找理大校园的步行路线、指你去设施（AED、饮水机、无障碍洗手间、"
                 "食堂），并查询开放时间。试试说：“从Y座到图书馆怎么走？”或者“最近的AED在哪里？”",
    },
    "speech.help": {
        "en-US": "I can do three things:\n"
                 '  1. Route between campus places, e.g. "from Block Y to the library"\n'
                 '  2. Re-plan around closures: "the link between Core C and Core D is closed"\n'
                 '  3. Find facilities and their opening hours: "is the canteen open?"\n'
                 "Ask in English or Chinese. If I do not know something, I will say so.",
        "zh-HK": "我可以做三件事：\n"
                 "  1. 計畫校園內嘅步行路線，例如「由Y座去圖書館點行？」\n"
                 "  2. 遇到封路時重新計畫：「Core C同Core D之間嗰段封咗」\n"
                 "  3. 搵設施同開放時間：「飯堂開咗未？」\n"
                 "用中文或者英文問都可以。唔知嘅話我會直接講唔知。",
        "zh-CN": "我可以做三件事：\n"
                 "  1. 规划校园内的步行路线，例如“从Y座到图书馆怎么走？”\n"
                 "  2. 遇到封路时重新规划：“Core C和Core D之间那段封了”\n"
                 "  3. 找设施和开放时间：“食堂开了吗？”\n"
                 "用中文或者英文提问都可以。不知道的话我会直说。",
    },
    # -- the route, spoken ---------------------------------------------------
    "speech.route": {
        "en-US": "From {origin} to {destination} is about {metres} metres.",
        "zh-HK": "由{origin}行去{destination}，大約 {metres} 米。",
        "zh-CN": "从{origin}步行到{destination}，约 {metres} 米。",
    },
    "speech.route.avoids_stairs": {
        "en-US": " This route avoids all stairs marked on the map.",
        "zh-HK": "已避開地圖標示嘅樓梯。",
        "zh-CN": "已避开地图上标记的楼梯。",
    },
    "speech.route.stairs": {
        "en-US": " The route includes stairs.",
        "zh-HK": "途中要行樓梯。",
        "zh-CN": "途中需要走楼梯。",
    },
    "speech.route.no_stairs": {
        "en-US": " No stairs are marked on this route.",
        "zh-HK": "地圖冇標示呢條路線有樓梯。",
        "zh-CN": "地图上这条路线没有标注楼梯。",
    },
    # -- no route, spoken ----------------------------------------------------
    "speech.unreachable.stairs": {
        "en-US": "I couldn't find a route under the current restrictions that avoids all "
                 "mapped stairs.",
        "zh-HK": "避開地圖標示嘅樓梯之後，按而家嘅條件搵唔到路線。",
        "zh-CN": "避开地图上标记的楼梯后，按当前条件找不到路线。",
    },
    "speech.unreachable.other": {
        "en-US": "I could not find a route with those restrictions. The options are on screen.",
        "zh-HK": "按而家嘅限制，搵唔到路線。你可以睇下畫面上嘅其他選擇。",
        "zh-CN": "按当前条件找不到路线。你可以查看屏幕上的其他选项。",
    },
    # -- a facility answer ---------------------------------------------------
    "place.with_location": {
        "en-US": "{name} ({location})",
        "zh-HK": "{name}（{location}）",
        "zh-CN": "{name}（{location}）",
    },
    "facility.overview.none": {
        "en-US": "No facilities are recorded on this map.",
        "zh-HK": "地圖上暫時冇設施記錄。",
        "zh-CN": "地图上暂无设施记录。",
    },
    "facility.overview.list": {
        "en-US": "The map has {categories} and other facilities. Which kind do you want?",
        "zh-HK": "地圖上有{categories}等設施。你想搵邊一類？",
        "zh-CN": "地图上有{categories}等设施。你想找哪一类？",
    },
    "facility.no_match": {
        "en-US": 'The campus map has no record of "{subject}".',
        "zh-HK": "校園地圖搵唔到「{subject}」嘅記錄。",
        "zh-CN": "校园地图上没有找到「{subject}」的记录。",
    },
    "facility.unknown_reference": {
        "en-US": 'I could not place "{subject}". Where should I start from?',
        "zh-HK": "未能確認你講嘅「{subject}」。你想由邊度開始搵？",
        "zh-CN": "没能确认你说的「{subject}」。你想从哪里开始找？",
    },
    "facility.choice_missing": {
        "en-US": "Which facility do you mean?",
        "zh-HK": "你指邊個設施？請講個名，或者揀畫面上嘅編號。",
        "zh-CN": "你指哪个设施？请说出名称，或选择列表中的编号。",
    },
    "facility.ask_category": {
        "en-US": "Which kind of facility? A water station or a toilet, for example.",
        "zh-HK": "你想搵邊類設施？例如飲水機或者洗手間。",
        "zh-CN": "你想找哪类设施？比如饮水机或洗手间。",
    },
    "facility.no_route": {
        "en-US": "The map has no route to {name}.",
        "zh-HK": "地圖未有去{name}嘅路線。",
        "zh-CN": "地图上没有通往{name}的路线。",
    },
    "facility.hours.one": {
        "en-US": "{name}: {schedule}{current}.",
        "zh-HK": "{name}，{schedule}{current}。",
        "zh-CN": "{name}，{schedule}{current}。",
    },
    "facility.hours.open_now": {
        "en-US": ", open now according to the timetable",
        "zh-HK": "，按時間表而家開放",
        "zh-CN": "，按时间表现在开放",
    },
    "facility.hours.closed_now": {
        "en-US": ", closed now according to the timetable",
        "zh-HK": "，按時間表而家休息",
        "zh-CN": "，按时间表现在不开放",
    },
    "facility.hours_options": {
        "en-US": "{total} places match, including {names}. Which timetable do you want?",
        "zh-HK": "有 {total} 個符合條件嘅地點，包括{names}。你想查邊個嘅開放時間？",
        "zh-CN": "有 {total} 个符合条件的地点，包括{names}。你想查哪个的开放时间？",
    },
    "facility.from_reference": {
        "en-US": "The nearest one is {name}, about {metres} metres from {reference}.",
        "zh-HK": "由{reference}起計，最近嘅係{name}，直線距離大約 {metres} 米。",
        "zh-CN": "从{reference}算起，最近的是{name}，直线距离约 {metres} 米。",
    },
    "facility.from_reference.selected": {
        "en-US": "{name} is about {metres} metres from {reference}.",
        "zh-HK": "由{reference}起計，{name}，直線距離大約 {metres} 米。",
        "zh-CN": "从{reference}算起，{name}，直线距离约 {metres} 米。",
    },
    "facility.same_place": {
        "en-US": "The nearest one is {name}, mapped at the same spot as {reference}.",
        "zh-HK": "最近嘅係{name}，地圖上同{reference}標喺同一個位置。",
        "zh-CN": "最近的是{name}，地图上与{reference}标在同一位置。",
    },
    "facility.same_place.selected": {
        "en-US": "{name} is mapped at the same spot as {reference}.",
        "zh-HK": "{name}，地圖上同{reference}標喺同一個位置。",
        "zh-CN": "{name}，地图上与{reference}标在同一位置。",
    },
    "facility.under_metre": {
        "en-US": "{name} is less than a metre from {reference} in a straight line.",
        "zh-HK": "{name}同{reference}嘅直線距離唔夠一米。",
        "zh-CN": "{name}与{reference}的直线距离不到一米。",
    },
    "facility.found": {
        "en-US": "The map has {name}.",
        "zh-HK": "地圖上搵到{name}。",
        "zh-CN": "地图上找到了{name}。",
    },
    # -- the timetable -------------------------------------------------------
    "hours.day.today": {"en-US": "today", "zh-HK": "今日", "zh-CN": "今天"},
    "hours.day.public_holiday": {
        "en-US": "public holiday", "zh-HK": "公眾假期", "zh-CN": "公众假期",
    },
    # A day is a name, not a numeral: handing the renderer 一二三四五六 for English printed
    # "二08:00-23:00", which is not a sentence in any of the three languages.
    "hours.day.mon": {"en-US": "Monday", "zh-HK": "星期一", "zh-CN": "星期一"},
    "hours.day.tue": {"en-US": "Tuesday", "zh-HK": "星期二", "zh-CN": "星期二"},
    "hours.day.wed": {"en-US": "Wednesday", "zh-HK": "星期三", "zh-CN": "星期三"},
    "hours.day.thu": {"en-US": "Thursday", "zh-HK": "星期四", "zh-CN": "星期四"},
    "hours.day.fri": {"en-US": "Friday", "zh-HK": "星期五", "zh-CN": "星期五"},
    "hours.day.sat": {"en-US": "Saturday", "zh-HK": "星期六", "zh-CN": "星期六"},
    "hours.day.sun": {"en-US": "Sunday", "zh-HK": "星期日", "zh-CN": "星期日"},
    # "Tuesday 08:00-23:00" wants a space; 星期二08:00至23:00 does not.
    "hours.day.joiner": {"en-US": " ", "zh-HK": "", "zh-CN": ""},
    "hours.closed": {"en-US": "closed", "zh-HK": "休息", "zh-CN": "不开放"},
    "hours.not_published": {
        "en-US": "{label}: times are not published",
        "zh-HK": "{label}嘅開放時間未有公布",
        "zh-CN": "{label}的开放时间未公布",
    },
    "hours.none": {
        "en-US": "The campus map publishes no opening hours",
        "zh-HK": "校園地圖未有公布開放時間",
        "zh-CN": "校园地图未公布开放时间",
    },
    "hours.range": {"en-US": "{start}-{end}", "zh-HK": "{start}至{end}", "zh-CN": "{start}至{end}"},
    "hours.list_separator": {"en-US": ", ", "zh-HK": "、", "zh-CN": "、"},
    # -- the detail panel ----------------------------------------------------
    "details.numbered": {"en-US": "{index}. {label}", "zh-HK": "{index}. {label}",
                         "zh-CN": "{index}. {label}"},
    "details.metres": {"en-US": "{label} · {metres} m", "zh-HK": "{label} · {metres} 米",
                       "zh-CN": "{label} · {metres} 米"},
    "details.with_timetable": {"en-US": "{label} · {times}", "zh-HK": "{label} · {times}",
                               "zh-CN": "{label} · {times}"},
    "details.count": {"en-US": "{label} · {count}", "zh-HK": "{label} · {count}",
                      "zh-CN": "{label} · {count}"},
    "details.more": {
        "en-US": "Showing the first {shown}, of {total}.",
        "zh-HK": "顯示頭 {shown} 個，共 {total} 個。",
        "zh-CN": "显示前 {shown} 个，共 {total} 个。",
    },
}

LANGUAGES = ("en-US", "zh-HK", "zh-CN")


def t(key: str, language: str, **fields: object) -> str:
    """The sentence for `key` in `language`, with `fields` filled in.

    An unknown key raises -- a typo in a key is a bug, not a language problem -- but a
    missing language falls back to English, because a sentence in the wrong script is a
    better answer than silence.
    """
    row = COPY.get(key)
    if row is None:
        raise KeyError(f"no such message: {key}")
    # Present-but-empty is a translation, not a missing one: 星期二08:00 wants no joiner,
    # and silently borrowing English's space from a falsy check is the wrong fallback.
    text = row[language] if language in row else row.get("en-US")
    if text is None:
        raise KeyError(f"{key} has no {language} or en-US text")
    return text.format(**fields) if fields else text
