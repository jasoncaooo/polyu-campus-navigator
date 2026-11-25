"""The one vocabulary shared by the data pipeline and the dialogue layer.

This file answers three questions, in exactly one place:

  1. which campus-map category becomes which service label?
                                                   -> CAMPUS_SERVICES
  2. how does a user's everyday wording ("toilets", "step-free", "water")
     map onto those canonical capabilities?        -> FACILITY_WORDS, SERVICE_WORDS
  3. what does a point accept, given the category it was published under?
                                                   -> CAMPUS_SERVICES

Why the file exists
-------------------
Those tables used to be copy-pasted into the graph builder and into
`campus_nav/dialogue.py`. The copies drifted, one key was corrupted by a character
encoding bug, and every place filed under it silently dropped off the map. One table,
imported twice, cannot drift.

The rule the tables obey
------------------------
Only what the published category name guarantees. A published "Toilet" is an
accessible toilet only where the record says so; nothing is inferred from a name
that does not claim it. Anything not listed stays unknown, and the knowledge base is
required to say so.
"""
from __future__ import annotations

import re

# ------------------------------------------------------------------- campus map
# Campus-map category -> (label the bot uses, canonical capabilities).
# An empty capability tuple means "this place exists and is navigable, but the bot
# makes no claim about what it offers".
CAMPUS_SERVICES: dict[str, tuple[str, tuple[str, ...]]] = {
    "WaterStation": ("Water Station", ("water_refill",)),
    # The campus map publishes the whole institution, not just its amenities. These
    # are places you would ask for by name ("where is the Academic Registry?"), so they
    # belong on the map; the capability tuple stays empty because the category name
    # says nothing about what the office offers.
    "AcademicOffice": ("Faculty / Department", ()),
    "AdministrativeOffice": ("Office", ()),
    "PublicTransport": ("Bus Stop", ()),
    "LectureTheatre": ("Lecture Theatre", ()),
    "Building": ("Building", ()),
    "CoreLocation": ("Facility", ()),
    "Toilet": ("Accessible Toilet", ("toilet",)),   # only where the POI says accessible
    "AED": ("AED", ("defibrillator",)),
    "Canteen": ("Canteen", ()),
    "Clinic": ("Clinic", ()),
    "LactationRoom": ("Lactation Room", ()),
    "PrayerRoom": ("Prayer Room", ()),
    "Bank": ("Bank", ()),
    "Bookstore": ("Bookstore", ()),
    "Postbox": ("Postbox", ()),
}

# ----------------------------------------------------------------- user wording
# Facility wording -> the capability it asks for. A capability is what the published
# category guarantees (an accessible toilet, a water station, an AED), and it keeps
# ONE lookup path for "which places offer X".
FACILITY_WORDS: dict[str, str] = {
    "aed": "defibrillator", "defibrillators": "defibrillator",
    "defibrillator": "defibrillator", "heart starter": "defibrillator",
    "toilet": "toilet", "toilets": "toilet", "washroom": "toilet",
    "restroom": "toilet", "bathroom": "toilet", "loo": "toilet",
    "water": "water_refill", "water station": "water_refill",
    "refill": "water_refill", "drinking water": "water_refill",
    "fountain": "water_refill", "water fountain": "water_refill",
    "canteen": "canteen", "cafeteria": "canteen", "restaurant": "canteen",
    "clinic": "clinic", "bank": "bank", "atm": "bank",
    "postbox": "postbox", "post box": "postbox", "letter box": "postbox",
    "bookstore": "bookstore", "bookshop": "bookstore",
    "lactation": "lactation room", "nursing room": "lactation room",
    "prayer": "prayer room", "prayer room": "prayer room",
    # Errands and symptoms: the words people use for a *need* rather than a place. These
    # lived in the model's prompt as examples to copy, which worked until the model had to
    # pick between them and a list of amenities it could not see -- so they moved here, next
    # to the list of services the map actually carries, where they can be read and tested.
    "sick": "clinic", "unwell": "clinic", "ill": "clinic", "doctor": "clinic",
    "nurse": "clinic", "medicine": "clinic", "first aid": "clinic", "injured": "clinic",
    "injury": "clinic", "fever": "clinic", "headache": "clinic", "hurt": "clinic",
    "print": "bookstore", "printing": "bookstore", "photocopy": "bookstore",
    "stationery": "bookstore", "notebook": "bookstore", "art supplies": "bookstore",
    "cash": "bank", "money": "bank", "withdraw": "bank", "cashpoint": "bank",
    "eat": "canteen", "food": "canteen", "lunch": "canteen", "dinner": "canteen",
    "breakfast": "canteen", "hungry": "canteen", "snack": "canteen",
    "drink": "water_refill", "thirsty": "water_refill", "bottle": "water_refill",
}

# How each capability is written as a *service label* by the campus map itself. The graph
# stores the publisher's wording ("Catering Outlets", "Clinics", "Aed"), so a lookup on the
# internal name silently finds nothing: "canteen" matched no node at all, and the reply only
# reached the canteens because the meaning search happened to. A join has to be on the words
# that are actually stored.
CAPABILITY_LABELS: dict[str, tuple[str, ...]] = {
    "canteen": ("catering", "canteen"),
    "clinic": ("clinic",),
    "defibrillator": ("aed",),
    "toilet": ("toilet",),
    "water_refill": ("water station",),
    "lactation room": ("lactation",),
    "prayer room": ("prayer",),
    "bank": ("bank",),
    "bookstore": ("bookstore", "bookshop"),
    "postbox": ("postbox",),
    "bus stop": ("bus stop",),
    "lecture theatre": ("lecture theatre",),
}


# One matcher for those words, longest first, on word boundaries.
_AMENITY = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(re.escape(phrase) for phrase in sorted(FACILITY_WORDS, key=len, reverse=True))
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)

# Wording that means "a facility", not "the building it sits in".
SERVICE_WORDS: tuple[str, ...] = (
    "water", "refill", "aed", "defibrillator", "toilet", "washroom", "restroom",
    "canteen", "clinic", "postbox", "lactation", "prayer", "atm",
)


def label_key(text: str) -> str:
    """The form two name lists are joined on: case, spacing and dotted initials folded.

    The builder resolves a building label against the place names in the authored file,
    and the display layer resolves the same label again when it renders a card. Both
    call this, because a join key that differs by one space fails silently -- the same
    way a corrupted key once did.
    """
    return " ".join((text or "").casefold().replace(".", " ").split())


def amenity_in_text(text: str) -> str:
    """The amenity wording these words contain, or "" when they name none.

    This is the boundary the model used to be handed: it was given the list of amenities
    and told to choose from it, so an activity with no label in that list ("I want to have a
    swim") got the nearest wrong one, confidently. The list is the engine's business --
    the engine can also see the fifty-two services the map publishes and the five hundred
    and eighty-nine places it carries, and the model can see neither.

    Longest wording first, so "water station" is not reported as "water", and whole words
    only: "bank" must not match "embankment", and "aed" must not match inside a longer word.
    """
    match = _AMENITY.search(text or "")
    return match.group(0).lower() if match else ""


def facility_capability(word: str) -> str:
    """'washroom' -> 'toilet'; an unknown word is returned unchanged (lowercased)."""
    text = (word or "").strip().lower()
    return FACILITY_WORDS.get(text, FACILITY_WORDS.get(text.rstrip("s"), text))


# ------------------------------------------------------------- drawn geometry
# The "place" column decides how a drawn shape is painted:
#
#   empty            -> a building, extruded
#   "square"         -> flat, painted as a plaza
#   "garden"         -> flat, painted as green
#   "pool"           -> flat, painted as water
#
# Anything else non-empty is still FLAT -- an unknown word must never turn a surface into
# a tower -- and is painted as a plaza.
_PLACE_KIND = {
    "square": "plaza",
    "garden": "green",
    "pool": "water",
}

FLOOR_HEIGHT_M = 3.2


def shape_kind(props: dict, is_line: bool = False) -> str:
    """What a drawn feature is: a block to extrude, or a flat surface."""
    props = props or {}
    if props.get("kind"):
        return str(props["kind"])
    if is_line:
        return "path"
    place = str(props.get("place") or "").strip().lower()
    if not place:
        return "building"
    return _PLACE_KIND.get(place, "plaza")


def feature_height(props: dict) -> float | None:
    """Metres of extrusion, from a height tag or a storey count, or None.

    None is not a failure: the renderer then uses its own hashed height, which
    claims nothing about the real building.
    """
    props = props or {}
    for key in ("height", "building:height"):
        raw = props.get(key)
        if raw not in (None, ""):
            try:
                return float(str(raw).split()[0])
            except (TypeError, ValueError):
                pass
    levels = props.get("building:levels")
    if levels not in (None, ""):
        try:
            return round(float(str(levels).split(";")[0]) * FLOOR_HEIGHT_M, 1)
        except (TypeError, ValueError):
            pass
    return None


def human_category(category: str) -> str:
    """Canonical capability -> the phrase used in a reply."""
    return {
        "water_refill": "drinking water", "toilet": "accessible toilets",
        "defibrillator": "AEDs (defibrillators)",
    }.get(category, category.replace("_", " "))
