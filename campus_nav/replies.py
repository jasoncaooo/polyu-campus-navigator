"""What a turn produces: the reply, and the places and routes it talks about.

These used to live in dialogue.py, which meant hours.py could not describe a reply without
importing the dialogue that owns it. A plain data module has no such problem, and it keeps
the what (a reply) apart from the how (the dialogue that decides one).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .format import FacilityResult
from .nlu import Slots
from .search import RouteResult


@dataclass
class Reply:
    text: str
    route: RouteResult | None = None
    slots: Slots | None = None
    notes: list[str] = field(default_factory=list)
    narrated: bool = False        # True when the model wrote the prose
    # Structured places the reply is talking about, so a map or a card view can
    # render them without re-parsing the prose.
    facilities: list[dict] = field(default_factory=list)
    # Every route worth offering, cheapest first. `route` is routes[0]; the rest are
    # real alternatives the user can pick between.
    routes: list[RouteResult] = field(default_factory=list)
    # What a text-to-speech engine should say. Set explicitly by the handler, because
    # reading the printed table aloud is unlistenable and guessing here would let the
    # audio drift from the screen.
    spoken: str = ""
    # A one-line fact this reply has to lead with, whatever the prose says. A closure is
    # the reason a route changed, so "Closed Catering Outlets." goes first, in one fixed
    # wording -- leaving it in the collapsed details hid the cause of the answer.
    lead_prefix: str = ""
    facility_result: FacilityResult | None = None
