# PolyU Campus Navigator

**COMP4431 Artificial Intelligence (2026/27) — Group Project · Group 21**

A chatbot that guides you through navigating around PolyU campus, and points you to facilities
you might need (e.g. AEDs, water stations, accessible toilets, canteens, or clinics). You can
communicate with it in three language: English, Mandarin or Cantonese. It also supports speech
recognition allowing you to ask for directions hands-free while walking around campus.

## Run it

```bash
python -m campus_nav.web --open      # web app  ->  http://127.0.0.1:8000
```

Python 3.10+ to run it — the Python code uses the standard library only, so there is
nothing to `pip install`. **It runs with an empty `.env`**: no keys means the
deterministic grammar parses and the built-in scorer resolves place names. Copy
`.env.example` and add keys to switch individual layers to their model-backed path:

```bash
python tools/build.py embed --recreate   # fill the vector index with every place name
```

The interface ships as a committed bundle, so no build step is needed to demo it.

### The layers, and what happens without a key

Each layer has a deterministic twin, so a missing key or a dead network costs coverage,
never the application. The server prints which path is live at startup, and
`GET /api/status` exposes the same under `places.mode`.

| layer | primary | fallback | without it |
|---|---|---|---|
| conversation | tool-calling assistant (`agent.py`) | grammar dialogue (`dialogue.py` + `nlu.parse_with_grammar`) | fewer phrasings understood; nothing breaks |
| name identity | embeddings + vector index (`place_index.py`) | built-in word scorer (`lexical.py`) | exact-ish names still resolve; no paraphrases |
| facts | BM25 over 561 campus-map records (`kb.py`) | — (always local) | — |
| routing | A* + UCS over the walkway graph (`search.py`) | — (always local, never a model) | — |
| wording | the assistant's own sentence | templates in `dialogue.py` | plainer sentences |

**The model is given no vocabulary at all.** Three vocabularies exist in this app — the 589
places on the map, the 52 services it publishes, and the words people use for what they want
("a swim", "somewhere to eat", "feel sick") — and only the engine can see any of them. So the
assistant is given tools, not a catalogue:

```
message ──> model chooses tools ──> engine executes and returns facts
                 │                        │
        search(name) ─────────────────────┤ real ids, names, buildings (names or descriptions)
        ask(question) ────────────────────┤ the turn ends with one question
        path(src, dest, via) ─────────────┤ A* route, metres, minutes
        facilities(item, near) ───────────┤ service search, hours, distances
        hours(place, when) ───────────────┤ quoted timetable, or "not published"
        close(place) ─────────────────────┴ closure + re-plan

without a model: grammar slots ──> the same engine functions
```

That split is the whole design. Earlier the model was handed the amenity list and told to
choose from it, so any need without a label got the nearest wrong one — "I want to have a
swim" became *water station*, confidently. It is not that the model is weak: it was asked a
question whose answer requires the campus, and it cannot see the campus. The assistant may
therefore report *what the user said* and choose *which tool to call*; it may never invent a
place id, a distance or a timetable, and every id it passes to `path()` is validated against
the graph first.

### Where the bilingual code lives

Chinese is not a special case bolted onto English logic; it is data in four places, and no
module branches on language to decide *what to do*:

| layer | file | what is bilingual there |
|---|---|---|
| reading | `nlu.py` | the grammar patterns (one regex, alternation) and the day words |
| wording | `copy.py` | every sentence, keyed by language, English mandatory as the fallback |
| naming | `local_names.py` | place, building, floor and service labels in both scripts |
| shapes | `format.py` | nothing — it returns rows, days and states, never prose |

`render.py` composes the four into a sentence. The test that keeps this honest is
`test_render.py`: every message must have English, a missing language falls back to it, and
an unknown key is a bug rather than a translation gap.

### What the conversation remembers, and for how long

Three separate stores, each with a different job and a different lifetime:

| store | holds | lives in | cleared by |
|---|---|---|---|
| `SessionState` | origin, destination, stops, closures, stair preference, last list | the server, one per browser cookie (32 at most; the oldest browser is dropped first) | "Clear", "reset", or the browser evicting |
| parser history | the last 6 messages (3 exchanges, 400 chars each) | the same server-side object, trimmed on every turn | the same |
| transcript | the visible conversation | that browser's `localStorage`, versioned | the Clear button, or the error panel |

The parser window is deliberately short, and it is *not* the memory: the journey travels as
a one-line system message built from `SessionState` on every turn ("CURRENT JOURNEY … origin
Block V; destination the library; stops in order: Core P"). That is a fact rather than prose,
so it survives a long aside without the conversation window having to grow, and the model may
repeat those endpoints but never invent one. Resetting clears the state *and* the history —
clearing only the state once left the parser able to read a conversation the user had erased.

Three rules make this safe rather than optimistic. They are code, not prompt text, because
a prompt can be ignored and these cannot.

- **The model may report a place, never create one — and never report one the user did not
  say.** `nlu._validated()` keeps a slot only if the resolver recognises it *and* every word
  of it is traceable to the message (`nlu._named_by_user`, with tolerance for spelling but
  not for substitution). Both halves are load-bearing. Observed failure: the prompt's own
  example `"origin": "Block V"` came back as the answer to "where is the library?" — and
  since Block V is a real place, every check downstream agreed with it. The prompt now
  carries no place catalogue at all: the resolver names places, so the model only has to
  copy the user's words.
- **A name has to be in the words.** Dense place matching always returns neighbours, so a
  single unknown word lands on whatever is nearest: "want" scored highest for "Block W
  (Industrial Centre)" and "Hi" for the HTI department, and both became "which place did you
  mean?". A one-word query must therefore appear in a published name; a phrase of two or
  more words may still match on meaning, which is what the embeddings are for ("somewhere to
  fill my bottle"). Building codes are matched deterministically instead — "Core P", "Block
  VA", "Wing FJ" resolve exactly or not at all, so "Core Y" answers "I do not know Core Y,
  did you mean Block Y?" instead of offering four wrong cores.
- **Grammar is structure, not vocabulary.** Each rule below was written after a bug and
  each is a statement about how the language works, so it holds for sentences nobody
  enumerated: a destination needs a *request* in front of its "to" (`get/go/walk/route …
  to`), not just the preposition, because "I want to make sure I pass through Core P"
  otherwise ends at Core P; a bare "A to B" is only a pair when **both** sides resolve; a
  stop clause at either end of the sentence becomes the stop (`via`, `stop at`, `pass
  through`, with "on the way" cut off it); a specific rule beats an incidental name mention,
  so "what facilities are there" is a request for the list and not a walk to a cleanroom
  called "… Facility"; English plurals fold to the published singular ("toilets" →
  "Toilet (Block V)"); and the grammar asks the engine's amenity table when its own short
  word list misses — "lunch" is catering, "sick" is a clinic — so a need is understood with
  or without the model. `tests/test_robustness.py` exists to keep them properties rather
  than patches.
- **A closure must have been said.** Closing a place blocks every route through it, so a
  closure is kept only when the message contains closure words (both scripts) and names the
  place. An invented closure refuses the walk somebody actually asked for.
- **A number must have been computed.** A model sentence containing a figure the engine did
  not produce is discarded by `narrate.grounded_numbers()`; digits inside a place name
  ("7-Eleven") count as the name, not as a figure.

The graph is read from `data/processed_data/campus_graph.json` **once, at startup**: after
`python tools/build.py graph` you must restart the server, or the browser keeps being served
the graph the old process is holding in memory. (A graph built before the `category` field
existed looks like a broken legend — every chip matches nothing, and every place falls into
"main locations".)

Getting the restart wrong in the other direction is now impossible to get wrong quietly:
the server refuses a port that is already listening (`web._Server`, which turns address
reuse off on Windows). Windows otherwise lets a second process bind a live port, and then an
hour-old copy answers some of the requests — indistinguishable, from the outside, from a fix
that did not work.

To change the interface (Node 18+), the source lives in `web/` and builds into
`campus_nav/static/`:

```bash
cd web && npm install
npm run build      # writes campus_nav/static/ — commit the result
npm run dev        # hot reload, proxying /api to the Python app on :8000
```

Components are styled with Tailwind utilities and take their colours from the palette
block in `web/src/styles.css`; icons come from [Lucide](https://lucide.dev). That
stylesheet keeps only what a utility cannot express: the palette, the avatar's paint and
state animations, and the map layers the drawing code creates by class name.

With a DeepSeek key in `.env` the model drives each turn through the tools in
`campus_nav/agent.py` (`search`, `ask`, `path`, `facilities`, `hours`, `close`, `reset`) and
phrases the answer; it never chooses the route and never states a fact that was not
computed. Without a key the deterministic grammar parses the message and calls the same
engine functions (`dialogue.py`), and the app behaves identically — it simply understands
fewer ways of asking.

Place words are matched by embedding the phrase and reading the nearest place names back
from the vector index (`campus_nav/place_index.py`); when that index is unconfigured or
unreachable, `campus_nav/lexical.py` scores names directly against the graph instead.

Two matching rules keep the nearest-neighbour list from answering a question nobody asked.
In English, a whole-word match on a published name beats cosine proximity
(`PlaceIndex.NAME_EVIDENCE`): "library" belongs to "Block L (Pao Yue-Kong Library)", while its
nearest neighbours are a bookshop, a café and an AED. In Chinese, names are head-final inside
a segment — "L座(包玉剛圖書館)" ends with 圖書館 — while a qualifier comes first, so "圖書館咖啡廳"
is the café. `place_index.cjk_named()` applies that, and then prefers the building over the
things standing inside it.

## Team project and provenance

Portfolio snapshot of the COMP4431 Group 21 project. Original team repository: https://github.com/cy-katryn/CampusNavigator . This is a collaborative project; this repository does not imply sole authorship. 

### Personal contributions — CAO Wei

- Developed speech-to-text (STT) and text-to-speech (TTS) features, enabling voice input and spoken responses in the campus navigation assistant.
- Created the project presentation slides and delivered the project presentation.

The routing engine and broader AI system are team project features; the personal contributions listed above describe my own role.

Resume wording: [Portfolio and resume description](PORTFOLIO-2026-11-25-v2.md).

## What it does

**Map** — extruded building blocks in PolyU red with grey roofs, paved squares and lawns
painted flat, plus the place markers and facility pins that the walkways connect. Pins,
letters and line weights are drawn in map units, so they grow with the map as you zoom in,
exactly as the buildings do. Routes appear only while you are navigating: up to three ways
there, each in its own colour, with the chosen one drawn solid over the others. The legend
in the bottom-left corner switches each layer on, grouped the way the campus map publishes
its places: **Places** (blocks & buildings, main locations, and the walkway links
themselves), **Amenities** (public transport, lecture theatres, AEDs, water stations,
toilets, catering, the clinic, lactation and prayer rooms) and **Offices** (faculties &
schools, administrative offices).

**Chat** — asks in plain language for a route between two places, re-plans when you report a
link closed, lists the nearest facility of a kind, quotes opening hours from the published
table, and abstains when the data has nothing. A need works too: *"I want to have a swim"*
answers with Michael Clinton Swimming Pool and how far it is. The amenity vocabulary the
model chooses from is a list of *services*, so an activity would be squeezed into the nearest
one (a swim became the nearest water fountain); instead the places themselves are searched by
meaning, and only matches close to the best one are offered. When there is more than one way there, the
reply lists them — pick one and the map draws it. Stops work the same way: *"from Block Y to
Core A, through Core P and the library"* is planned leg by leg (four searches, three legs) and
Remembered across turns, too: "along the way, pass through Block X" extends the walk in
progress, "I also want a quick breakfast on the way" asks the engine to find the nearest
catering outlet with the smallest detour and says what that cost — and "no stops" puts the
direct route back. Each is offered as up to three choices, made by swapping **one** leg for
its next-best route
rather than by taking the cross product of every leg. The reply says which legs it joined and
the map draws each option separately.

```
you> how do I get from Block V to the library?
bot> Route: Block V (Jockey Club Innovation Tower) -> Block L (Pao Yue-Kong Library)
     Total: 206 m over 1 leg, 1 flight of steps (about 3 min at an easy pace)
     Search (astar): expanded 64 nodes, peak frontier 34, 0.39 ms, heuristic=euclidean
     There are 2 other ways there:
       2. 221 m over 1 leg, 1 flight of steps
       3. 251 m over 1 leg, 2 flights of steps
     You will pass: Block X, Block V (Jockey Club Innovation Tower), Block VA, Core J.
```

The blocks named on the walk are the campus map's **Buildings / Cores / Wings / Blocks**
only — not the squares, galleries, pools and sports centres it files under "Main
Locations", and nothing inside a building. Each one is marked on the route line at the metre
mark where you pass it, which is the same number the engine measured.

Open the reply's **route breakdown** toggle and it shows the search itself: the first eight nodes A*
expanded, each with `g` (metres walked so far), `h` (straight-line distance still to go) and
`f = g + h`. That is the receipt behind "the model did not choose this route" — a source and
destination dropdown has no such list to show, because a dropdown has already decided which
places exist and how they connect.

Follow-ups need no repetition — "how far is it?", "the library is closed", "re-open it",
"where is the nearest AED?" continue the same conversation. Each *browser* has its own
conversation (a cookie keys it, so two people do not share a state), and the transcript is
kept in that browser's `localStorage`, so a refresh keeps both the words and the line on the
map. A phrase naming several places gets the candidates listed; a phrase naming nothing gets
the closest names on the map suggested instead of a guess.

## Using voice navigation

Choose English, 普通话 or 廣東話 and press **Speak**. The transcript lands in the
composer and **Send** plans the route, exactly as typing it would. When the sentence names
two places, one quiet line says what was heard — *Heard: Block Y → Core A · change* — because
speech recognition mishears building codes and that is the only spot where a misheard block
can be corrected; **change** opens the two fields, each with a searchable list of campus
places. The fields open on their own only when a name is unanswered. For example, try
“从图书馆到赛马会创新楼，不走楼梯” or “邊度有飲水機？”.
With **Auto-read** on, the navigator reads replies in the selected language.
**Replay** repeats the latest reply. Playback requires an appropriate system voice.

For offline recognition, install the local recognizer once on the computer running
the Python server. Setup requires Git, CMake, curl and a C/C++ compiler. It builds
whisper.cpp and downloads and verifies a roughly 574 MB multilingual model.

```bash
python tools/setup_voice.py
python -m campus_nav.web --open
```

Open the page on localhost or HTTPS and allow microphone access. Each recording
can last up to 30 seconds. Without the local model, voice input uses the browser's
speech recognition when supported, which may require internet access. To reuse
an installation, set `WHISPER_CLI` and `WHISPER_MODEL` to its executable and
model paths before starting the server.

## Data

Three authored inputs, one generated file the app reads:

| File | Holds |
|---|---|
| `data/authored/campus_buildings.geojson` | the structure polygons — the 2.5D map |
| `data/authored/campus_paths.geojson`  | the walkways, one LineString per stretch |
| `data/authored/campus_pois.geojson`   | every POI the PolyU campus map publishes |
| `data/processed_data/campus_graph.json` | **generated** — the only file the app reads |

```bash
python tools/build.py graph          # rebuild after editing any GeoJSON
python tools/build.py embed --recreate   # embed every place name into the vector index
```

`tools/build.py` is deterministic arithmetic on what you drew. It measures along your lines,
never estimates, and stops with a message naming the place when something is missing rather
than inventing a route. To change the map: edit a GeoJSON in `data/authored/`, run
`python tools/build.py graph`, reload the page.

## Tests

```bash
python -m pytest tests -q      # 109 tests + 24 subtests: no network, no keys, no server
```

| File | What it pins |
|---|---|
| `tests/test_nlu_grammar.py` | the grammar's phrasings, one pinned sentence per rule, and the intent arbitration order |
| `tests/test_llm_path.py` | the model path against a stubbed client: slot echo, id→name, closure evidence, every fallback, grounded numbers |
| `tests/test_dialogue.py` | walk time, an ambiguous closure that names its candidates, close-then-re-open, greetings, "Core Y" → "did you mean Block Y", "did you mean" on a misspelt place |
| `tests/test_place_index.py` | name resolution in both modes: the name-evidence gate, one answer per name, the lexical band |
| `tests/test_render.py` | the catalogue and the renderer: English present for every message, a missing language falls back, an unknown key is a bug, each arm says what it was given |
| `tests/test_robustness.py` | the properties, over phrasings written *after* the fixes and unlike them: filler names no place, a code resolves exactly or not at all, a preposition is not a route, a trailing stop is a stop, an overview question is not a journey, and a need is answered by places that mean the same thing |
| `tests/test_cjk_names.py` | Chinese name evidence: head-final matching inside a bracket, place before facility |
| `tests/test_landmarks.py` | landmarks along a route: the 20 m band, order along the walk, the cap, buildings only |
| `tests/test_off_network.py` | published-but-unreachable places stay in the graph and are refused in words |

## Layout

| Path | What |
|---|---|
| `campus_nav/` | the application: graph, A*/UCS search, dialogue, NLU, knowledge base, place index (embeddings + vector store) and its offline scorer, web server |
| `campus_nav/copy.py` | every sentence the app says, as data, keyed by language — adding a language is a column, not a branch |
| `campus_nav/format.py` | facility and timetable *shapes* with no prose: which rows, which day, what state |
| `campus_nav/render.py` | what gets said about a reply: narration if there is any, else the catalogue, else the reply's own words. Also the only owner of facility and timetable sentences — those used to be a second, Chinese-only layer |
| `campus_nav/replies.py` | the reply types (`Reply`), so the behaviours that build them do not import each other |
| `campus_nav/hours.py` | opening hours: quote the published table, list the candidates, or say there is no table — the one behaviour that consults the knowledge base instead of the router |
| `tools/replay.py` | replay a fixed prompt list and diff the **facts** of the answers — the safety net for refactoring (timings are ignored) |
| `web/` | the interface source: React 18 + TypeScript, styled with Tailwind, icons from Lucide, built by Vite |
| `campus_nav/static/` | **generated** — the built interface the server serves |
| `tools/build.py` | build the graph, embed the place names |
| `data/raw_data/` | the scraped JSON file the knowledge base reads |
| `data/authored/` | the GeoJSON you draw and edit |

## Known limitations

- **Opening hours are joined by title, because the two ids do not correspond.** The
  published tables cover 33 of 561 records, and a record is matched to a place by the name
  both sources print — which holds for 326 of the 329 places. The three that no published
  title matches are unreachable, and a scrape that stopped quoting the id used by the map is
  not repairable from this side. (There used to be a node-id join here; it never matched a
  single record, so it has been removed rather than left looking functional.)
- **No indoor geometry.** A facility inside a building is reached through the building; the
  reply gives the floor, not a corridor. Lifts and internal stairs are not modelled, and
  there are no doorway nodes, so every walk ends at the place itself.
- **The walkway network has gaps.** Footways that pass metres apart without a shared node
  force detours, and a place standing more than 120 m from any walkway is attached by one
  straight connector rather than a path.
- **The fallback tiers are weaker than the models they replace.** Without an LLM the app
  understands only phrasings the grammar enumerates; without the vector index a place must
  be named roughly as the map publishes it. Both are stated in the reply when they matter
  (`/api/status` reports which mode is live) rather than silently degrading.
- **Alternatives on a stop journey can be a metre apart.** They are different lines around the
  same buildings — the difference a walker cares about is usually the steps and the side of the
  building, and those are printed. A direct route offers the same three-choice list it always
  did.
- **A vague wish returns closest matches, not a category.** "Somewhere quiet to study" can
  offer a bookshop before a plaza, because the ranking is embedding similarity. The reply
  says "closest places for …" rather than pretending otherwise.
- **A saved conversation can outlive the build that wrote it.** The transcript is versioned
  and each restored reply is checked against the shape this build draws; a mismatch is
  dropped, and anything that still throws while rendering shows a recovery panel (with a
  button that clears the saved transcript) instead of a blank page.
- **Some published places are not on the walkway network.** Halls, the West Kowloon campus
  and a few schools on other sites sit 150–1600 m from the nearest drawn walkway. They stay
  in the graph — drawn with a dashed ring — and the reply says plainly that no walking route
  can reach them, instead of failing as if the search had broken.
- **Positions are published positions, not a survey.** One campus-map coordinate per building
  is shared by every POI inside it — the floor field is what distinguishes them.
- **Distances are modelled.** Edge length is measured along the drawn walkway; the last few
  metres from a door to the path is a straight connector. The quoted walk time is that
  distance divided by 1.25 m/s — an easy pace for anybody, not a personal one.

## Attribution

- Walkway geometry: © OpenStreetMap contributors, ODbL.
- Place names, positions and opening hours: the PolyU campus map.
- Building outlines derived from the same OpenStreetMap data, edited for this project.