# Glendalf Fireside Conversation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-screen, immersive fireside conversation at `/begin/fireside` where the visitor types and "Glendalf" (Dr. Glen as wizard) replies in Glen's cloned voice with streaming subtitles, ending on the hook *"I think I know what your body is asking for… shall we go and find it?"* — shipped dark behind `FIRESIDE_ENABLED`.

**Architecture:** A new multi-turn SSE agent endpoint mirrors the existing `/chat` streaming pattern (anthropic `messages.stream` → `data: {json}\n\n` frames) using the ASH-ally persona as the brain. Reply text streams to subtitles instantly; voice is rendered in parallel via the existing `POST /chat/tts` (ElevenLabs clone) and played when ready, with the latency covered by an instant pre-cached "filler" clip + a "pondering" video loop (the presentation dance). A new `fireside_sessions` sqlite table (mirroring `dashboard/journal_store.py`) persists turns + a session-keyed ASH coverage map, the latter computed fire-and-forget by reusing the **pure** functions already in `dashboard/ash_map.py` (no email key needed). The hook close is model-decided but server-gated, signalled with a `⟦HOOK⟧` sentinel parsed the same way `/chat` parses `⟦CTA⟧`.

**Tech Stack:** Python 3.12 / Flask (`app.py`), `anthropic` SDK streaming (`_cl.messages.stream`), sqlite3 (`chat_log.db`), ElevenLabs TTS (`/chat/tts`), vanilla JS + HTML5 `<video>`/`Audio` (`static/begin-fireside.html`), pytest, headless-browser render-verify.

## Global Constraints

- **Flag:** All new surfaces gate on `FIRESIDE_ENABLED = os.environ.get("FIRESIDE_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")` — default OFF. `/begin/fireside` and `/begin/fireside/agent` return `("", 404)` when off.
- **Do NOT touch** the live `/begin/doorway` route, `static/begin-doorway.html`, the journey quest, or any `/chat` behavior. Reuse, don't modify.
- **Voice clone:** ElevenLabs voice id comes from env (`ELEVENLABS_VOICE_ID`, prod = `jFxSqMckq2I4mET3C5QC`), model hardcoded `eleven_turbo_v2_5`. Reach it ONLY through the existing `POST /chat/tts` endpoint — do not add a second TTS path.
- **Conversational model:** `claude-haiku-4-5-20251001` (the same constant string `/chat` uses), exposed as `fireside_agent.FIRESIDE_MODEL` so it is a one-line swap to Opus later. Decision rationale in §"Resolved open items".
- **DB path:** `LOG_DB = Path(os.environ.get("DATA_DIR", <repo root>)) / "chat_log.db"`. All store functions take a caller-supplied `sqlite3` connection and call `init_table(cx)` on first use (the `dashboard/journal_store.py` contract). The caller holds `_db_lock` around writes.
- **Spoken output rules:** Glendalf's replies are read aloud + shown as subtitles — they MUST be plain prose: no markdown, no bullet lists, no headers, no emoji, 2–4 sentences, at most one question. `max_tokens=512`.
- **Anonymous v1:** session identity = the `amg_session` cookie only (get-or-create like `/begin/doorway`). `user_email`/`user_name` columns exist but stay null. No GHL, no payment, no Remedy Match in this slice — the conversation ENDS on the hook.
- **Assets are manifest-driven & placeholder-swappable.** v1 ships generated filler audio (real clone) + placeholder video loops; real HeyGen/footage swaps in later by replacing files named in `static/fireside/fireside-manifest.json` — no code change.
- **Tests:** `pytest` from repo root (root `conftest.py` does the gevent monkey-patch). Module-level flag constants require `importlib.reload(app)` after `monkeypatch.setenv(...)` — use the `_reload_app` fixture pattern. Mock the anthropic client with the `_FakeCl`/`_FakeStream` pattern (below). Never hit the network in unit tests.

---

## Resolved open items (spec §9)

These decisions are baked into the tasks below; recorded here so the implementer understands the "why".

1. **Model = Haiku 4.5** (`claude-haiku-4-5-20251001`). The presentation dance already hides 2–4s of TTS latency; adding Opus's slower token generation would delay the *subtitle stream* (the thing the user watches), hurting the "alive" feel. Depth is carried by a strong persona prompt + the ASH coverage context, not raw model size. Exposed as `FIRESIDE_MODEL` for a trivial later swap. Per-turn coverage extract also stays Haiku (reuses `ash_map._haiku_extract`).
2. **Hook trigger = heuristic-gated, model-timed, server-enforced.** The model is *permitted* to close only when `hook_eligible(turn_count, coverage)` is true: `turn_count >= 8` (hard cap) OR (`turn_count >= 4` AND ≥3 ASH dimensions are no longer "untouched"). When permitted, the model writes the hook invitation in its own warm words and appends a bare `⟦HOOK⟧` marker; the server hides the marker from subtitles, and only honors `hook:true` if eligibility *also* passes server-side (defense in depth — the model can't end early).
3. **Per-turn analysis on typed input = YES, lightweight, fire-and-forget.** Reuse `ash_map._haiku_extract` (text-only Haiku, structured forced-tool-use, never raises) + `ash_map.merge_turn` (pure) against a *session-scoped* memory dict stored in `fireside_sessions.ash_coverage`. Runs in a daemon thread after the reply so it never blocks the stream. Feeds the next turn's system prompt via `ash_map.context_block`. No new analysis code is written.
4. **Subtitle timing = stream text immediately; let voice catch up.** Tokens render word-by-word as the SSE arrives (the spec's lean). The MP3 is fetched in parallel and plays when ready; it is acceptable for subtitles to slightly lead the voice.
5. **Pondering loop = one looping clip in v1, but the manifest field is an array** (`pondering_loops`) so a couple of short clips can be dropped in and chosen at random later with zero code change. v1 ships one placeholder loop.
6. **Interjection cadence = client-side, light, rate-capped.** Trigger: input focused + non-empty + idle ≥ `3500ms` since the last keystroke. Caps: never on turn 1, at most once per 3 traveler turns, hard cap 3 per session. Plays one random clip from the manifest's `interjections` (a pre-cached static mp3 — no server round-trip, so zero latency and zero `/chat/tts` rate-limit pressure), then returns to listening. Content-aware interjection is a follow-on.

---

## File structure

| File | New/Modify | Responsibility |
|---|---|---|
| `dashboard/fireside_store.py` | **Create** | Pure sqlite store for `fireside_sessions` (get-or-create, append turn, update coverage, mark ended). Mirrors `journal_store.py`. No Flask import. |
| `dashboard/fireside_agent.py` | **Create** | Pure brain logic: `GLENDALF_PERSONA`, `FIRESIDE_MODEL`, `HOOK_SENTINEL`, `hook_eligible`, `parse_hook`, `build_system`, `build_messages`. Reuses `ash_map` pure fns. No Flask, no network in the pure fns. |
| `app.py` | **Modify** | Add `FIRESIDE_ENABLED` flag; `GET /begin/fireside` (serve HTML, 404 off); `POST /begin/fireside/agent` (SSE); `_fireside_coverage_async` helper. |
| `static/begin-fireside.html` | **Create** | Full-screen fireside UI + the presentation-dance state machine + interjection timer. Includes `/static/tts-output.js`. |
| `static/fireside/fireside-manifest.json` | **Create** | Asset manifest (intro/loops/fillers/interjections). |
| `static/fireside/audio/*.mp3` | **Create** | Filler + interjection clips (generated via the clone; placeholder fallback). |
| `static/fireside/video/*` | **Create** | Placeholder intro/speaking/pondering loops + poster. Swapped for real footage later. |
| `scripts/gen_fireside_fillers.py` | **Create** | One-shot generator: clone-render each filler/interjection phrase to mp3 (ffmpeg-silent fallback offline). |
| `tests/test_fireside_store.py` | **Create** | Store unit tests. |
| `tests/test_fireside_agent.py` | **Create** | Brain pure-logic unit tests. |
| `tests/test_fireside_routes.py` | **Create** | Route tests (flag-off 404, SSE stream, hook stripping, persistence). |

---

## Shared test scaffolding (referenced by Tasks 1–4)

Several test files reuse these helpers. Each test file that needs them defines its own copy (tests read out of order — do not factor into a shared import).

**The fake anthropic streaming client** (mirrors `tests/test_sales_pages_phase2.py`):

```python
import types

class _FakeStream:
    def __init__(self, toks): self._toks = toks
    def __enter__(self): return self
    def __exit__(self, *a): return False
    @property
    def text_stream(self):
        for t in self._toks: yield t

class _FakeMessages:
    def __init__(self, toks, boom=False): self._toks = toks; self.boom = boom; self.calls = 0
    def stream(self, **kw):
        self.calls += 1
        if self.boom: raise RuntimeError("claude down")
        return _FakeStream(self._toks)

class _FakeCl:
    def __init__(self, toks, boom=False): self.messages = _FakeMessages(toks, boom)
```

**The app-reload fixture** (module-level flag picks up env only on reload):

```python
import importlib

def _reload_app(monkeypatch, tmp_path, enabled="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FIRESIDE_ENABLED", enabled)
    import app as appmod
    importlib.reload(appmod)
    return appmod

def _frames(resp):
    return resp.get_data(as_text=True)
```

---

### Task 1: `fireside_sessions` store

**Files:**
- Create: `dashboard/fireside_store.py`
- Test: `tests/test_fireside_store.py`

**Interfaces:**
- Consumes: nothing (leaf module). A caller-supplied `sqlite3.Connection`.
- Produces (later tasks rely on these exact signatures):
  - `init_table(cx) -> None`
  - `get_or_create(cx, amg_session: str) -> dict` — returns the decoded latest **non-ended** session for `amg_session`, creating one if none exists or the latest is ended. Decoded dict keys: `id, amg_session, user_email, user_name, started_at, last_turn_at, turn_count, ended_at, transcript (list), ash_coverage (dict), signals (dict|None)`.
  - `append_turn(cx, fireside_id: int, speaker: str, text: str) -> None` — appends `{"speaker","text","ts"}` to `transcript`, bumps `last_turn_at`; increments `turn_count` **only** when `speaker == "traveler"`.
  - `update_coverage(cx, fireside_id: int, coverage: dict) -> None` — replaces `ash_coverage` JSON.
  - `mark_ended(cx, fireside_id: int) -> None` — stamps `ended_at`.
  - `get(cx, fireside_id: int) -> dict | None` — decoded row by id (for tests/callers).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fireside_store.py
import sqlite3
import pytest
from dashboard import fireside_store as fs


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    return cx


def test_get_or_create_creates_then_reuses(tmp_path):
    cx = _cx(tmp_path)
    a = fs.get_or_create(cx, "sess-1")
    assert a["id"] >= 1
    assert a["amg_session"] == "sess-1"
    assert a["turn_count"] == 0
    assert a["transcript"] == []
    assert a["ash_coverage"] == {}
    assert a["ended_at"] is None
    b = fs.get_or_create(cx, "sess-1")
    assert b["id"] == a["id"]  # reused, not duplicated


def test_get_or_create_new_after_ended(tmp_path):
    cx = _cx(tmp_path)
    a = fs.get_or_create(cx, "sess-2")
    fs.mark_ended(cx, a["id"])
    b = fs.get_or_create(cx, "sess-2")
    assert b["id"] != a["id"]  # ended session is not resumed in v1


def test_append_turn_counts_only_traveler(tmp_path):
    cx = _cx(tmp_path)
    s = fs.get_or_create(cx, "sess-3")
    fs.append_turn(cx, s["id"], "traveler", "I'm so tired lately.")
    fs.append_turn(cx, s["id"], "glendalf", "Tell me where you feel it.")
    fs.append_turn(cx, s["id"], "traveler", "In my chest.")
    got = fs.get(cx, s["id"])
    assert got["turn_count"] == 2  # two traveler turns
    assert [t["speaker"] for t in got["transcript"]] == ["traveler", "glendalf", "traveler"]
    assert got["transcript"][0]["text"] == "I'm so tired lately."
    assert got["transcript"][0]["ts"]  # stamped
    assert got["last_turn_at"]


def test_update_coverage_roundtrips(tmp_path):
    cx = _cx(tmp_path)
    s = fs.get_or_create(cx, "sess-4")
    cov = {"summary": "tired, chest-centered", "dimensions": {"symptoms": {"state": "opened"}}}
    fs.update_coverage(cx, s["id"], cov)
    got = fs.get(cx, s["id"])
    assert got["ash_coverage"] == cov


def test_mark_ended_sets_timestamp(tmp_path):
    cx = _cx(tmp_path)
    s = fs.get_or_create(cx, "sess-5")
    assert fs.get(cx, s["id"])["ended_at"] is None
    fs.mark_ended(cx, s["id"])
    assert fs.get(cx, s["id"])["ended_at"]


def test_get_missing_returns_none(tmp_path):
    cx = _cx(tmp_path)
    fs.init_table(cx)
    assert fs.get(cx, 999) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python -m pytest tests/test_fireside_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.fireside_store'`.

- [ ] **Step 3: Write the store**

```python
# dashboard/fireside_store.py
"""Sqlite store for the Glendalf fireside conversation (anonymous v1).

One row per fireside session, keyed in practice by the amg_session cookie. Pure
module: every function takes a caller-supplied sqlite3 connection and calls
init_table() on first use (mirrors dashboard/journal_store.py). No Flask import;
the caller holds the DB lock around writes.
"""
import json
import sqlite3

_JSON_COLS = ("transcript", "ash_coverage", "signals")
_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ','now')"


def init_table(cx) -> None:
    cx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS fireside_sessions (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          amg_session   TEXT,
          user_email    TEXT,
          user_name     TEXT,
          started_at    TEXT DEFAULT ({_NOW}),
          last_turn_at  TEXT,
          turn_count    INTEGER NOT NULL DEFAULT 0,
          ended_at      TEXT,
          transcript    TEXT NOT NULL DEFAULT '[]',
          ash_coverage  TEXT NOT NULL DEFAULT '{{}}',
          signals       TEXT
        )
        """
    )
    cx.execute(
        "CREATE INDEX IF NOT EXISTS idx_fireside_amg "
        "ON fireside_sessions(amg_session, ended_at)"
    )
    cx.commit()


def _decode(row: sqlite3.Row) -> dict:
    d = dict(row)
    for c in _JSON_COLS:
        v = d.get(c)
        if isinstance(v, str):
            try:
                d[c] = json.loads(v)
            except (ValueError, TypeError):
                d[c] = None
    return d


def get(cx, fireside_id: int) -> dict | None:
    init_table(cx)
    cx.row_factory = sqlite3.Row
    row = cx.execute(
        "SELECT * FROM fireside_sessions WHERE id = ?", (int(fireside_id),)
    ).fetchone()
    return _decode(row) if row is not None else None


def get_or_create(cx, amg_session: str) -> dict:
    init_table(cx)
    cx.row_factory = sqlite3.Row
    row = cx.execute(
        "SELECT * FROM fireside_sessions "
        "WHERE amg_session = ? AND ended_at IS NULL "
        "ORDER BY id DESC LIMIT 1",
        (amg_session or "",),
    ).fetchone()
    if row is not None:
        return _decode(row)
    cur = cx.execute(
        f"INSERT INTO fireside_sessions (amg_session, last_turn_at) "
        f"VALUES (?, {_NOW})",
        (amg_session or "",),
    )
    cx.commit()
    return get(cx, cur.lastrowid)


def append_turn(cx, fireside_id: int, speaker: str, text: str) -> None:
    init_table(cx)
    cx.row_factory = sqlite3.Row
    row = cx.execute(
        "SELECT transcript FROM fireside_sessions WHERE id = ?", (int(fireside_id),)
    ).fetchone()
    if row is None:
        return
    try:
        transcript = json.loads(row["transcript"]) or []
    except (ValueError, TypeError):
        transcript = []
    ts = cx.execute(f"SELECT {_NOW}").fetchone()[0]
    transcript.append({"speaker": speaker, "text": text or "", "ts": ts})
    inc = 1 if speaker == "traveler" else 0
    cx.execute(
        "UPDATE fireside_sessions "
        "SET transcript = ?, last_turn_at = ?, turn_count = turn_count + ? "
        "WHERE id = ?",
        (json.dumps(transcript), ts, inc, int(fireside_id)),
    )
    cx.commit()


def update_coverage(cx, fireside_id: int, coverage: dict) -> None:
    init_table(cx)
    cx.execute(
        "UPDATE fireside_sessions SET ash_coverage = ? WHERE id = ?",
        (json.dumps(coverage or {}), int(fireside_id)),
    )
    cx.commit()


def mark_ended(cx, fireside_id: int) -> None:
    init_table(cx)
    cx.execute(
        f"UPDATE fireside_sessions SET ended_at = {_NOW} WHERE id = ?",
        (int(fireside_id),),
    )
    cx.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python -m pytest tests/test_fireside_store.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e5fec1df
git add dashboard/fireside_store.py tests/test_fireside_store.py
git commit -m "feat(fireside): fireside_sessions sqlite store"
```

---

### Task 2: Glendalf brain (pure logic)

**Files:**
- Create: `dashboard/fireside_agent.py`
- Test: `tests/test_fireside_agent.py`

**Interfaces:**
- Consumes: `dashboard/ash_map.py` pure functions — `ash_map.context_block(memory: dict) -> str`, `ash_map.DIM_KEYS: list[str]` (imported; do not re-implement).
- Produces (Task 4 relies on these exact names):
  - `FIRESIDE_MODEL: str` = `"claude-haiku-4-5-20251001"`
  - `HOOK_SENTINEL: str` = `"⟦HOOK⟧"`
  - `MIN_HOOK_TURNS = 4`, `HARD_CAP_TURNS = 8`, `MIN_DIMS_TOUCHED = 3`, `MAX_HISTORY_TURNS = 12`
  - `GLENDALF_PERSONA: str`
  - `hook_eligible(turn_count: int, coverage: dict) -> bool`
  - `parse_hook(full_text: str) -> tuple[str, bool]` — returns `(clean_text_without_marker, hook_present)`
  - `build_system(coverage: dict, turn_count: int) -> str`
  - `build_messages(transcript: list[dict], user_message: str) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fireside_agent.py
from dashboard import fireside_agent as fa


def test_parse_hook_present_strips_marker():
    raw = "I think I know what your body is asking for. Shall we go and find it?\n⟦HOOK⟧"
    clean, hooked = fa.parse_hook(raw)
    assert hooked is True
    assert "⟦HOOK⟧" not in clean
    assert clean.endswith("find it?")


def test_parse_hook_absent():
    clean, hooked = fa.parse_hook("Tell me more about that.")
    assert hooked is False
    assert clean == "Tell me more about that."


def test_hook_eligible_hard_cap():
    assert fa.hook_eligible(8, {}) is True
    assert fa.hook_eligible(9, {"dimensions": {}}) is True


def test_hook_eligible_min_turns_and_dims():
    cov = {"dimensions": {
        "symptoms": {"state": "opened"},
        "terrain": {"state": "explored"},
        "spirit": {"state": "opened"},
        "mind": {"state": "untouched"},
    }}
    assert fa.hook_eligible(4, cov) is True   # 4 turns, 3 touched
    assert fa.hook_eligible(3, cov) is False  # too few turns
    thin = {"dimensions": {"symptoms": {"state": "opened"}}}
    assert fa.hook_eligible(4, thin) is False  # only 1 touched


def test_hook_eligible_early_is_false():
    assert fa.hook_eligible(1, {}) is False
    assert fa.hook_eligible(0, {"dimensions": {}}) is False


def test_build_system_includes_persona_and_context():
    cov = {"summary": "weary traveler", "dimensions": {"symptoms": {"state": "opened", "opened_excerpt": "always tired"}}}
    sys_low = fa.build_system(cov, turn_count=2)
    assert "Glendalf" in sys_low
    assert "weary traveler" in sys_low          # context_block folded in
    assert "do not close" in sys_low.lower()    # hook forbidden early
    sys_hi = fa.build_system(cov, turn_count=8)
    assert "⟦HOOK⟧" in sys_hi                    # hook permitted at cap
    assert "shall we go and find it" in sys_hi.lower()


def test_build_messages_maps_roles_and_appends():
    transcript = [
        {"speaker": "traveler", "text": "hi"},
        {"speaker": "glendalf", "text": "welcome, friend"},
        {"speaker": "traveler", "text": ""},          # empty -> dropped
    ]
    msgs = fa.build_messages(transcript, "I feel stuck")
    assert msgs[0] == {"role": "user", "content": "hi"}
    assert msgs[1] == {"role": "assistant", "content": "welcome, friend"}
    assert msgs[-1] == {"role": "user", "content": "I feel stuck"}
    assert all(m["content"] for m in msgs)         # no empty turns


def test_build_messages_caps_history():
    transcript = [{"speaker": "traveler", "text": f"m{i}"} for i in range(40)]
    msgs = fa.build_messages(transcript, "now")
    # last MAX_HISTORY_TURNS of history + the new user message
    assert len(msgs) == fa.MAX_HISTORY_TURNS + 1
    assert msgs[-1]["content"] == "now"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python -m pytest tests/test_fireside_agent.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.fireside_agent'`.

- [ ] **Step 3: Write the brain**

```python
# dashboard/fireside_agent.py
"""Glendalf — the fireside conversational brain (pure logic).

The conversational purpose is the ASH health ally (see
docs/superpowers/specs/2026-06-27-ash-health-ally-foundation.md), re-presented
as an intimate fireside talk. This module holds only PURE, network-free logic:
persona assembly, the hook-eligibility heuristic, hook-sentinel parsing, and
turn-history mapping. The actual streaming LLM call lives in app.py's route
(mirroring /chat). Session-scoped ASH coverage reuses the pure functions in
dashboard/ash_map.py — no email key, no rebuild.
"""
from dashboard import ash_map

FIRESIDE_MODEL = "claude-haiku-4-5-20251001"   # swap to an Opus id for more depth
HOOK_SENTINEL = "⟦HOOK⟧"

MIN_HOOK_TURNS = 4
HARD_CAP_TURNS = 8
MIN_DIMS_TOUCHED = 3
MAX_HISTORY_TURNS = 12

GLENDALF_PERSONA = (
    "You are Glendalf — a warm, wise, unhurried wizard-healer by his fire. You "
    "are Dr. Glen Swartwout's voice and clinical lens in story form: the listener "
    "who translates the body's quiet messages into meaning (Wellness Whispering).\n"
    "\n"
    "YOUR WAY OF BEING:\n"
    "- Meet the traveler where they are. Some unload; some test the water. Both are welcome; never force.\n"
    "- Listen for their own words and mirror them (their language, their model of the world).\n"
    "- Reflect back what you heard and felt before you ask anything: 'I hear you say…', 'It sounds like…'.\n"
    "- A symptom is not the enemy; it is a message the body is sending. Honor their intuition that there is an answer, and that it is not in suppression or substitution but in partnering with the body's own intelligence.\n"
    "- Follow the thread THEY are pulling. Only step toward a new area when something they said opens the door to it — never a non-sequitur.\n"
    "- Ask at most ONE gentle question, and only when it deepens what they already opened.\n"
    "\n"
    "HOW YOU SPEAK (this is read aloud in your own voice and shown as subtitles):\n"
    "- Plain, warm prose. NO markdown, NO bullet points, NO headings, NO emoji.\n"
    "- 2 to 4 sentences. Unhurried but not rambling.\n"
    "- Speak as 'I' to 'you'. Never mention being an AI, a model, or a system.\n"
)

_HOOK_FORBIDDEN = (
    "PACING: You are still getting to know this traveler. Keep listening and "
    "reflecting. Do NOT close the conversation yet, and do NOT use the word "
    "marker under any circumstances."
)

_HOOK_PERMISSION = (
    "CLOSING: You have now heard enough to land a meaningful reflection. When — "
    "and only when — it feels true and earned, close warmly: name in one breath "
    "what you sense their body is asking for, then invite them onward in your own "
    "words, anchored on this meaning: 'I think I know what your body is asking "
    "for… shall we go and find it?'. After that closing line, on a new line, "
    "output the exact marker " + HOOK_SENTINEL + " and nothing after it. If it does "
    "not yet feel earned this turn, keep listening instead and omit the marker."
)


def hook_eligible(turn_count: int, coverage: dict) -> bool:
    tc = int(turn_count or 0)
    if tc >= HARD_CAP_TURNS:
        return True
    if tc < MIN_HOOK_TURNS:
        return False
    dims = (coverage or {}).get("dimensions") or {}
    touched = sum(
        1 for k in ash_map.DIM_KEYS
        if (dims.get(k) or {}).get("state", "untouched") != "untouched"
    )
    return touched >= MIN_DIMS_TOUCHED


def parse_hook(full_text: str) -> tuple[str, bool]:
    text = full_text or ""
    if HOOK_SENTINEL in text:
        return (text.replace(HOOK_SENTINEL, "").rstrip(), True)
    return (text, False)


def build_system(coverage: dict, turn_count: int) -> str:
    ctx = ash_map.context_block(coverage or {})
    gate = _HOOK_PERMISSION if hook_eligible(turn_count, coverage) else _HOOK_FORBIDDEN
    return (
        GLENDALF_PERSONA
        + "\n--- WHAT YOU ALREADY KNOW ABOUT THIS TRAVELER ---\n"
        + ctx
        + "\n\n"
        + gate
    )


def build_messages(transcript: list, user_message: str) -> list:
    msgs = []
    for t in (transcript or [])[-MAX_HISTORY_TURNS:]:
        content = (t.get("text") or "").strip()
        if not content:
            continue
        role = "assistant" if t.get("speaker") == "glendalf" else "user"
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": (user_message or "").strip()})
    return msgs
```

> Note: `build_messages` caps to the last `MAX_HISTORY_TURNS` **before** dropping empties, so `test_build_messages_caps_history` (40 non-empty turns) yields exactly `MAX_HISTORY_TURNS + 1`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python -m pytest tests/test_fireside_agent.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e5fec1df
git add dashboard/fireside_agent.py tests/test_fireside_agent.py
git commit -m "feat(fireside): Glendalf brain pure logic (persona, hook gate, history)"
```

---

### Task 3: Flag + `/begin/fireside` serve route

**Files:**
- Modify: `app.py` (add flag near the other flag constants ~line 4183; add route near `begin_doorway` ~line 1962)
- Test: `tests/test_fireside_routes.py` (the flag + serve cases; Task 4 adds the agent cases to the same file)

**Interfaces:**
- Consumes: `FIRESIDE_ENABLED` (new module-level bool), `STATIC` (existing `Path`), `send_from_directory`, `request`, `uuid` (all already imported).
- Produces: `GET /begin/fireside` → serves `static/begin-fireside.html` with no-cache headers + amg_session cookie; 404 when flag off.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fireside_routes.py
import importlib


def _reload_app(monkeypatch, tmp_path, enabled="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FIRESIDE_ENABLED", enabled)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_fireside_page_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="false")
    r = appmod.app.test_client().get("/begin/fireside")
    assert r.status_code == 404


def test_fireside_page_served_when_on(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    r = appmod.app.test_client().get("/begin/fireside")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "fireside" in body.lower()
    # sets the anonymous session cookie
    assert any("amg_session" in (h or "") for h in r.headers.getlist("Set-Cookie"))
```

> The serve test needs `static/begin-fireside.html` to exist. Add a minimal stub now so this task is independently testable; Task 6 replaces it with the full UI:

```bash
cd /tmp/wt-deploy-chat-e5fec1df
printf '<!doctype html><html><head><title>Fireside</title></head><body><main id="fireside">stub</main></body></html>' > static/begin-fireside.html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python -m pytest tests/test_fireside_routes.py -q`
Expected: FAIL — `assert 404 == 200` (route not defined yet; Flask returns 404 even with flag on).

- [ ] **Step 3a: Add the flag constant**

In `app.py`, immediately after the `PAY_IT_FORWARD_ENABLED = ...` line (~line 4183), add:

```python
FIRESIDE_ENABLED = os.environ.get("FIRESIDE_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 3b: Add the serve route**

In `app.py`, immediately after the `begin_doorway()` function (ends ~line 1970), add:

```python
@app.route("/begin/fireside")
def begin_fireside():
    if not FIRESIDE_ENABLED:
        return ("", 404)
    resp = send_from_directory(STATIC, "begin-fireside.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", uuid.uuid4().hex, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python -m pytest tests/test_fireside_routes.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e5fec1df
git add app.py static/begin-fireside.html tests/test_fireside_routes.py
git commit -m "feat(fireside): FIRESIDE_ENABLED flag + /begin/fireside serve route (dark)"
```

---

### Task 4: `POST /begin/fireside/agent` SSE endpoint + coverage

**Files:**
- Modify: `app.py` (add the agent route after `begin_fireside()`; add `_fireside_coverage_async` helper near it)
- Test: `tests/test_fireside_routes.py` (extend with agent cases)

**Interfaces:**
- Consumes: `fireside_store` (Task 1), `fireside_agent` (Task 2), `ash_map` (existing), `_cl` (existing anthropic client), `_db_lock`, `LOG_DB`, `sqlite3`, `sse()` (existing helper, line ~1687), `stream_with_context`, `Response`, `chat_cta.stream_visible` (existing), `threading` (imported).
- Produces: `POST /begin/fireside/agent`. Request JSON `{"message": str, "session_id"?: str}`. SSE frames: `data: {"token": "..."}\n\n` per visible delta, then a terminal `data: {"done": true, "hook": bool, "fireside_id": int, "turn_count": int, "session_id": str}\n\n`. On model failure: `data: {"error": true, "detail": "..."}\n\n`. 404 when flag off; 400 on empty message.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_fireside_routes.py`)

```python
import types

class _FakeStream:
    def __init__(self, toks): self._toks = toks
    def __enter__(self): return self
    def __exit__(self, *a): return False
    @property
    def text_stream(self):
        for t in self._toks: yield t

class _FakeMessages:
    def __init__(self, toks, boom=False): self._toks = toks; self.boom = boom; self.calls = 0
    def stream(self, **kw):
        self.calls += 1
        if self.boom: raise RuntimeError("claude down")
        return _FakeStream(self._toks)

class _FakeCl:
    def __init__(self, toks, boom=False): self.messages = _FakeMessages(toks, boom)


def _post(appmod, message):
    # disable the fire-and-forget coverage thread so tests stay deterministic + offline
    return appmod.app.test_client().post("/begin/fireside/agent", json={"message": message})


def test_agent_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="false")
    assert _post(appmod, "hi").status_code == 404


def test_agent_empty_message_400(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    assert _post(appmod, "   ").status_code == 400


def test_agent_streams_tokens_and_persists(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_cl", _FakeCl(["I hear ", "you, ", "friend."]))
    body = _post(appmod, "I'm exhausted").get_data(as_text=True)
    assert "I hear " in body
    assert '"done": true' in body
    assert '"hook": false' in body
    # persisted: one traveler turn + one glendalf turn
    import sqlite3
    from dashboard import fireside_store as fs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        s = fs.get_or_create(cx, "")  # no cookie -> amg_session ""
    # transcript should hold both turns under the session we just created
    assert any(t["speaker"] == "glendalf" and "I hear you, friend." == t["text"]
               for t in s["transcript"])


def test_agent_hides_hook_marker_and_flags_when_eligible(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    # Force eligibility regardless of turn count/coverage
    from dashboard import fireside_agent as fa
    monkeypatch.setattr(fa, "hook_eligible", lambda *a, **k: True)
    monkeypatch.setattr(appmod, "_cl",
                        _FakeCl(["Shall we go and find it?", "\n", "⟦HOOK⟧"]))
    body = _post(appmod, "I think I'm ready").get_data(as_text=True)
    assert "⟦HOOK⟧" not in body          # marker never reaches the client
    assert "Shall we go and find it?" in body
    assert '"hook": true' in body


def test_agent_hook_marker_ignored_when_not_eligible(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    from dashboard import fireside_agent as fa
    monkeypatch.setattr(fa, "hook_eligible", lambda *a, **k: False)
    monkeypatch.setattr(appmod, "_cl", _FakeCl(["Too soon.", "⟦HOOK⟧"]))
    body = _post(appmod, "first thing I say").get_data(as_text=True)
    assert "⟦HOOK⟧" not in body
    assert '"hook": false' in body        # server refuses to honor an early close


def test_agent_error_frame_on_model_failure(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    monkeypatch.setattr(appmod, "_fireside_coverage_async", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_cl", _FakeCl([], boom=True))
    body = _post(appmod, "hello").get_data(as_text=True)
    assert '"error": true' in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python -m pytest tests/test_fireside_routes.py -q`
Expected: FAIL — agent tests get 404/no route (`_fireside_coverage_async` attr missing / route undefined).

- [ ] **Step 3a: Add the coverage helper**

In `app.py`, just before the `begin_fireside()` route, add:

```python
def _fireside_coverage_async(fireside_id, user_text, ally_text, coverage):
    """Fire-and-forget: update the session ASH coverage map after a reply.
    Reuses ash_map's PURE functions (session-scoped, no email key). Never blocks
    the stream; any failure degrades to 'learned nothing this turn'."""
    def _work():
        try:
            from dashboard import fireside_store, ash_map
            extracted = ash_map._haiku_extract(coverage or {}, user_text, ally_text)
            merged = ash_map.merge_turn(coverage or {}, extracted)
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                fireside_store.update_coverage(cx, fireside_id, merged)
        except Exception as e:
            print(f"[fireside] coverage update failed: {e!r}", flush=True)
    threading.Thread(target=_work, daemon=True).start()
```

- [ ] **Step 3b: Add the agent route**

In `app.py`, immediately after `begin_fireside()`, add:

```python
@app.route("/begin/fireside/agent", methods=["POST", "OPTIONS"])
def begin_fireside_agent():
    if request.method == "OPTIONS":
        return ("", 200)
    if not FIRESIDE_ENABLED:
        return ("", 404)

    from dashboard import fireside_store, fireside_agent
    from dashboard.chat_cta import stream_visible

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty"}), 400
    session_id = (request.cookies.get("amg_session")
                  or (data.get("session_id") or "").strip()
                  or uuid.uuid4().hex)

    # Read state + record the traveler turn under the lock.
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        sess = fireside_store.get_or_create(cx, session_id)
        fireside_id = sess["id"]
        coverage = sess.get("ash_coverage") or {}
        transcript = sess.get("transcript") or []
        prior_turns = int(sess.get("turn_count") or 0)
        fireside_store.append_turn(cx, fireside_id, "traveler", message)

    this_turn = prior_turns + 1
    system = fireside_agent.build_system(coverage, this_turn)
    messages = fireside_agent.build_messages(transcript, message)
    full = []

    def generate():
        def _toks():
            with _cl.messages.stream(
                model=fireside_agent.FIRESIDE_MODEL,
                max_tokens=512,
                system=system,
                messages=messages,
            ) as stream:
                for tok in stream.text_stream:
                    full.append(tok)
                    yield tok
        try:
            for delta in stream_visible(_toks(), sentinel=fireside_agent.HOOK_SENTINEL):
                yield sse({"token": delta})
        except Exception as e:
            yield sse({"error": True, "detail": str(e)})
            return

        clean, hooked = fireside_agent.parse_hook("".join(full))
        hooked = bool(hooked and fireside_agent.hook_eligible(this_turn, coverage))
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            fireside_store.append_turn(cx, fireside_id, "glendalf", clean)
            if hooked:
                fireside_store.mark_ended(cx, fireside_id)
        _fireside_coverage_async(fireside_id, message, clean, coverage)
        yield sse({"done": True, "hook": hooked, "fireside_id": fireside_id,
                   "turn_count": this_turn, "session_id": session_id})

    resp = Response(stream_with_context(generate()), content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    if not request.cookies.get("amg_session"):
        resp.set_cookie("amg_session", session_id, max_age=60 * 60 * 24 * 365,
                        httponly=True, samesite="Lax", secure=request.is_secure)
    return resp
```

> Why `stream_visible` works here: it fully drains the source generator even after the sentinel (the fix documented in `chat_cta.py` lines 17–23), so `full` accumulates `⟦HOOK⟧` for `parse_hook` to detect, while the marker is never *yielded* to the client. The hook sentence preceding the marker streams normally.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python -m pytest tests/test_fireside_routes.py -q`
Expected: PASS (8 passed — 2 from Task 3 + 6 here).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e5fec1df
git add app.py tests/test_fireside_routes.py
git commit -m "feat(fireside): SSE Glendalf agent endpoint + session-scoped ASH coverage"
```

---

### Task 5: Assets — manifest, filler/interjection audio, placeholder video

**Files:**
- Create: `static/fireside/fireside-manifest.json`
- Create: `scripts/gen_fireside_fillers.py`
- Create: `static/fireside/audio/*.mp3` (generated)
- Create: `static/fireside/video/intro-poster.jpg`, `intro.mp4`, `speaking-loop.mp4`, `pondering-1.mp4` (placeholders)
- Test: `tests/test_fireside_routes.py` (add a manifest-served check)

**Interfaces:**
- Consumes: ElevenLabs via the same payload `_el_tts` uses (script may reuse `app._el_tts` if importable, else inline urllib); `ffmpeg` for placeholders.
- Produces: `static/fireside/fireside-manifest.json` served at `/static/fireside/fireside-manifest.json` via the existing `/static/<path>` catch-all (line 3109). Schema below — Task 6 consumes it.

- [ ] **Step 1: Write the manifest**

```json
{
  "intro_video": "/static/fireside/video/intro.mp4",
  "intro_poster": "/static/fireside/video/intro-poster.jpg",
  "speaking_loop": "/static/fireside/video/speaking-loop.mp4",
  "pondering_loops": ["/static/fireside/video/pondering-1.mp4"],
  "fillers": [
    {"id": "hmm",     "kind": "think",  "text": "Hmmm…",                                  "file": "/static/fireside/audio/filler-hmm.mp3"},
    {"id": "well",    "kind": "think",  "text": "Well…",                                  "file": "/static/fireside/audio/filler-well.mp3"},
    {"id": "isee",    "kind": "ack",    "text": "I see.",                                 "file": "/static/fireside/audio/filler-isee.mp3"},
    {"id": "interesting", "kind": "ack", "text": "That's very interesting.",             "file": "/static/fireside/audio/filler-interesting.mp3"},
    {"id": "consider", "kind": "think", "text": "Let me consider that.",                 "file": "/static/fireside/audio/filler-consider.mp3"},
    {"id": "contemplate", "kind": "think", "text": "You've given me much to contemplate there.", "file": "/static/fireside/audio/filler-contemplate.mp3"},
    {"id": "heres",   "kind": "bridge", "text": "Here's what I think…",                  "file": "/static/fireside/audio/filler-heres.mp3"}
  ],
  "interjections": [
    {"id": "ah",      "text": "Ah—",                       "file": "/static/fireside/audio/inter-ah.mp3"},
    {"id": "goon",    "text": "Yes, go on…",               "file": "/static/fireside/audio/inter-goon.mp3"},
    {"id": "mm",      "text": "Mm—",                        "file": "/static/fireside/audio/inter-mm.mp3"},
    {"id": "oh",      "text": "Oh…",                        "file": "/static/fireside/audio/inter-oh.mp3"},
    {"id": "heart",   "text": "Now that's the heart of it—","file": "/static/fireside/audio/inter-heart.mp3"}
  ]
}
```

- [ ] **Step 2: Write the generator script**

```python
# scripts/gen_fireside_fillers.py
"""Render the fireside filler + interjection phrases to mp3 in Glen's clone.

Reads static/fireside/fireside-manifest.json, renders each `text` via ElevenLabs
(same voice/model as /chat/tts), and writes the `file` path. Run once:

    doppler run -p remedy-match -c prd -- python3 scripts/gen_fireside_fillers.py

Offline / no key: pass --placeholder to emit short silent mp3s via ffmpeg so the
UI and render-verify still work; swap in real clips later by re-running with a key.
"""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "static" / "fireside" / "fireside-manifest.json"
EL_BASE = "https://api.elevenlabs.io/v1"


def _abs(rel: str) -> Path:
    return ROOT / rel.lstrip("/")


def _silent(out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", "0.7", "-q:a", "9", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _render(text: str, out: Path):
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "")
    payload = json.dumps({
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.80, "style": 0.20},
    }).encode()
    req = urllib.request.Request(
        f"{EL_BASE}/text-to-speech/{voice}", data=payload,
        headers={"xi-api-key": api_key, "Content-Type": "application/json",
                 "Accept": "audio/mpeg"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        audio = resp.read()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio)


def main():
    placeholder = "--placeholder" in sys.argv or not os.environ.get("ELEVENLABS_API_KEY")
    manifest = json.loads(MANIFEST.read_text())
    clips = list(manifest["fillers"]) + list(manifest["interjections"])
    for c in clips:
        out = _abs(c["file"])
        if placeholder:
            _silent(out)
            print(f"[placeholder] {out.name}")
        else:
            _render(c["text"], out)
            print(f"[rendered]    {out.name}  <- {c['text']!r}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate the audio (real if creds present, else placeholder)**

```bash
cd /tmp/wt-deploy-chat-e5fec1df
mkdir -p static/fireside/audio static/fireside/video
# Real clone render (preferred):
doppler run -p remedy-match -c prd -- python3 scripts/gen_fireside_fillers.py \
  || python3 scripts/gen_fireside_fillers.py --placeholder
ls static/fireside/audio/*.mp3 | wc -l   # expect 12
```
Expected: 12 mp3 files (7 fillers + 5 interjections).

- [ ] **Step 4: Generate placeholder video + poster** (swapped for real footage later)

```bash
cd /tmp/wt-deploy-chat-e5fec1df
# Warm dark "firelit" poster
ffmpeg -y -f lavfi -i color=c=0x1a0f08:s=1280x720 -frames:v 1 \
  static/fireside/video/intro-poster.jpg
# 3 short looping placeholders (solid warm tone; distinct brightness so transitions are visible)
for name in intro speaking-loop pondering-1; do
  case $name in intro) col=0x241405;; speaking-loop) col=0x2e1808;; *) col=0x1a0f08;; esac
  ffmpeg -y -f lavfi -i color=c=$col:s=1280x720:d=6 -r 24 -pix_fmt yuv420p \
    static/fireside/video/$name.mp4
done
ls static/fireside/video
```
Expected: `intro-poster.jpg intro.mp4 pondering-1.mp4 speaking-loop.mp4`.

- [ ] **Step 5: Add the manifest-served test** (append to `tests/test_fireside_routes.py`)

```python
def test_manifest_served(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path, enabled="true")
    r = appmod.app.test_client().get("/static/fireside/fireside-manifest.json")
    assert r.status_code == 200
    data = r.get_json()
    assert data["intro_video"].endswith("intro.mp4")
    assert len(data["fillers"]) >= 5
    assert len(data["interjections"]) >= 3
```

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python -m pytest tests/test_fireside_routes.py::test_manifest_served -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /tmp/wt-deploy-chat-e5fec1df
git add scripts/gen_fireside_fillers.py static/fireside/fireside-manifest.json \
        static/fireside/audio static/fireside/video tests/test_fireside_routes.py
git commit -m "feat(fireside): asset manifest + filler/interjection audio + placeholder video"
```

---

### Task 6: The fireside UI + presentation dance + interjections

**Files:**
- Create (replace the Task 3 stub): `static/begin-fireside.html`

**Interfaces:**
- Consumes: `GET /static/fireside/fireside-manifest.json`; `POST /begin/fireside/agent` (SSE token/done/error frames); `POST /chat/tts` (reply MP3); `/static/tts-output.js` (`TTS.attach` for the replay affordance + speechSynthesis fallback).
- Produces: the full immersive experience. No exports.

**The presentation-dance state machine (spec §5):**
`INTRO` → `LISTENING` → (send) → `PONDERING` (instant filler clip + pondering video + SSE subtitles streaming) → `SPEAKING` (reply MP3 plays, speaking video) → `LISTENING` … → `ENDED` (hook).

- [ ] **Step 1: Write the full UI file**

Write `static/begin-fireside.html` with exactly this content:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>By the Fire with Glendalf</title>
<style>
  :root { --warm:#f3e2c7; --dim:#b89a72; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:#0c0703; color:var(--warm);
    font-family: Georgia, 'Times New Roman', serif; overflow:hidden; }
  #fireside { position:fixed; inset:0; display:flex; flex-direction:column;
    align-items:center; justify-content:flex-end; }
  #stage { position:absolute; inset:0; width:100%; height:100%; object-fit:cover;
    background:#0c0703; }
  #vignette { position:absolute; inset:0; pointer-events:none;
    background:radial-gradient(ellipse at 50% 40%, transparent 30%, rgba(0,0,0,.78) 100%); }
  #subtitle { position:relative; z-index:2; max-width:760px; width:92%;
    text-align:center; font-size:clamp(20px,3.4vw,30px); line-height:1.45;
    text-shadow:0 2px 18px rgba(0,0,0,.95); margin-bottom:18px; min-height:1.5em; }
  #replay { display:inline-flex; vertical-align:middle; margin-left:.4em; }
  #replay .tts-btn { background:none; border:none; color:var(--dim); cursor:pointer;
    font-size:.7em; }
  #composer { position:relative; z-index:2; width:92%; max-width:760px;
    display:flex; gap:10px; margin-bottom:max(22px, env(safe-area-inset-bottom));
    transition:opacity .4s; }
  #say { flex:1; background:rgba(20,12,6,.72); border:1px solid #6b5436;
    color:var(--warm); border-radius:26px; padding:14px 20px;
    font-size:18px; font-family:inherit; outline:none; }
  #say::placeholder { color:#7d6647; }
  #send { background:#6b5436; border:none; color:var(--warm); border-radius:50%;
    width:52px; height:52px; font-size:22px; cursor:pointer; }
  #send:disabled { opacity:.4; cursor:default; }
  #begin-overlay { position:absolute; inset:0; z-index:5; display:flex;
    align-items:center; justify-content:center; background:rgba(6,3,1,.55); }
  #begin-overlay button { background:#6b5436; color:var(--warm); border:none;
    border-radius:30px; padding:16px 34px; font-size:20px; font-family:inherit;
    cursor:pointer; }
  #yes { display:none; position:relative; z-index:2; margin-bottom:26px;
    background:linear-gradient(#caa86a,#8a6a38); color:#1a0f06; border:none;
    border-radius:30px; padding:16px 40px; font-size:22px; font-family:inherit;
    cursor:pointer; box-shadow:0 0 26px 6px rgba(220,180,90,.7);
    animation:glow 2.2s ease-in-out infinite; }
  @keyframes glow { 0%,100%{box-shadow:0 0 22px 4px rgba(220,180,90,.55);}
    50%{box-shadow:0 0 40px 12px rgba(240,200,110,.95);} }
  .hidden { display:none !important; }
</style>
</head>
<body>
<main id="fireside">
  <video id="stage" playsinline muted loop preload="auto"></video>
  <div id="vignette"></div>
  <div id="subtitle" aria-live="polite"></div>
  <button id="yes" type="button">Yes</button>
  <form id="composer" autocomplete="off">
    <input id="say" name="say" type="text" placeholder="Tell me what's on your heart…"
           aria-label="Speak to Glendalf" />
    <button id="send" type="submit" aria-label="Send">➤</button>
  </form>
  <div id="begin-overlay"><button id="begin-btn" type="button">Sit by the fire</button></div>
</main>

<script src="/static/tts-output.js"></script>
<script>
(function () {
  "use strict";
  var stage = document.getElementById('stage');
  var sub = document.getElementById('subtitle');
  var say = document.getElementById('say');
  var send = document.getElementById('send');
  var composer = document.getElementById('composer');
  var yes = document.getElementById('yes');
  var overlay = document.getElementById('begin-overlay');
  var beginBtn = document.getElementById('begin-btn');

  var manifest = null;
  var state = 'intro';
  var lastFiller = null, lastInter = null;
  var travelerTurns = 0, interjections = 0, turnsSinceInterjection = 99;
  var idleTimer = null;
  var fillerAudio = new Audio();
  var replyAudio = new Audio();

  function pick(arr, lastId) {
    if (!arr || !arr.length) return null;
    if (arr.length === 1) return arr[0];
    var c; do { c = arr[Math.floor(Math.random() * arr.length)]; } while (c.id === lastId);
    return c;
  }

  function setVideo(src, opts) {
    opts = opts || {};
    if (stage.getAttribute('data-src') === src) return;
    stage.setAttribute('data-src', src);
    stage.src = src;
    stage.loop = opts.loop !== false;
    stage.muted = true;
    stage.play().catch(function () {});
  }

  function pondering() { var p = manifest.pondering_loops; setVideo(p[Math.floor(Math.random() * p.length)], {loop:true}); }
  function listening()  { pondering(); state = 'listening'; }   // pondering loop doubles as idle
  function speakingLoop(){ setVideo(manifest.speaking_loop, {loop:true}); }

  function playClip(audio, file) {
    try { audio.pause(); audio.currentTime = 0; } catch (e) {}
    audio.src = file; audio.muted = false;
    return audio.play().catch(function () {});
  }

  // --- Interjection: rare, on a typing pause, rate-capped, never on turn 1 ---
  function maybeInterject() {
    if (state !== 'listening') return;
    if (!say.value.trim()) return;
    if (travelerTurns < 1) return;                 // never before the first real turn
    if (interjections >= 3) return;                // hard cap per session
    if (turnsSinceInterjection < 3) return;        // at most once per 3 turns
    var clip = pick(manifest.interjections, lastInter);
    if (!clip) return;
    lastInter = clip.id; interjections++; turnsSinceInterjection = 0;
    playClip(fillerAudio, clip.file);
  }
  function armIdle() {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(maybeInterject, 3500);
  }

  // --- A turn ---
  function sendTurn(message) {
    state = 'pondering';
    send.disabled = true;
    if (idleTimer) clearTimeout(idleTimer);
    travelerTurns++; turnsSinceInterjection++;
    sub.textContent = '';
    // 1. instant filler (zero-latency, so never silent)
    var f = pick(manifest.fillers, lastFiller);
    if (f) { lastFiller = f.id; playClip(fillerAudio, f.file); }
    // 2. pondering video while we think
    pondering();
    // 3 + 4. stream subtitles via SSE while TTS renders in parallel
    streamReply(message);
  }

  function streamReply(message) {
    var full = '';
    fetch('/begin/fireside/agent', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: message})
    }).then(function (r) {
      var reader = r.body.getReader(), dec = new TextDecoder(), buf = '';
      (function pump() {
        return reader.read().then(function (res) {
          if (res.done) return finishReply(full, false);
          buf += dec.decode(res.value, {stream: true});
          var parts = buf.split('\n\n'); buf = parts.pop();
          parts.forEach(function (p) {
            var line = p.replace(/^data: /, '').trim();
            if (!line) return;
            var msg; try { msg = JSON.parse(line); } catch (e) { return; }
            if (msg.token) { full += msg.token; sub.textContent = full; }
            else if (msg.error) { finishReply(full || 'My thoughts wandered. Say that again?', false); }
            else if (msg.done) { finishReply(full, !!msg.hook); }
          });
          return pump();
        });
      })();
    }).catch(function () { finishReply('My thoughts wandered. Tell me again?', false); });
  }

  function finishReply(text, hook) {
    sub.textContent = text;
    // render the reply voice (parallel path); play when ready, then return to listening
    fetch('/chat/tts', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text})
    }).then(function (r) {
      if (!r.ok) throw new Error('tts ' + r.status);
      return r.blob();
    }).then(function (blob) {
      var url = URL.createObjectURL(blob);
      replyAudio.src = url; replyAudio.muted = false;
      speakingLoop(); state = 'speaking';
      replyAudio.onended = function () { URL.revokeObjectURL(url); afterReply(text, hook); };
      replyAudio.onerror = function () { URL.revokeObjectURL(url); afterReply(text, hook); };
      replyAudio.play().catch(function () { afterReply(text, hook); });
    }).catch(function () {
      // voice failed -> speechSynthesis fallback via TTS helper, then continue
      if (window.TTS && window.TTS.attachAndSpeak) { try { window.TTS.attachAndSpeak(sub, text); } catch (e) {} }
      setTimeout(function () { afterReply(text, hook); }, 600);
    });
  }

  function afterReply(text, hook) {
    // free replay affordance (cached /chat/tts) + speechSynthesis fallback
    var holder = document.getElementById('replay');
    if (holder) holder.remove();
    holder = document.createElement('span'); holder.id = 'replay';
    sub.appendChild(holder);
    if (window.TTS && window.TTS.attach) { try { window.TTS.attach(holder, text); } catch (e) {} }
    if (hook) { endOnHook(); return; }
    send.disabled = false; listening(); say.focus();
  }

  function endOnHook() {
    state = 'ended';
    composer.classList.add('hidden');
    yes.style.display = 'inline-block';
    listening();
  }

  // --- Intro ---
  function startIntro() {
    overlay.remove();
    state = 'intro';
    setVideo(manifest.intro_video, {loop: false});
    stage.muted = false;
    stage.onended = function () { stage.muted = true; listening(); say.focus(); };
    // safety: if the intro can't play, fall straight to listening
    stage.play().catch(function () { listening(); say.focus(); });
  }

  // --- wire up ---
  composer.addEventListener('submit', function (e) {
    e.preventDefault();
    var m = say.value.trim();
    if (!m || state === 'pondering' || state === 'speaking' || state === 'ended') return;
    say.value = '';
    sendTurn(m);
  });
  say.addEventListener('input', armIdle);
  say.addEventListener('blur', function () { if (idleTimer) clearTimeout(idleTimer); });
  yes.addEventListener('click', function () {
    // End of this slice: the Remedy Match wiring is the next spec.
    sub.textContent = 'Then let us go and find it.';
  });
  beginBtn.addEventListener('click', startIntro);

  fetch('/static/fireside/fireside-manifest.json')
    .then(function (r) { return r.json(); })
    .then(function (m) { manifest = m; stage.poster = m.intro_poster; })
    .catch(function () { sub.textContent = 'The fire is not lit just now. Please return shortly.'; });
})();
</script>
</body>
</html>
```

- [ ] **Step 2: Quick syntax sanity (no test framework for HTML; verify it parses)**

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python3 -c "import pathlib,html.parser as h; p=h.HTMLParser(); p.feed(pathlib.Path('static/begin-fireside.html').read_text()); print('html ok')"`
Expected: `html ok` (no exception).

- [ ] **Step 3: Run the route tests to confirm the page still serves**

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python -m pytest tests/test_fireside_routes.py -q`
Expected: PASS (all — the serve test now sees the real `fireside` content).

- [ ] **Step 4: Commit**

```bash
cd /tmp/wt-deploy-chat-e5fec1df
git add static/begin-fireside.html
git commit -m "feat(fireside): full-screen fireside UI + presentation dance + interjections"
```

---

### Task 7: Render-verify the experience (headless browser)

**Files:**
- No new product files. A throwaway verify script + a manual headless run. (Render-verify is the project's discipline: assert real DOM + zero console errors — not just that the script was injected.)

**Interfaces:**
- Consumes: the running app with `FIRESIDE_ENABLED=true`, a mocked agent so no network/keys are needed.

- [ ] **Step 1: Start the app locally with the flag on and a stubbed agent**

Create `/tmp/fireside_verify_server.py`:

```python
# /tmp/fireside_verify_server.py  — run the app with the fireside agent stubbed
import os
os.environ["FIRESIDE_ENABLED"] = "true"
os.environ.setdefault("DATA_DIR", "/tmp/fireside_verify_data")
os.makedirs(os.environ["DATA_DIR"], exist_ok=True)

import app as appmod

# Stub the streaming model so verify needs no API key.
import types
class _S:
    def __init__(s, t): s._t = t
    def __enter__(s): return s
    def __exit__(s, *a): return False
    @property
    def text_stream(s):
        for x in s._t: yield x
class _M:
    def stream(s, **k): return _S(["I hear the weight in that. ", "Tell me when it began."])
appmod._cl = types.SimpleNamespace(messages=_M())
# Stub /chat/tts so no ElevenLabs key is needed: return a tiny silent mp3.
import subprocess, pathlib
sil = pathlib.Path("/tmp/fireside_verify_data/sil.mp3")
if not sil.exists():
    subprocess.run(["ffmpeg","-y","-f","lavfi","-i","anullsrc=r=44100:cl=mono",
                    "-t","0.4","-q:a","9",str(sil)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
_silbytes = sil.read_bytes()
@appmod.app.route("/__verify_noop")
def _noop(): return "ok"
# Monkeypatch the tts route's renderer by overriding _el_tts to return silence.
appmod._el_tts = lambda script: (_silbytes, None)

appmod.app.run(port=5099, threaded=True)
```

Run (background): `cd /tmp/wt-deploy-chat-e5fec1df && doppler run -p remedy-match -c prd -- python3 /tmp/fireside_verify_server.py &`
(Doppler supplies Pinecone/Anthropic keys needed at import; the model + tts are stubbed above so no real calls happen.)
Wait for: `Running on http://127.0.0.1:5099`.

- [ ] **Step 2: Drive it headless and assert DOM + zero console errors**

Use the Chrome automation tools (the project's render-verify discipline). Steps:
1. Navigate to `http://127.0.0.1:5099/begin/fireside`.
2. Assert `#begin-overlay` is present; click `#begin-btn`.
3. Assert the `<video id="stage">` has a `src` and `#say` is focusable.
4. Type into `#say` (`"I've been so tired and I don't know why"`) and submit.
5. Within ~3s assert `#subtitle` text contains `"I hear the weight in that."` (subtitles streamed).
6. Assert the `#stage` `data-src` transitions to the speaking loop after the reply, then back to a pondering loop (listening).
7. Read console messages — assert **zero** error-level entries.
8. To exercise the hook end-state, repeat sends until `turn_count >= 8` (the hard cap) OR temporarily set `appmod`’s eligibility by replaying; assert `#yes` becomes visible and `#composer` hides. (Acceptable to verify the hook branch via the unit test `test_agent_hides_hook_marker_and_flags_when_eligible` instead, and verify only the live happy-path + zero-console-errors here.)

Record the run as a short GIF (`fireside_render_verify.gif`) for the PR.

Expected: subtitles stream in, filler+pondering cover the gap, voice plays (silent stub) then returns to listening, **zero console errors**.

- [ ] **Step 3: Stop the verify server**

Run: `pkill -f fireside_verify_server.py || true`

- [ ] **Step 4: Full test sweep**

Run: `cd /tmp/wt-deploy-chat-e5fec1df && python -m pytest tests/test_fireside_store.py tests/test_fireside_agent.py tests/test_fireside_routes.py -q`
Expected: ALL PASS.

- [ ] **Step 5: Commit any verify artifacts (the GIF) if kept**

```bash
cd /tmp/wt-deploy-chat-e5fec1df
git add -A
git commit -m "test(fireside): headless render-verify (DOM + zero console errors)" --allow-empty
```

---

## Final integration checklist (before opening the PR)

- [ ] `FIRESIDE_ENABLED` unset → `/begin/fireside` and `/begin/fireside/agent` both 404 (run with the flag off once).
- [ ] `/begin/doorway` and the journey quest are untouched (`git diff --stat origin/main` shows only fireside files + the two app.py insertions).
- [ ] No SDD scratch leaked: `git diff --name-only origin/main..HEAD | grep -i superpowers/sdd` is empty (see `feedback_sdd_scratch_git_leak`).
- [ ] All three test files pass from repo root.
- [ ] Open the PR; ship dark (flag stays OFF in Render until Glen flips it). Note in the PR: real HeyGen intro + footage + clone-recorded fillers are swapped in by replacing the files named in `fireside-manifest.json` — no code change.

---

## Self-review (done against the spec)

- **§2 flow** (arrive→intro→listening→type→pondering+filler→speak→loop→hook): Tasks 4+6 implement the state machine; intro/listening/speaking/pondering all present. ✔
- **§3 reuse**: `/chat/tts` (Task 6), SSE pattern mirrored (Task 4), amg_session + chat_log.db (Tasks 1,4), ash_map pure fns (Tasks 2,4). Nothing rebuilt. ✔
- **§4 brain**: multi-turn, follows the thread, ASH coverage, model-decided hook → Task 2 persona + Task 4 wiring. ✔
- **§5 presentation dance**: instant filler + pondering loop + immediate subtitles + parallel TTS + speaking loop + return to listening + replay = Task 6 `sendTurn/streamReply/finishReply/afterReply`. Strategic interjections = `maybeInterject/armIdle`. ✔
- **§6 data**: `fireside_sessions` with all spec columns (transcript, ash_coverage, signals, lifecycle) = Task 1. ✔
- **§7 assets**: intro + speaking loop + pondering loop + filler repertoire + interjection subset, manifest-driven, placeholder-swappable = Task 5. ✔
- **§8 flag + verification**: `FIRESIDE_ENABLED` 404-when-off (Task 3); unit tests for store, agent endpoint (streamed turns, persistence, coverage, hook) (Tasks 1,2,4); render-verify (Task 7). ✔
- **§9 open items**: all six resolved in "Resolved open items" and baked into tasks. ✔
- **Placeholder scan**: every code step contains complete, runnable code; no TBD/TODO. ✔
- **Type consistency**: `fireside_store` signatures (Task 1 interfaces) match Task 4 calls; `fireside_agent` names (`FIRESIDE_MODEL`, `HOOK_SENTINEL`, `hook_eligible`, `parse_hook`, `build_system`, `build_messages`) match Task 4 usage; manifest keys (`intro_video`, `speaking_loop`, `pondering_loops`, `fillers`, `interjections`) match Task 6 JS. ✔
