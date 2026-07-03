# Portal Element-Engine Rewire Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the client portal home chat compute and persist each paid member's TCM five-element state, so a downstream feature (the Glendalf backdrop) can key off the element they need to nourish.

**Architecture:** Extract the existing TCM analysis engine out of the standalone Journal into a shared module. Add a per-member element-state store. Run the engine (text mode) over a rolling window of the member's recent portal-chat messages in the same fire-and-forget background lane the chat already uses for triage, gated to paid members. Expose the current element state on the portal API for the backdrop to read. The standalone Journal is untouched and stays Glen's personal tool.

**Tech Stack:** Python 3, Flask blueprints, sqlite (`chat_log.db` / `LOG_DB`), Anthropic Messages API (Haiku 4.5, forced tool-use), pytest.

## Global Constraints

- Repo: `~/deploy-chat`. Implement inside a session worktree (this repo is shared across sessions).
- Stores live in `dashboard/`, take a caller-supplied `sqlite3` connection, never open their own. JSON-shaped columns are stored as TEXT and decoded on read.
- Routes obtain a connection with `with _db_lock, sqlite3.connect(LOG_DB) as cx:` (writes) or `with sqlite3.connect(LOG_DB) as cx:` (reads), matching neighbours.
- Background work is fire-and-forget: `threading.Thread(target=..., daemon=True).start()`, wrapped so it never raises into the request.
- Element keys are exactly `Wood`, `Fire`, `Earth`, `Metal`, `Water`, scored numerically (the engine emits 0–100).
- Paid-membership gate is `_active_membership_for_email(email)` (the paid-coaching membership), NOT `is_member(...)` (that is the ToS/consent tier).
- Anthropic calls use forced tool-use (`tools` + `tool_choice`) so the API returns already-valid structured data. Never parse model free-text JSON as the primary path.
- Tests import modules directly and stub the network; they do NOT import `app` and do NOT need Doppler. Run with `python3 -m pytest tests/<file> -v` from the repo root.

---

### Task 1: Extract the TCM analysis engine into a shared module

Pull the element-analysis core out of `journal_blueprint.py` into `dashboard/tcm_analysis.py`, then re-import it back so the Journal keeps working with zero behaviour change. This is a pure move plus glue; existing Journal tests must stay green untouched.

**Files:**
- Create: `dashboard/tcm_analysis.py`
- Modify: `journal_blueprint.py` (remove the moved symbols, add a re-import)
- Test: `tests/test_tcm_analysis.py` (new), plus existing `tests/test_journal_haiku.py` must still pass

**Interfaces:**
- Produces (importable from `dashboard.tcm_analysis`): constants `ANTHROPIC_MESSAGES: str`, `HAIKU_MODEL: str`, `HUME_48_EMOTIONS: list[str]`, `HAIKU_SYSTEM_PROMPT: str`, `ANALYSIS_TOOL: dict`; functions `_haiku_analyze(transcript: str, lexical: dict) -> dict` and `_extract_json(text: str) -> dict | None`.
- `_haiku_analyze` returns a dict with at least `emotions`, `elements`, `treasures` (each a `{name: number}` map) on success.

- [ ] **Step 1: Create the shared module by moving the engine core**

Create `dashboard/tcm_analysis.py`. Move these symbols VERBATIM out of `journal_blueprint.py` into it (cut from the blueprint, paste here), preserving their exact bodies:
- `ANTHROPIC_MESSAGES` (currently `journal_blueprint.py:65`)
- `HAIKU_MODEL` (`journal_blueprint.py:66`)
- `HUME_48_EMOTIONS` (`journal_blueprint.py:582`–594)
- `HAIKU_SYSTEM_PROMPT` (`journal_blueprint.py:595`–643)
- `_NUM_MAP` and `ANALYSIS_TOOL` (`journal_blueprint.py:648`–674)
- `_haiku_analyze` (`journal_blueprint.py:675`–740)
- `_extract_json` (`journal_blueprint.py:743`–758)

Add this module header and imports at the top of the new file (the moved code uses `os`, `json`, `requests`):

```python
"""Shared TCM/emotional voice-analysis engine.

Extracted from journal_blueprint.py so both the standalone voice Journal and the
member portal chat can run the same Haiku element analysis. Pure functions +
constants only; no Flask, no store, no DB. Callers pass in transcript + lexical.
"""
import json
import os

import requests
```

Leave `_lexical_features`, `_whisper_transcribe`, `_embed_ada002`, and `_top_n_emotions` in `journal_blueprint.py` — the portal text path does not need them and moving them widens the blast radius.

- [ ] **Step 2: Re-import the moved symbols back into the blueprint**

In `journal_blueprint.py`, where the moved definitions used to be, add (near the other `from dashboard import ...` imports, e.g. after line 57):

```python
from dashboard.tcm_analysis import (
    ANTHROPIC_MESSAGES,
    HAIKU_MODEL,
    HUME_48_EMOTIONS,
    HAIKU_SYSTEM_PROMPT,
    ANALYSIS_TOOL,
    _haiku_analyze,
    _extract_json,
)
```

Confirm nothing else in `journal_blueprint.py` still redefines these names (the affirmations helper at line ~437 uses `HAIKU_MODEL`; it now resolves via the import — good).

- [ ] **Step 3: Run the existing Journal tests to prove no behaviour change**

Run: `python3 -m pytest tests/test_journal_haiku.py -v`
Expected: PASS (all 3 tests). They call `journal_blueprint._haiku_analyze` and monkeypatch `jb.requests.post`; because `requests` is a shared module singleton, patching it still intercepts the call now living in `tcm_analysis`.

- [ ] **Step 4: Write a direct test against the shared module**

Create `tests/test_tcm_analysis.py`:

```python
import json
from dashboard import tcm_analysis as tcm


class _Resp:
    def __init__(self, body):
        self._body = body
        self.ok = True
        self.status_code = 200
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _tool_body(analysis):
    return {"content": [{"type": "tool_use", "name": "emit_analysis", "input": analysis}],
            "stop_reason": "tool_use"}


def test_haiku_analyze_returns_parsed_elements(monkeypatch):
    analysis = {"emotions": {"Calmness": 0.7},
                "elements": {"Wood": 10, "Fire": 60, "Earth": 20, "Metal": 5, "Water": 5},
                "treasures": {"Qi": 55}}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(tcm.requests, "post", lambda *a, **k: _Resp(_tool_body(analysis)))
    out = tcm._haiku_analyze("today was heavy but hopeful", {"word_count": 6})
    assert out["elements"]["Fire"] == 60
    assert set(out["elements"]) == {"Wood", "Fire", "Earth", "Metal", "Water"}


def test_haiku_analyze_forces_the_tool(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return _Resp(_tool_body({"emotions": {}, "elements": {}, "treasures": {}}))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(tcm.requests, "post", fake_post)
    tcm._haiku_analyze("hello", {"word_count": 1})
    assert captured["payload"]["tool_choice"]["name"] == "emit_analysis"
```

- [ ] **Step 5: Run the new test**

Run: `python3 -m pytest tests/test_tcm_analysis.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add dashboard/tcm_analysis.py journal_blueprint.py tests/test_tcm_analysis.py
git commit -m "refactor: extract TCM analysis engine to dashboard/tcm_analysis.py"
```

---

### Task 2: Per-member element-state store

A single row per member holding their latest element scores plus the derived dominant and deficient elements. The deficient element (lowest score) is what to nourish and what the backdrop keys off.

**Files:**
- Create: `dashboard/member_element_state.py`
- Test: `tests/test_member_element_state.py`

**Interfaces:**
- Produces:
  - `init_table(cx) -> None`
  - `deficient_element(element_scores: dict) -> str | None` (lowest-scoring of the five; `None` if no usable scores)
  - `dominant_element(element_scores: dict) -> str | None` (highest-scoring)
  - `upsert(cx, email: str, element_scores: dict, source: str = "portal_chat") -> dict | None` (writes and returns the stored row)
  - `get(cx, email: str) -> dict | None` (row dict with `element_scores` decoded to a dict, plus `dominant_element`, `deficient_element`, `source`, `updated_at`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_member_element_state.py`:

```python
import sqlite3
from dashboard import member_element_state as mes


def _cx():
    cx = sqlite3.connect(":memory:")
    mes.init_table(cx)
    return cx


def test_deficient_is_lowest_scoring_element():
    scores = {"Wood": 80, "Fire": 60, "Earth": 40, "Metal": 20, "Water": 5}
    assert mes.deficient_element(scores) == "Water"
    assert mes.dominant_element(scores) == "Wood"


def test_deficient_handles_empty_or_garbage():
    assert mes.deficient_element({}) is None
    assert mes.deficient_element(None) is None
    assert mes.deficient_element({"Wood": "n/a"}) is None


def test_upsert_then_get_roundtrips_and_derives():
    cx = _cx()
    scores = {"Wood": 80, "Fire": 60, "Earth": 40, "Metal": 20, "Water": 5}
    row = mes.upsert(cx, "Jane@Example.com ", scores, source="portal_chat")
    assert row["deficient_element"] == "Water"
    got = mes.get(cx, "jane@example.com")
    assert got["element_scores"]["Fire"] == 60
    assert got["dominant_element"] == "Wood"
    assert got["source"] == "portal_chat"


def test_upsert_overwrites_same_email():
    cx = _cx()
    mes.upsert(cx, "j@x.com", {"Wood": 80, "Fire": 1, "Earth": 40, "Metal": 20, "Water": 50})
    mes.upsert(cx, "j@x.com", {"Wood": 1, "Fire": 80, "Earth": 40, "Metal": 20, "Water": 50})
    got = mes.get(cx, "j@x.com")
    assert got["deficient_element"] == "Wood"
    assert cx.execute("SELECT COUNT(*) FROM member_element_state").fetchone()[0] == 1


def test_get_missing_returns_none():
    assert mes.get(_cx(), "nobody@x.com") is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 -m pytest tests/test_member_element_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.member_element_state'`.

- [ ] **Step 3: Write the store**

Create `dashboard/member_element_state.py`:

```python
"""Per-member TCM five-element state (one row per email).

Written from the member's portal-chat analysis; read by the Glendalf backdrop
to pick the setting for the element they need to nourish (the deficient one).
Caller supplies the sqlite3 connection (same pattern as journal_store).
"""
import json
import sqlite3

_ELEMENTS = ("Wood", "Fire", "Earth", "Metal", "Water")


def init_table(cx):
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS member_element_state (
          email TEXT PRIMARY KEY,
          element_scores TEXT,
          dominant_element TEXT,
          deficient_element TEXT,
          source TEXT,
          updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )
        """
    )


def _scored(element_scores):
    if not isinstance(element_scores, dict):
        return {}
    return {k: element_scores[k] for k in _ELEMENTS
            if isinstance(element_scores.get(k), (int, float))}


def deficient_element(element_scores):
    """Lowest-scoring of the five elements = what to nourish. None if unusable."""
    scored = _scored(element_scores)
    return min(scored, key=scored.get) if scored else None


def dominant_element(element_scores):
    scored = _scored(element_scores)
    return max(scored, key=scored.get) if scored else None


def upsert(cx, email, element_scores, source="portal_chat"):
    email = (email or "").strip().lower()
    if not email:
        return None
    init_table(cx)
    dom = dominant_element(element_scores)
    dfc = deficient_element(element_scores)
    cx.execute(
        """
        INSERT INTO member_element_state
          (email, element_scores, dominant_element, deficient_element, source, updated_at)
        VALUES (?,?,?,?,?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        ON CONFLICT(email) DO UPDATE SET
          element_scores=excluded.element_scores,
          dominant_element=excluded.dominant_element,
          deficient_element=excluded.deficient_element,
          source=excluded.source,
          updated_at=excluded.updated_at
        """,
        (email, json.dumps(element_scores or {}), dom, dfc, source),
    )
    return get(cx, email)


def get(cx, email):
    email = (email or "").strip().lower()
    if not email:
        return None
    init_table(cx)
    cx.row_factory = sqlite3.Row
    row = cx.execute(
        "SELECT * FROM member_element_state WHERE email=?", (email,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["element_scores"] = json.loads(d["element_scores"]) if d.get("element_scores") else {}
    except Exception:
        d["element_scores"] = {}
    return d
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_member_element_state.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/member_element_state.py tests/test_member_element_state.py
git commit -m "feat: per-member TCM element-state store"
```

---

### Task 3: Refresh element state from the portal chat

An orchestration function that pulls the member's recent client messages, runs the engine in text mode over that rolling window, and upserts their element state. Then wire it into the portal-chat POST route as a paid-member-gated, fire-and-forget background thread.

**Files:**
- Create: `dashboard/portal_element.py`
- Modify: `app.py` (the `/api/portal/<token>/chat` POST handler, in the post-answer background block near `app.py:13497`–13511)
- Test: `tests/test_portal_element.py`

**Interfaces:**
- Consumes: `dashboard.tcm_analysis._haiku_analyze`, `dashboard.portal_chat.list_messages`, `dashboard.member_element_state.upsert`.
- Produces: `refresh(cx, email: str, window: int = 6) -> dict | None` — returns the stored element-state row, or `None` when there is too little text or the analysis yields no elements.

- [ ] **Step 1: Write the failing test**

Create `tests/test_portal_element.py`:

```python
import sqlite3
from dashboard import portal_element as pe
from dashboard import portal_chat, member_element_state as mes


def _cx():
    cx = sqlite3.connect(":memory:")
    portal_chat.init_table(cx)
    mes.init_table(cx)
    return cx


def _seed(cx, email, texts):
    for t in texts:
        portal_chat.add_message(cx, email, "client", t, author="You")


def test_refresh_writes_element_state_from_recent_client_msgs(monkeypatch):
    cx = _cx()
    _seed(cx, "j@x.com", [
        "I keep waking at 3am full of dread and I can't get warm.",
        "Everything feels like too much and I am exhausted to my bones.",
    ])
    monkeypatch.setattr(pe, "_haiku_analyze", lambda transcript, lexical: {
        "emotions": {}, "treasures": {},
        "elements": {"Wood": 40, "Fire": 30, "Earth": 20, "Metal": 25, "Water": 5},
    })
    row = pe.refresh(cx, "j@x.com")
    assert row is not None
    assert row["deficient_element"] == "Water"
    assert mes.get(cx, "j@x.com")["deficient_element"] == "Water"


def test_refresh_skips_when_too_little_text(monkeypatch):
    cx = _cx()
    _seed(cx, "j@x.com", ["ok"])
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("should not analyze")

    monkeypatch.setattr(pe, "_haiku_analyze", _boom)
    assert pe.refresh(cx, "j@x.com") is None
    assert called["n"] == 0


def test_refresh_returns_none_when_no_elements(monkeypatch):
    cx = _cx()
    _seed(cx, "j@x.com", ["I keep waking at 3am full of dread and cannot get warm at all."])
    monkeypatch.setattr(pe, "_haiku_analyze", lambda t, l: {"elements": {}})
    assert pe.refresh(cx, "j@x.com") is None


def test_refresh_only_uses_client_messages(monkeypatch):
    cx = _cx()
    _seed(cx, "j@x.com", ["I keep waking at 3am full of dread and cannot get warm at all."])
    portal_chat.add_message(cx, "j@x.com", "assistant", "IGNORE ME " * 20, author="Ask Dr. Glen")
    seen = {}
    monkeypatch.setattr(pe, "_haiku_analyze",
                        lambda transcript, lexical: seen.update(t=transcript) or {"elements": {"Water": 5, "Wood": 9}})
    pe.refresh(cx, "j@x.com")
    assert "IGNORE ME" not in seen["t"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 -m pytest tests/test_portal_element.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.portal_element'`.

- [ ] **Step 3: Write the orchestration module**

Create `dashboard/portal_element.py`:

```python
"""Refresh a member's TCM element state from their recent portal-chat messages.

Text mode: the portal chat is typed, so we run the engine over a rolling window
of the member's recent client messages (not one line — a single message carries
almost no elemental signal). Upgrades to full voice/lexical analysis for free
once portal voice-in ships; the engine call is identical.
"""
from dashboard.tcm_analysis import _haiku_analyze
from dashboard import portal_chat, member_element_state

_WINDOW = 6      # most-recent client messages to analyze together
_MIN_CHARS = 60  # below this the transcript is too thin to score meaningfully


def refresh(cx, email, window=_WINDOW):
    email = (email or "").strip().lower()
    if not email:
        return None
    msgs = portal_chat.list_messages(cx, email, limit=50) or []
    client_texts = [m["content"] for m in msgs
                    if m.get("role") == "client" and (m.get("content") or "").strip()]
    transcript = "\n".join(client_texts[-window:]).strip()
    if len(transcript) < _MIN_CHARS:
        return None
    haiku = _haiku_analyze(transcript, {"word_count": len(transcript.split())})
    elements = (haiku or {}).get("elements") or {}
    if not elements:
        return None
    return member_element_state.upsert(cx, email, elements, source="portal_chat")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_portal_element.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Wire the background refresh into the portal-chat route**

In `app.py`, in the `/api/portal/<token>/chat` POST handler, immediately after the triage thread block (the `_triage_portal_message` thread started near `app.py:13507`), add:

```python
        # Refresh the member's TCM element state from their recent chat (paid
        # members only; drives the Glendalf backdrop). Fire-and-forget, text mode.
        try:
            import threading as _t3

            def _refresh_element(em):
                try:
                    if not _active_membership_for_email(em):
                        return
                    from dashboard import portal_element
                    with _db_lock, sqlite3.connect(LOG_DB) as _ecx:
                        portal_element.refresh(_ecx, em)
                except Exception as e:
                    print(f"[portal-element] refresh failed: {e!r}", flush=True)

            _t3.Thread(target=_refresh_element, args=(email,), daemon=True).start()
        except Exception:
            pass
```

- [ ] **Step 6: Verify the app still imports cleanly**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('app.py parses')"`
Expected: `app.py parses` (syntax check without needing the app's runtime env).

- [ ] **Step 7: Commit**

```bash
git add dashboard/portal_element.py tests/test_portal_element.py app.py
git commit -m "feat: refresh member TCM element state from portal chat (paid-gated)"
```

---

### Task 4: Expose the element state on the portal API

Add the member's current element state (including the deficient element and its lowercased setting name) to the `GET /api/portal/<token>` response so the Glendalf backdrop can read it.

**Files:**
- Modify: `app.py` (the `api_client_portal` route at `app.py:13292`)
- Test: `tests/test_portal_element_api.py`

**Interfaces:**
- Consumes: `dashboard.member_element_state.get`.
- Produces: an `element_state` key on the portal JSON: either `null`, or `{"element_scores": {...}, "dominant_element": "...", "deficient_element": "Water", "setting": "water", "source": "...", "updated_at": "..."}`.

- [ ] **Step 1: Write the failing test for the setting-name shaping helper**

Create `tests/test_portal_element_api.py`:

```python
import sqlite3
from dashboard import member_element_state as mes
from dashboard import portal_element_view as pev


def _cx():
    cx = sqlite3.connect(":memory:")
    mes.init_table(cx)
    return cx


def test_view_adds_lowercased_setting():
    cx = _cx()
    mes.upsert(cx, "j@x.com", {"Wood": 80, "Fire": 60, "Earth": 40, "Metal": 20, "Water": 5})
    view = pev.element_view(cx, "j@x.com")
    assert view["deficient_element"] == "Water"
    assert view["setting"] == "water"


def test_view_is_none_when_no_state():
    assert pev.element_view(_cx(), "nobody@x.com") is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 -m pytest tests/test_portal_element_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.portal_element_view'`.

- [ ] **Step 3: Write the view helper**

Create `dashboard/portal_element_view.py`:

```python
"""Shape a member's element state for the portal API: adds `setting` (the
lowercased deficient element = the Glendalf backdrop to show)."""
from dashboard import member_element_state


def element_view(cx, email):
    row = member_element_state.get(cx, email)
    if not row:
        return None
    dfc = row.get("deficient_element")
    row["setting"] = dfc.lower() if dfc else None
    return row
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_portal_element_api.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Add `element_state` to the portal API response**

In `app.py`, inside `api_client_portal` (route at `app.py:13292`), after `email` is resolved from `portal` and while the `cx` from `with sqlite3.connect(LOG_DB) as cx:` is still open, build the view and include it in the response dict. Add near where the response is assembled:

```python
        try:
            from dashboard import portal_element_view as _pev
            element_state = _pev.element_view(cx, (portal.get("email") or "").strip().lower())
        except Exception:
            element_state = None
```

Then add `"element_state": element_state,` to the `jsonify({...})` payload this route returns. (If the `cx` block has already closed by the response-building point, open a fresh `with sqlite3.connect(LOG_DB) as cx:` for the lookup, matching the read pattern used elsewhere in the route.)

- [ ] **Step 6: Verify the app still parses**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('app.py parses')"`
Expected: `app.py parses`.

- [ ] **Step 7: Run the full new-test suite together**

Run: `python3 -m pytest tests/test_tcm_analysis.py tests/test_member_element_state.py tests/test_portal_element.py tests/test_portal_element_api.py tests/test_journal_haiku.py -v`
Expected: PASS (all).

- [ ] **Step 8: Commit**

```bash
git add dashboard/portal_element_view.py tests/test_portal_element_api.py app.py
git commit -m "feat: expose member element_state (+ backdrop setting) on portal API"
```

---

## Out of scope (deliberately deferred)

- **The Glendalf backdrop render itself** (swapping the video/scene by `setting`). Needs the four new element settings produced (Water/Earth/Metal/Wood; Fire = fireside exists). This plan only produces the signal that drives it.
- **Voice-in / full lexical fidelity.** V1 is text mode. When portal tap-to-talk ships, pass the real Whisper transcript + `_lexical_features(...)` into `portal_element.refresh` instead of the text window; the engine call is unchanged.
- **Gating the standalone Journal.** It stays Glen's personal tool, untouched.
- **Deficient-element TCM refinement** (mother-element / generating-cycle logic). V1 uses the simple lowest-score element. Revisit if the backdrop feels off.

## Self-review notes

- Spec coverage: engine extraction (Task 1), per-member store (Task 2), feed-from-portal-chat = the reshaped Option A (Task 3), backdrop read (Task 4). The element→backdrop mapping keys off the deficient element per the spec's chosen mapping.
- Paid-membership gate uses `_active_membership_for_email`, matching the spec's "activated with paid membership."
- Type consistency: `element_scores` is a `{name: number}` dict throughout; `deficient_element`/`dominant_element` return element-name strings or `None`; `refresh`/`upsert`/`get`/`element_view` signatures are consistent across tasks and tests.
