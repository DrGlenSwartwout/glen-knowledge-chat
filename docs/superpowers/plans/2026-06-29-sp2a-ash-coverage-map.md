# SP2a — ASH Coverage Map + Ally Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `dashboard/ash_map.py` — the durable, email-keyed ASH 12-dimension coverage map + ally memory + a per-turn Haiku updater that fills it — fully testable with no app/Flask import.

**Architecture:** One pure module over a caller-supplied `sqlite3.Connection`. A constant `ASH_DIMENSIONS` defines the 12 canonical dimensions. `get`/`init_table` mirror `dashboard/journal_store.py` storage. `merge_turn` is a pure forward-only state merge. `_haiku_extract` is one self-contained LLM call mirroring `journal_blueprint._haiku_analyze` (forced `tool_choice` structured output) but **never raises** — it returns an empty default on any error, because it runs fire-and-forget after an ally reply. `update_from_turn` orchestrates get→extract→merge→persist. `context_block` formats memory for an ally system prompt.

**Tech Stack:** Python 3, stdlib `sqlite3`/`json`/`datetime`, `requests` (module-level import, for the test monkeypatch), Anthropic Messages API with Haiku 4.5.

## Global Constraints

- New module path: `dashboard/ash_map.py`. New test: `tests/test_ash_map.py`.
- Pure-module: the module MUST NOT import `app`, `begin_funnel`, or Flask. All DB functions take a caller-supplied `cx` (`sqlite3.Connection`); the module never opens its own connection and needs no `_db_lock`.
- `import requests` at module top-level (tests monkeypatch `ash_map.requests.post`, mirroring `tests/test_journal_haiku.py`).
- Haiku model constant: `HAIKU_MODEL = "claude-haiku-4-5-20251001"`. Endpoint: `ANTHROPIC_MESSAGES = "https://api.anthropic.com/v1/messages"`. Header `anthropic-version: 2023-06-01`. API key from `os.environ["ANTHROPIC_API_KEY"]`.
- The 12 canonical dimension keys, in order: `body, mind, spirit, inheritance, personal_history, epigenetics, symptoms, terrain, diagnosis, treatment, regulation, prognosis`.
- State ladder (monotonic forward-only): `untouched(0) < opened(1) < explored(2) < deep(3)`.
- Email is normalized lowercased + stripped (`_norm_email`) everywhere it is stored or looked up.
- Tests run with plain pytest, no doppler, no network (`python3 -m pytest tests/test_ash_map.py -v`).
- `_haiku_extract` and `update_from_turn` MUST NOT raise on LLM/HTTP error — they degrade to the empty default `{"dimensions": {}, "summary": ""}`.

---

### Task 1: Constants, blank map, and email normalization

**Files:**
- Create: `dashboard/ash_map.py`
- Test: `tests/test_ash_map.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ASH_DIMENSIONS: list[dict]` — 12 ordered `{"key": str, "name": str, "meaning": str}`.
  - `DIM_KEYS: list[str]` — the 12 keys in order.
  - `STATE_ORDER: dict[str,int]` — `{"untouched":0,"opened":1,"explored":2,"deep":3}`.
  - `_norm_email(email: str) -> str` — lowercased, stripped.
  - `_blank_map() -> dict` — `{key: {"state":"untouched","opened_excerpt":"","notes":"","last_touched_at":None}}` for all 12 keys. Returns a fresh dict each call (no shared mutable state).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ash_map.py
import dashboard.ash_map as am


def test_twelve_canonical_dimensions_in_order():
    assert am.DIM_KEYS == [
        "body", "mind", "spirit", "inheritance", "personal_history",
        "epigenetics", "symptoms", "terrain", "diagnosis", "treatment",
        "regulation", "prognosis",
    ]
    # ASH_DIMENSIONS carries key/name/meaning for each, in the same order
    assert [d["key"] for d in am.ASH_DIMENSIONS] == am.DIM_KEYS
    for d in am.ASH_DIMENSIONS:
        assert d["name"] and d["meaning"]


def test_state_order_ladder():
    assert am.STATE_ORDER == {"untouched": 0, "opened": 1, "explored": 2, "deep": 3}


def test_norm_email():
    assert am._norm_email("  Foo@Bar.COM ") == "foo@bar.com"


def test_blank_map_has_all_twelve_untouched_and_is_fresh():
    m = am._blank_map()
    assert set(m.keys()) == set(am.DIM_KEYS)
    for k in am.DIM_KEYS:
        assert m[k] == {
            "state": "untouched", "opened_excerpt": "",
            "notes": "", "last_touched_at": None,
        }
    # fresh dict each call — mutating one does not leak into the next
    m["body"]["notes"] = "x"
    assert am._blank_map()["body"]["notes"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ash_map.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.ash_map'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/ash_map.py
"""Email-keyed ASH coverage map + ally memory (SP2a).

A durable per-person record of which of the 12 ASH dimensions a conversation
has touched, plus a rolling "who they are" summary and verbatim opening
excerpts. Pure module: all DB functions take a caller-supplied sqlite3
connection (no app/Flask import), mirroring dashboard/journal_store.py. The
per-turn updater (_haiku_extract) mirrors journal_blueprint._haiku_analyze's
forced-tool-use structured output, but never raises — it runs fire-and-forget
after an ally reply, so any failure degrades to "learned nothing this turn".
"""
import json
import os
import sqlite3
from datetime import datetime, timezone

import requests  # module-level so tests can monkeypatch ash_map.requests.post

ANTHROPIC_MESSAGES = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# The 12 ASH dimensions — canonical keys, display names, and one-line meanings
# the updater prompt renders so Haiku can map turn content -> dimensions.
ASH_DIMENSIONS = [
    {"key": "body", "name": "Body / States of Matter",
     "meaning": "the physical body's substance, density, structure"},
    {"key": "mind", "name": "Mind / 5 C's",
     "meaning": "mental focus, emotional patterns, how they connect and communicate"},
    {"key": "spirit", "name": "Spirit / 5 Elements",
     "meaning": "meaning, purpose, emotional-elemental balance"},
    {"key": "inheritance", "name": "Inheritance / 5 Generations",
     "meaning": "family, genetic, lineage health patterns"},
    {"key": "personal_history", "name": "Personal History / 5 Penetration",
     "meaning": "their own health history and how deep issues have gone"},
    {"key": "epigenetics", "name": "Epigenetics / 5 Infoceuticals",
     "meaning": "bioenergetic / informational regulation (terrain, organs, meridians, systems)"},
    {"key": "symptoms", "name": "Symptoms / 5 Cardinal Signs",
     "meaning": "active symptoms: pain, heat, swelling, redness, loss of function"},
    {"key": "terrain", "name": "Terrain / 5 R's",
     "meaning": "the body's vitality and capacity to heal"},
    {"key": "diagnosis", "name": "Diagnosis / 5 Pathology Types",
     "meaning": "diagnosed conditions or tissue changes"},
    {"key": "treatment", "name": "Treatment / 5 Therapy Levels",
     "meaning": "treatments they use and how invasive vs. supportive"},
    {"key": "regulation", "name": "Regulation / 5 Levels",
     "meaning": "how the body responds when they try to heal"},
    {"key": "prognosis", "name": "Prognosis / 5 Stages",
     "meaning": "seriousness or trajectory of their main concern"},
]
DIM_KEYS = [d["key"] for d in ASH_DIMENSIONS]

STATE_ORDER = {"untouched": 0, "opened": 1, "explored": 2, "deep": 3}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%fZ")


def _norm_email(email: str) -> str:
    return (email or "").strip().lower()


def _blank_map() -> dict:
    return {
        k: {"state": "untouched", "opened_excerpt": "",
            "notes": "", "last_touched_at": None}
        for k in DIM_KEYS
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ash_map.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/ash_map.py tests/test_ash_map.py
git commit -m "feat(ash_map): SP2a constants, blank map, email norm"
```

---

### Task 2: `merge_turn` — pure forward-only state merge

**Files:**
- Modify: `dashboard/ash_map.py`
- Test: `tests/test_ash_map.py`

**Interfaces:**
- Consumes: `_blank_map`, `_now_iso`, `STATE_ORDER`, `DIM_KEYS` from Task 1.
- Produces:
  - `merge_turn(memory: dict, updater_output: dict) -> dict` — PURE. `memory` is `{email?, summary, dimensions:{...12...}, created_at?, updated_at?}` (or anything with a `dimensions` dict + `summary`); `updater_output` is `{"dimensions": {<key>: {"state","excerpt","notes"}}, "summary": str}`. Returns a NEW memory dict; does not mutate inputs. Rules:
    - For each touched dimension: `state = max(current, proposed)` by `STATE_ORDER` (forward-only, never downgrades); set `opened_excerpt` only if currently empty (and proposed excerpt non-empty); append `notes` delta joined by `\n`, skipping a delta line already present verbatim; stamp `last_touched_at = _now_iso()`.
    - Unknown/invalid dimension keys in `updater_output` are ignored.
    - Untouched dimensions are left exactly as-is.
    - `summary`: replace with `updater_output["summary"]` when non-empty/truthy; otherwise keep the prior summary.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ash_map.py
import copy


def _mem(summary="", dims=None):
    m = {"summary": summary, "dimensions": am._blank_map()}
    for k, v in (dims or {}).items():
        m["dimensions"][k].update(v)
    return m


def test_merge_bumps_state_forward_and_sets_excerpt_once():
    mem = _mem()
    out = {"dimensions": {"symptoms": {"state": "opened",
            "excerpt": "my knee aches every morning", "notes": "AM knee pain"}},
           "summary": "Cautious, in pain."}
    merged = am.merge_turn(mem, out)
    s = merged["dimensions"]["symptoms"]
    assert s["state"] == "opened"
    assert s["opened_excerpt"] == "my knee aches every morning"
    assert s["notes"] == "AM knee pain"
    assert s["last_touched_at"] is not None
    assert merged["summary"] == "Cautious, in pain."
    # untouched dims stay untouched
    assert merged["dimensions"]["body"]["state"] == "untouched"


def test_merge_never_downgrades_and_preserves_first_excerpt():
    mem = _mem(dims={"symptoms": {"state": "deep",
        "opened_excerpt": "first words", "notes": "old"}})
    out = {"dimensions": {"symptoms": {"state": "opened",
            "excerpt": "second words", "notes": "new detail"}}, "summary": ""}
    merged = am.merge_turn(mem, out)
    s = merged["dimensions"]["symptoms"]
    assert s["state"] == "deep"               # max(deep, opened) = deep, no downgrade
    assert s["opened_excerpt"] == "first words"  # excerpt set once, preserved
    assert s["notes"] == "old\nnew detail"    # appended


def test_merge_dedupes_identical_note_line():
    mem = _mem(dims={"terrain": {"state": "explored", "notes": "low vitality"}})
    out = {"dimensions": {"terrain": {"state": "explored",
            "excerpt": "", "notes": "low vitality"}}, "summary": ""}
    merged = am.merge_turn(mem, out)
    assert merged["dimensions"]["terrain"]["notes"] == "low vitality"  # not duplicated


def test_merge_empty_summary_preserves_prior_and_input_not_mutated():
    mem = _mem(summary="Prior who-they-are.")
    snapshot = copy.deepcopy(mem)
    out = {"dimensions": {"mind": {"state": "opened", "excerpt": "", "notes": "n"}},
           "summary": ""}
    merged = am.merge_turn(mem, out)
    assert merged["summary"] == "Prior who-they-are."  # empty summary keeps prior
    assert mem == snapshot                             # input untouched


def test_merge_ignores_unknown_dimension_keys():
    mem = _mem()
    out = {"dimensions": {"not_a_dim": {"state": "deep", "excerpt": "x", "notes": "y"}},
           "summary": ""}
    merged = am.merge_turn(mem, out)
    assert "not_a_dim" not in merged["dimensions"]
    assert all(merged["dimensions"][k]["state"] == "untouched" for k in am.DIM_KEYS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ash_map.py -k merge -v`
Expected: FAIL — `AttributeError: module 'dashboard.ash_map' has no attribute 'merge_turn'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/ash_map.py
import copy as _copy


def merge_turn(memory: dict, updater_output: dict) -> dict:
    """Apply one updater result to a memory, PURELY. Forward-only state ladder,
    set-once excerpt, deduped note accumulation. Returns a new dict."""
    merged = _copy.deepcopy(memory)
    dims = merged.setdefault("dimensions", _blank_map())
    now = _now_iso()

    for key, delta in (updater_output.get("dimensions") or {}).items():
        if key not in DIM_KEYS or not isinstance(delta, dict):
            continue
        cell = dims.setdefault(key, {
            "state": "untouched", "opened_excerpt": "",
            "notes": "", "last_touched_at": None})

        proposed = delta.get("state", "untouched")
        cur_rank = STATE_ORDER.get(cell.get("state", "untouched"), 0)
        prop_rank = STATE_ORDER.get(proposed, 0)
        if prop_rank > cur_rank:
            cell["state"] = proposed

        excerpt = (delta.get("excerpt") or "").strip()
        if excerpt and not cell.get("opened_excerpt"):
            cell["opened_excerpt"] = excerpt

        note = (delta.get("notes") or "").strip()
        if note:
            existing = cell.get("notes", "")
            existing_lines = existing.split("\n") if existing else []
            if note not in existing_lines:
                existing_lines.append(note)
                cell["notes"] = "\n".join(line for line in existing_lines if line)

        cell["last_touched_at"] = now

    new_summary = (updater_output.get("summary") or "").strip()
    if new_summary:
        merged["summary"] = new_summary

    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ash_map.py -k merge -v`
Expected: PASS (5 merge tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/ash_map.py tests/test_ash_map.py
git commit -m "feat(ash_map): pure forward-only merge_turn"
```

---

### Task 3: `init_table` + `get` + upsert persistence round-trip

**Files:**
- Modify: `dashboard/ash_map.py`
- Test: `tests/test_ash_map.py`

**Interfaces:**
- Consumes: `_blank_map`, `_norm_email`, `_now_iso`, `DIM_KEYS` from Task 1.
- Produces:
  - `init_table(cx) -> None` — `CREATE TABLE IF NOT EXISTS ash_ally_memory (...)` per spec; commits.
  - `get(cx, email) -> dict` — calls `init_table`; returns `{email, summary, dimensions:{<all 12>}, created_at, updated_at}`. A never-seen email returns an all-untouched skeleton (NOT persisted): `{email:<norm>, summary:"", dimensions:_blank_map(), created_at:None, updated_at:None}`. A stored row's `dimensions_json` is decoded and any missing keys are backfilled from `_blank_map()` so callers always see all 12.
  - `_upsert(cx, email, summary, dimensions) -> None` — internal: insert-or-replace, preserving `created_at` on update, stamping `updated_at = _now_iso()`. (Used by Task 5.)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ash_map.py
import sqlite3


def _cx():
    return sqlite3.connect(":memory:")


def test_get_unseen_email_returns_all_untouched_skeleton():
    cx = _cx()
    m = am.get(cx, "  New@User.com ")
    assert m["email"] == "new@user.com"
    assert m["summary"] == ""
    assert set(m["dimensions"].keys()) == set(am.DIM_KEYS)
    assert all(m["dimensions"][k]["state"] == "untouched" for k in am.DIM_KEYS)
    assert m["created_at"] is None and m["updated_at"] is None


def test_upsert_then_get_round_trips_and_backfills_missing_keys():
    cx = _cx()
    am.init_table(cx)
    dims = am._blank_map()
    dims["symptoms"].update({"state": "opened", "opened_excerpt": "knee", "notes": "AM"})
    # store a PARTIAL dimensions map (only one key) to prove get() backfills the rest
    am._upsert(cx, "a@b.com", "A summary.", {"symptoms": dims["symptoms"]})
    got = am.get(cx, "A@B.com")
    assert got["summary"] == "A summary."
    assert got["dimensions"]["symptoms"]["state"] == "opened"
    assert got["dimensions"]["symptoms"]["opened_excerpt"] == "knee"
    # the other 11 keys are backfilled as untouched
    assert got["dimensions"]["body"]["state"] == "untouched"
    assert set(got["dimensions"].keys()) == set(am.DIM_KEYS)
    assert got["created_at"] and got["updated_at"]


def test_upsert_preserves_created_at_on_update():
    cx = _cx()
    am.init_table(cx)
    am._upsert(cx, "a@b.com", "first", am._blank_map())
    first = am.get(cx, "a@b.com")["created_at"]
    am._upsert(cx, "a@b.com", "second", am._blank_map())
    again = am.get(cx, "a@b.com")
    assert again["summary"] == "second"
    assert again["created_at"] == first  # created_at preserved across updates
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ash_map.py -k "get or upsert" -v`
Expected: FAIL — `AttributeError: module 'dashboard.ash_map' has no attribute 'get'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/ash_map.py
def init_table(cx) -> None:
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS ash_ally_memory (
          email           TEXT PRIMARY KEY,
          summary         TEXT NOT NULL DEFAULT '',
          dimensions_json TEXT NOT NULL DEFAULT '{}',
          created_at      TEXT NOT NULL,
          updated_at      TEXT NOT NULL
        )
        """
    )
    cx.commit()


def _full_dimensions(stored: dict) -> dict:
    """Backfill any missing of the 12 keys from a blank map so callers see all 12."""
    full = _blank_map()
    for k, v in (stored or {}).items():
        if k in full and isinstance(v, dict):
            full[k].update(v)
    return full


def get(cx, email: str) -> dict:
    init_table(cx)
    em = _norm_email(email)
    cx.row_factory = sqlite3.Row
    row = cx.execute(
        "SELECT summary, dimensions_json, created_at, updated_at "
        "FROM ash_ally_memory WHERE email = ?", (em,)
    ).fetchone()
    if row is None:
        return {"email": em, "summary": "", "dimensions": _blank_map(),
                "created_at": None, "updated_at": None}
    try:
        stored = json.loads(row["dimensions_json"]) or {}
    except (ValueError, TypeError):
        stored = {}
    return {
        "email": em,
        "summary": row["summary"] or "",
        "dimensions": _full_dimensions(stored),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _upsert(cx, email: str, summary: str, dimensions: dict) -> None:
    init_table(cx)
    em = _norm_email(email)
    now = _now_iso()
    existing = cx.execute(
        "SELECT created_at FROM ash_ally_memory WHERE email = ?", (em,)
    ).fetchone()
    created_at = existing[0] if existing else now
    cx.execute(
        "INSERT OR REPLACE INTO ash_ally_memory "
        "(email, summary, dimensions_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (em, summary or "", json.dumps(dimensions or {}), created_at, now),
    )
    cx.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ash_map.py -k "get or upsert" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/ash_map.py tests/test_ash_map.py
git commit -m "feat(ash_map): init_table, get skeleton/backfill, upsert round-trip"
```

---

### Task 4: `context_block` — format memory for an ally system prompt

**Files:**
- Modify: `dashboard/ash_map.py`
- Test: `tests/test_ash_map.py`

**Interfaces:**
- Consumes: `_blank_map`, `ASH_DIMENSIONS`, `DIM_KEYS` from Task 1.
- Produces:
  - `context_block(memory: dict) -> str` — formats memory for an ally system prompt. An empty/all-untouched memory yields a single line: `"This is your first conversation with them — nothing covered yet."`. Otherwise emits up to four labeled sections (omitting any that are empty):
    - `Who they are: <summary>` (only if summary non-empty)
    - `Already explored (do not re-ask): <name>: <notes>` per dim whose state is `explored` or `deep` (notes flattened to single-spaced).
    - `Opened, go deeper when they return to it: <name>: "<opened_excerpt>"` per dim whose state is `opened`.
    - `Not yet touched: <comma-joined names>` for `untouched` dims.
  - Dimension display names come from `ASH_DIMENSIONS` (the `name` field).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ash_map.py
def test_context_block_first_conversation_line():
    mem = {"summary": "", "dimensions": am._blank_map()}
    assert am.context_block(mem) == (
        "This is your first conversation with them — nothing covered yet."
    )


def test_context_block_populated_sections():
    mem = {"summary": "A tired caregiver in pain.", "dimensions": am._blank_map()}
    mem["dimensions"]["symptoms"].update(
        {"state": "explored", "notes": "AM knee pain\nworse in cold"})
    mem["dimensions"]["terrain"].update(
        {"state": "opened", "opened_excerpt": "I just have no energy left"})
    block = am.context_block(mem)
    assert "Who they are: A tired caregiver in pain." in block
    assert "Already explored (do not re-ask):" in block
    assert "AM knee pain worse in cold" in block          # notes flattened
    assert "Opened, go deeper when they return to it:" in block
    assert '"I just have no energy left"' in block
    assert "Not yet touched:" in block
    # an untouched dim's display name appears in the not-yet-touched list
    assert "Body / States of Matter" in block
    # touched dims are NOT in the not-yet-touched list
    not_touched_line = [l for l in block.splitlines() if l.startswith("Not yet touched:")][0]
    assert "Symptoms / 5 Cardinal Signs" not in not_touched_line
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ash_map.py -k context -v`
Expected: FAIL — `AttributeError: module 'dashboard.ash_map' has no attribute 'context_block'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/ash_map.py
_DIM_NAME = {d["key"]: d["name"] for d in ASH_DIMENSIONS}


def context_block(memory: dict) -> str:
    dims = (memory or {}).get("dimensions") or _blank_map()
    summary = ((memory or {}).get("summary") or "").strip()

    explored, opened, untouched = [], [], []
    for k in DIM_KEYS:
        cell = dims.get(k, {})
        state = cell.get("state", "untouched")
        name = _DIM_NAME[k]
        if state in ("explored", "deep"):
            notes = " ".join((cell.get("notes") or "").split())
            explored.append(f"{name}: {notes}".rstrip(": ").rstrip())
        elif state == "opened":
            ex = (cell.get("opened_excerpt") or "").strip()
            opened.append(f'{name}: "{ex}"' if ex else name)
        else:
            untouched.append(name)

    if not summary and not explored and not opened:
        return "This is your first conversation with them — nothing covered yet."

    lines = []
    if summary:
        lines.append(f"Who they are: {summary}")
    if explored:
        lines.append("Already explored (do not re-ask): "
                     + "; ".join(explored))
    if opened:
        lines.append("Opened, go deeper when they return to it: "
                     + "; ".join(opened))
    if untouched:
        lines.append("Not yet touched: " + ", ".join(untouched))
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ash_map.py -k context -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/ash_map.py tests/test_ash_map.py
git commit -m "feat(ash_map): context_block ally-prompt formatter"
```

---

### Task 5: `_haiku_extract` + `update_from_turn` — the LLM updater + orchestrator

**Files:**
- Modify: `dashboard/ash_map.py`
- Test: `tests/test_ash_map.py`

**Interfaces:**
- Consumes: `ASH_DIMENSIONS`, `HAIKU_MODEL`, `ANTHROPIC_MESSAGES`, `requests`, `get`, `merge_turn`, `_upsert` from earlier tasks.
- Produces:
  - `COVERAGE_TOOL: dict` — Anthropic tool def named `emit_coverage`; `input_schema` matches the spec's updater output (`dimensions` = object mapping dim keys to `{state(enum opened/explored/deep), excerpt(string), notes(string)}`; `summary` string). `required: ["dimensions"]`.
  - `_haiku_extract(memory: dict, user_text: str, ally_text: str = "") -> dict` — builds a Haiku request forcing `tool_choice=emit_coverage`, returns the parsed tool input `{"dimensions":{...},"summary":...}`. NEVER raises: missing API key, non-ok HTTP, exception, or a body with no `emit_coverage` tool_use block all return `{"dimensions": {}, "summary": ""}`.
  - `update_from_turn(cx, email, user_text, ally_text="") -> dict` — `get` → `_haiku_extract` → `merge_turn` → `_upsert` → return the merged memory (with `email`). Fire-and-forget-safe (inherits `_haiku_extract`'s no-raise behavior; persistence still happens even if extract returned the empty default — a no-op merge).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ash_map.py
class _Resp:
    def __init__(self, body, ok=True, status=200):
        self._b, self.ok, self.status_code = body, ok, status
        self.text = "x"

    def json(self):
        return self._b


def _coverage_body(payload):
    return {"content": [{"type": "tool_use", "name": "emit_coverage", "input": payload}]}


def test_haiku_extract_forces_tool_and_parses(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return _Resp(_coverage_body(
            {"dimensions": {"symptoms": {"state": "opened",
                "excerpt": "knee aches", "notes": "AM knee pain"}},
             "summary": "In pain."}))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(am.requests, "post", fake_post)
    out = am._haiku_extract(am._blank_map(), "my knee aches", "")
    assert out["dimensions"]["symptoms"]["state"] == "opened"
    assert out["summary"] == "In pain."
    p = captured["payload"]
    assert any(t.get("name") == "emit_coverage" for t in p["tools"])
    assert p["tool_choice"]["name"] == "emit_coverage"


def test_haiku_extract_no_key_returns_empty_default(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = am._haiku_extract(am._blank_map(), "hi", "")
    assert out == {"dimensions": {}, "summary": ""}


def test_haiku_extract_bad_response_returns_empty_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(am.requests, "post",
                        lambda *a, **k: _Resp({"content": []}, ok=True))
    out = am._haiku_extract(am._blank_map(), "hi", "")
    assert out == {"dimensions": {}, "summary": ""}


def test_haiku_extract_http_error_returns_empty_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(am.requests, "post", boom)
    assert am._haiku_extract(am._blank_map(), "hi", "") == {"dimensions": {}, "summary": ""}


def test_update_from_turn_persists_and_accumulates(monkeypatch):
    cx = sqlite3.connect(":memory:")
    seq = [
        {"dimensions": {"symptoms": {"state": "opened",
            "excerpt": "knee aches morning", "notes": "AM knee"},
                        "terrain": {"state": "opened", "excerpt": "", "notes": "low energy"}},
         "summary": "Turn one."},
        {"dimensions": {"symptoms": {"state": "deep",
            "excerpt": "ignored second excerpt", "notes": "worse in cold"},
                        "inheritance": {"state": "opened", "excerpt": "", "notes": "mother had it"}},
         "summary": "Turn two."},
    ]
    calls = {"i": 0}

    def fake_extract(memory, user_text, ally_text=""):
        out = seq[calls["i"]]
        calls["i"] += 1
        return out

    monkeypatch.setattr(am, "_haiku_extract", fake_extract)

    m1 = am.update_from_turn(cx, "u@x.com", "my knee aches and I'm wiped", "")
    assert m1["dimensions"]["symptoms"]["state"] == "opened"
    assert m1["dimensions"]["terrain"]["state"] == "opened"
    assert m1["summary"] == "Turn one."

    m2 = am.update_from_turn(cx, "u@x.com", "it's worse in the cold; mom had it too", "")
    # turn-1 excerpt preserved, state deepened, notes accumulated
    assert m2["dimensions"]["symptoms"]["state"] == "deep"
    assert m2["dimensions"]["symptoms"]["opened_excerpt"] == "knee aches morning"
    assert "AM knee" in m2["dimensions"]["symptoms"]["notes"]
    assert "worse in cold" in m2["dimensions"]["symptoms"]["notes"]
    assert m2["dimensions"]["inheritance"]["state"] == "opened"
    assert m2["summary"] == "Turn two."
    # untouched dims still untouched
    assert m2["dimensions"]["body"]["state"] == "untouched"
    # persisted: a fresh get sees turn-2 state
    assert am.get(cx, "u@x.com")["dimensions"]["symptoms"]["state"] == "deep"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ash_map.py -k "extract or update_from_turn" -v`
Expected: FAIL — `AttributeError: module 'dashboard.ash_map' has no attribute '_haiku_extract'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/ash_map.py
_DIM_LIST_FOR_PROMPT = "\n".join(
    f"  {d['key']}: {d['name']} — {d['meaning']}" for d in ASH_DIMENSIONS
)

_EXTRACT_SYSTEM = (
    "You quietly maintain a private health-conversation coverage map across 12 "
    "dimensions. Given the latest exchange and what is already known, report ONLY "
    "the dimensions this turn genuinely touched. Never invent; prefer fewer "
    "dimensions. For each touched dimension give: state (opened = first surfaced, "
    "explored = real detail given, deep = worked through), excerpt = the person's "
    "OWN words that opened/deepened it (or '' if none), notes = what was learned "
    "this turn. Also refresh a 1-2 sentence 'who they are' summary, or '' to keep "
    "the prior one.\n\nThe 12 dimensions:\n" + _DIM_LIST_FOR_PROMPT
)

COVERAGE_TOOL = {
    "name": "emit_coverage",
    "description": "Report which ASH dimensions this conversational turn touched.",
    "input_schema": {
        "type": "object",
        "properties": {
            "dimensions": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string",
                                  "enum": ["opened", "explored", "deep"]},
                        "excerpt": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["state"],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["dimensions"],
    },
}

_EMPTY_EXTRACT = {"dimensions": {}, "summary": ""}


def _haiku_extract(memory: dict, user_text: str, ally_text: str = "") -> dict:
    """One Haiku call mapping the latest turn -> touched dimensions. NEVER raises;
    returns the empty default on any failure (it runs fire-and-forget)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return dict(_EMPTY_EXTRACT)

    known = context_block(memory)
    user_message = (
        f"Already known about them:\n{known}\n\n"
        f"Latest exchange:\nPERSON: {user_text}\n"
        f"ALLY: {ally_text}\n\nReport the coverage now."
    )
    payload = {
        "model": HAIKU_MODEL,
        "max_tokens": 1024,
        "system": [{"type": "text", "text": _EXTRACT_SYSTEM,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_message}],
        "tools": [COVERAGE_TOOL],
        "tool_choice": {"type": "tool", "name": "emit_coverage"},
    }
    try:
        resp = requests.post(
            ANTHROPIC_MESSAGES,
            headers={"x-api-key": api_key,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json=payload, timeout=60,
        )
        if not resp.ok:
            return dict(_EMPTY_EXTRACT)
        body = resp.json()
        for b in body.get("content", []):
            if b.get("type") == "tool_use" and b.get("name") == "emit_coverage":
                inp = b.get("input")
                if isinstance(inp, dict):
                    inp.setdefault("dimensions", {})
                    inp.setdefault("summary", "")
                    return inp
        return dict(_EMPTY_EXTRACT)
    except Exception:
        return dict(_EMPTY_EXTRACT)


def update_from_turn(cx, email: str, user_text: str, ally_text: str = "") -> dict:
    """get -> Haiku extract -> pure merge -> persist -> return merged memory.
    Safe to call fire-and-forget after an ally reply."""
    memory = get(cx, email)
    extracted = _haiku_extract(memory, user_text, ally_text)
    merged = merge_turn(memory, extracted)
    _upsert(cx, email, merged.get("summary", ""), merged["dimensions"])
    merged["email"] = _norm_email(email)
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ash_map.py -k "extract or update_from_turn" -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add dashboard/ash_map.py tests/test_ash_map.py
git commit -m "feat(ash_map): _haiku_extract updater + update_from_turn orchestrator"
```

---

### Task 6: Full-suite green + 2-turn end-to-end verification walk

**Files:**
- Test: `tests/test_ash_map.py` (add the end-to-end walk)

**Interfaces:**
- Consumes: all of `dashboard/ash_map.py`.
- Produces: a single `test_two_turn_walk` confirming the spec's verification scenario end-to-end (updater mocked): turn 1 opens `symptoms` + `terrain`; turn 2 deepens `symptoms` and opens `inheritance` → assert states, the symptoms excerpt is the turn-1 wording, notes from both turns present, untouched dims still untouched, summary updated.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ash_map.py
def test_two_turn_walk(monkeypatch):
    cx = sqlite3.connect(":memory:")
    turns = iter([
        {"dimensions": {
            "symptoms": {"state": "opened", "excerpt": "stomach burns after meals",
                         "notes": "post-meal burning"},
            "terrain": {"state": "opened", "excerpt": "", "notes": "run down lately"}},
         "summary": "New, guarded, gut trouble."},
        {"dimensions": {
            "symptoms": {"state": "deep", "excerpt": "this should be ignored",
                         "notes": "waking at night with it"},
            "inheritance": {"state": "opened", "excerpt": "", "notes": "dad had ulcers"}},
         "summary": "Gut issue likely familial; opening up."},
    ])
    monkeypatch.setattr(am, "_haiku_extract",
                        lambda *a, **k: next(turns))

    am.update_from_turn(cx, "walk@x.com", "my stomach burns after I eat", "")
    final = am.update_from_turn(cx, "walk@x.com", "it wakes me at night; dad had ulcers", "")

    d = final["dimensions"]
    assert d["symptoms"]["state"] == "deep"
    assert d["symptoms"]["opened_excerpt"] == "stomach burns after meals"  # turn-1 wording
    assert "post-meal burning" in d["symptoms"]["notes"]
    assert "waking at night with it" in d["symptoms"]["notes"]
    assert d["terrain"]["state"] == "opened"
    assert d["inheritance"]["state"] == "opened"
    assert d["body"]["state"] == "untouched"   # untouched dims stay untouched
    assert final["summary"] == "Gut issue likely familial; opening up."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ash_map.py::test_two_turn_walk -v`
Expected: PASS already (all primitives exist) — if it FAILS, fix the implementation, not the test. This task is the integration gate.

- [ ] **Step 3: Run the full suite**

Run: `python3 -m pytest tests/test_ash_map.py -v`
Expected: PASS (all tests, ~20). No network, no doppler.

- [ ] **Step 4: Confirm pure-module (no app import leak)**

Run: `python3 -c "import dashboard.ash_map as m; print(len(m.DIM_KEYS), m.HAIKU_MODEL)"`
Expected: prints `12 claude-haiku-4-5-20251001` with no import error (proves no app/Flask/network-at-import dependency).

- [ ] **Step 5: Commit**

```bash
git add tests/test_ash_map.py
git commit -m "test(ash_map): two-turn end-to-end coverage walk"
```

---

## Self-Review

**Spec coverage:**
- 12 dimensions constant → Task 1 ✓
- Data model / table → Task 3 ✓
- `_blank_map`, `_norm_email` → Task 1 ✓
- `merge_turn` (forward-only, set-once excerpt, dedup notes, no mutate, summary replace/keep) → Task 2 ✓
- `init_table`, `get` (skeleton + backfill) → Task 3 ✓
- `context_block` (first-conversation + populated sections) → Task 4 ✓
- `_haiku_extract` (forced tool, never raises, empty default) + `COVERAGE_TOOL` schema → Task 5 ✓
- `update_from_turn` orchestrator → Task 5 ✓
- Testing list (merge / get / context_block / _haiku_extract / update_from_turn) → Tasks 2-5 ✓
- Verification 2-turn walk → Task 6 ✓
- Reuse (mirror journal_blueprint Haiku + journal_store storage; module-level `requests`; cx-supplied; no app import) → Global Constraints + Task 5/3 ✓

**Out of scope (correctly omitted):** no chat surface, no caller of `update_from_turn`, no `query_log` backfill, no 12×5 sub-cells. ✓

**Placeholder scan:** none — every code/test step carries full content.

**Type consistency:** `merge_turn(memory, updater_output)`, `get(cx, email)`, `_upsert(cx, email, summary, dimensions)`, `_haiku_extract(memory, user_text, ally_text="")`, `update_from_turn(cx, email, user_text, ally_text="")`, `context_block(memory)`, `COVERAGE_TOOL`/`emit_coverage`, `DIM_KEYS`/`ASH_DIMENSIONS`/`STATE_ORDER` — names consistent across all tasks.
