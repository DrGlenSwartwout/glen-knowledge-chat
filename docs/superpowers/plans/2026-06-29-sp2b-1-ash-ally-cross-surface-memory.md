# SP2b-1 — ASH Ally Cross-Surface Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `dashboard/ash_ally.py` (the fail-open, flag-gated cross-surface ASH-memory helper) and wire it into the 4 streaming chat surfaces so the ally "follows" an identified person, reading their `ash_map` memory into the prompt and updating it after each reply.

**Architecture:** A new pure module `dashboard/ash_ally.py` exposes `ally_overlay` (read → framed prompt block) and `record_turn` (write, lock-split so the Haiku extract never holds the DB lock). `ash_map` gains a `persist_extract` seam (the locked tail of `update_from_turn`) so `record_turn` can persist a merge without re-running the LLM. Each of the 4 SSE routes gets the same two touches: prepend `ally_overlay(...)` to its system prompt, and fire `record_turn(...)` in a background daemon thread after the reply. A feature flag `ASH_ALLY_ENABLED` keeps it dark until render-verified.

**Tech Stack:** Python 3, stdlib `sqlite3`/`threading`/`os`, the existing `dashboard/ash_map.py`, Anthropic Haiku (via `ash_map._haiku_extract`). Tests: pure-module pytest (no app import, no network) for the helper + seam.

## Global Constraints

- New module: `dashboard/ash_ally.py`. New test: `tests/test_ash_ally.py`. Modify: `dashboard/ash_map.py`, `tests/test_ash_map.py`, `app.py`.
- `dashboard/ash_ally.py` MUST NOT import `app` or Flask. It imports `ash_map` (same package) + stdlib only. The app's `LOG_DB` path and `_db_lock` are passed IN as arguments, never imported.
- Feature gate: `ENABLED()` reads `os.environ.get("ASH_ALLY_ENABLED")` truthy among `{"1","true","yes"}` case-insensitive, evaluated at CALL time (not import).
- **Fail-open everywhere:** `ally_overlay` and `record_turn` must NEVER raise into a caller — wrap bodies in try/except and degrade (`""` / no-op). A memory hiccup can never break a chat.
- **Lock discipline (required):** `record_turn` must NOT hold the lock across the Haiku extract. Lock only the fast sqlite read and the fast sqlite write; run `_haiku_extract` with no lock held.
- Subject-email rule per surface: `/chat` → `email or _member_email`; `/begin/match/chat` → `email` only if `for_whom != "someone-else"` else `""`; `/begin/concierge/chat` → `email`; `/api/portal/<token>/chat` → `email` (from the portal record). Empty subject → overlay `""`, record no-op.
- Overlay is prepended to the surface's SYSTEM prompt string (an empty overlay is a no-op, so an unconditional prepend-if-nonempty is safe).
- Record dispatch uses the app's existing idiom: `threading.Thread(target=ash_ally.record_turn, args=(...), daemon=True).start()`, wrapped in try/except so a spawn failure can't break the stream.
- Tests for the helper + seam run with plain pytest, no doppler, no network: `python3 -m pytest tests/test_ash_ally.py tests/test_ash_map.py -v`.
- `ash_map` public surface already present: `get(cx, email)`, `merge_turn(memory, updater_output)`, `context_block(memory)`, `_haiku_extract(memory, user_text, ally_text="")`, `_upsert(cx, email, summary, dimensions)`, `update_from_turn(cx, email, user_text, ally_text="")`, `DIM_KEYS`.

---

### Task 1: `ash_map.persist_extract` seam

Refactor the locked tail of `update_from_turn` into a reusable public function, so `record_turn` can persist a merge under a caller-held lock without re-running the LLM. Behavior of `update_from_turn` is unchanged.

**Files:**
- Modify: `dashboard/ash_map.py` (the `update_from_turn` region, ~line 307)
- Test: `tests/test_ash_map.py` (append)

**Interfaces:**
- Consumes: `get`, `merge_turn`, `_upsert`, `_norm_email`, `_haiku_extract` (all already in `ash_map`).
- Produces: `persist_extract(cx, email, extracted) -> dict` — `get(cx, email)` → `merge_turn(memory, extracted)` → `_upsert(cx, email, merged["summary"], merged["dimensions"])` → return the merged memory with `email` set. `update_from_turn` now = `get` → `_haiku_extract` → `persist_extract`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ash_map.py
def test_persist_extract_merges_and_persists():
    cx = sqlite3.connect(":memory:")
    extracted = {"dimensions": {"symptoms": {"state": "opened",
                 "excerpt": "knee aches", "notes": "AM knee"}}, "summary": "In pain."}
    merged = am.persist_extract(cx, "p@x.com", extracted)
    assert merged["dimensions"]["symptoms"]["state"] == "opened"
    assert merged["summary"] == "In pain."
    assert merged["email"] == "p@x.com"
    # persisted: a fresh get sees it
    assert am.get(cx, "p@x.com")["dimensions"]["symptoms"]["state"] == "opened"


def test_persist_extract_equals_manual_merge_then_persist():
    cx = sqlite3.connect(":memory:")
    extracted = {"dimensions": {"terrain": {"state": "explored", "excerpt": "",
                 "notes": "low energy"}}, "summary": "Tired."}
    expected = am.merge_turn(am.get(cx, "q@x.com"), extracted)
    got = am.persist_extract(cx, "q@x.com", extracted)
    assert got["dimensions"] == expected["dimensions"]
    assert got["summary"] == expected["summary"]


def test_update_from_turn_still_works_via_seam(monkeypatch):
    cx = sqlite3.connect(":memory:")
    monkeypatch.setattr(am, "_haiku_extract", lambda *a, **k: {
        "dimensions": {"mind": {"state": "opened", "excerpt": "racing thoughts",
                       "notes": "anxious"}}, "summary": "Anxious."})
    out = am.update_from_turn(cx, "r@x.com", "my mind races", "")
    assert out["dimensions"]["mind"]["state"] == "opened"
    assert out["summary"] == "Anxious."
    assert am.get(cx, "r@x.com")["dimensions"]["mind"]["state"] == "opened"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ash_map.py -k "persist_extract" -v`
Expected: FAIL — `AttributeError: module 'dashboard.ash_map' has no attribute 'persist_extract'`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `update_from_turn` (currently `get` → `_haiku_extract` → `merge_turn` → `_upsert` → return) so the locked tail is its own function. The current `update_from_turn` looks like:

```python
def update_from_turn(cx, email: str, user_text: str, ally_text: str = "") -> dict:
    """<existing docstring>"""
    memory = get(cx, email)
    extracted = _haiku_extract(memory, user_text, ally_text)
    merged = merge_turn(memory, extracted)
    _upsert(cx, email, merged.get("summary", ""), merged["dimensions"])
    merged["email"] = _norm_email(email)
    return merged
```

Refactor to:

```python
def persist_extract(cx, email: str, extracted: dict) -> dict:
    """Apply an already-extracted updater result to a person's memory and persist it.
    This is the locked tail of update_from_turn: get -> merge -> upsert. The caller
    holds the DB lock around this (it does fast sqlite I/O only, no LLM call), so a
    cross-surface writer can run the slow _haiku_extract OUTSIDE the lock and pass the
    result here. Returns the merged memory with email set."""
    memory = get(cx, email)
    merged = merge_turn(memory, extracted)
    _upsert(cx, email, merged.get("summary", ""), merged["dimensions"])
    merged["email"] = _norm_email(email)
    return merged


def update_from_turn(cx, email: str, user_text: str, ally_text: str = "") -> dict:
    """<keep the existing docstring verbatim>"""
    memory = get(cx, email)
    extracted = _haiku_extract(memory, user_text, ally_text)
    return persist_extract(cx, email, extracted)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ash_map.py -v`
Expected: PASS (all prior SP2a tests + the 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add dashboard/ash_map.py tests/test_ash_map.py
git commit -m "feat(ash_map): persist_extract seam (locked tail of update_from_turn)"
```

---

### Task 2: `ash_ally` module skeleton — `ENABLED` + `ally_overlay`

**Files:**
- Create: `dashboard/ash_ally.py`
- Test: `tests/test_ash_ally.py`

**Interfaces:**
- Consumes: `ash_map.get`, `ash_map.context_block`, `ash_map.DIM_KEYS`.
- Produces:
  - `ENABLED() -> bool` — `os.environ.get("ASH_ALLY_ENABLED","")` lowercased in `{"1","true","yes"}`.
  - `_is_blank(memory) -> bool` — True when `summary` empty AND every dimension `state == "untouched"`.
  - `FRAME_HEADER: str` / `FRAME_FOOTER: str` — the framing text below.
  - `ally_overlay(db_path, subject_email) -> str` — `""` when `not ENABLED()`, empty email, blank memory, or on ANY exception (fail-open). Otherwise opens a read connection on `db_path`, gets the memory, and returns `FRAME_HEADER + context_block(memory) + FRAME_FOOTER`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ash_ally.py
import sqlite3
import dashboard.ash_ally as aa
import dashboard.ash_map as am


def _seed(db_path, email, summary, dim, cell):
    cx = sqlite3.connect(db_path)
    m = am.get(cx, email)
    m["dimensions"][dim].update(cell)
    am._upsert(cx, email, summary, m["dimensions"])
    cx.close()


def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("ASH_ALLY_ENABLED", raising=False)
    assert aa.ENABLED() is False
    monkeypatch.setenv("ASH_ALLY_ENABLED", "TRUE")
    assert aa.ENABLED() is True
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    assert aa.ENABLED() is True
    monkeypatch.setenv("ASH_ALLY_ENABLED", "off")
    assert aa.ENABLED() is False


def test_overlay_empty_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("ASH_ALLY_ENABLED", raising=False)
    db = str(tmp_path / "t.db")
    _seed(db, "a@b.com", "A summary.", "symptoms", {"state": "explored", "notes": "AM knee"})
    assert aa.ally_overlay(db, "a@b.com") == ""


def test_overlay_empty_for_no_email_or_blank_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    db = str(tmp_path / "t.db")
    assert aa.ally_overlay(db, "") == ""            # no subject
    assert aa.ally_overlay(db, "never@seen.com") == ""  # unseen → blank memory


def test_overlay_returns_framed_context_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    db = str(tmp_path / "t.db")
    _seed(db, "a@b.com", "A tired caregiver.", "symptoms",
          {"state": "explored", "notes": "AM knee pain"})
    ov = aa.ally_overlay(db, "a@b.com")
    assert "WHAT YOU ALREADY KNOW ABOUT THIS PERSON" in ov   # frame header
    assert "A tired caregiver." in ov                        # the summary, via context_block
    assert "Never read this back as a list" in ov            # frame footer guidance


def test_overlay_fail_open_on_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    monkeypatch.setattr(am, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    # must swallow and return "" — never raise
    assert aa.ally_overlay(str(tmp_path / "t.db"), "a@b.com") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ash_ally.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.ash_ally'`

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/ash_ally.py
"""Cross-surface ASH ally memory (SP2b-1).

A thin, fail-open, flag-gated layer over dashboard/ash_map.py that lets the ally
"follow" an identified person across chat surfaces: ally_overlay() reads their
coverage map into a system-prompt block; record_turn() updates it after a reply.
Pure module: no app/Flask import. The app's LOG_DB path and _db_lock are passed in
as arguments (never imported), so this stays unit-testable and avoids a circular
import. Every public function degrades silently on error — a memory failure must
never break a chat.
"""
import os
import sqlite3

from dashboard import ash_map

_TRUE = {"1", "true", "yes"}

FRAME_HEADER = "━━━ WHAT YOU ALREADY KNOW ABOUT THIS PERSON ━━━\n"
FRAME_FOOTER = (
    "\n\nGreet them with continuity and pick up the threads they've opened. Don't re-ask "
    "what they've already shared. Never read this back as a list, never mention "
    "\"dimensions\", a \"map\", or that you track anything — just let it make you feel like "
    "someone who remembers them."
)


def ENABLED() -> bool:
    return os.environ.get("ASH_ALLY_ENABLED", "").strip().lower() in _TRUE


def _is_blank(memory: dict) -> bool:
    if (memory.get("summary") or "").strip():
        return False
    dims = memory.get("dimensions") or {}
    return all((c or {}).get("state", "untouched") == "untouched" for c in dims.values())


def ally_overlay(db_path, subject_email: str) -> str:
    """Framed memory block for a surface's system prompt, or '' (disabled / no email /
    nothing learned yet / any error). Fail-open: never raises."""
    try:
        if not ENABLED() or not (subject_email or "").strip():
            return ""
        with sqlite3.connect(db_path) as cx:
            memory = ash_map.get(cx, subject_email)
        if _is_blank(memory):
            return ""
        return FRAME_HEADER + ash_map.context_block(memory) + FRAME_FOOTER
    except Exception:
        return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ash_ally.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/ash_ally.py tests/test_ash_ally.py
git commit -m "feat(ash_ally): ENABLED flag + fail-open ally_overlay"
```

---

### Task 3: `ash_ally.record_turn` — the lock-split writer

**Files:**
- Modify: `dashboard/ash_ally.py`
- Test: `tests/test_ash_ally.py` (append)

**Interfaces:**
- Consumes: `ash_map.get`, `ash_map._haiku_extract`, `ash_map.persist_extract` (Task 1), `ENABLED`.
- Produces: `record_turn(db_path, lock, subject_email, user_text, ally_text="") -> None` — no-op when `not ENABLED()` or empty email. Fail-open (never raises). **Lock-split:** (1) under `lock`, open a connection and `get` the current memory; (2) with NO lock held, run `_haiku_extract`; (3) under `lock`, open a connection and `persist_extract`. Returns `None`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ash_ally.py
import threading


def test_record_turn_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("ASH_ALLY_ENABLED", raising=False)
    db = str(tmp_path / "t.db")
    called = {"extract": 0}
    monkeypatch.setattr(am, "_haiku_extract", lambda *a, **k: called.__setitem__("extract", called["extract"] + 1) or {"dimensions": {}, "summary": ""})
    assert aa.record_turn(db, threading.Lock(), "a@b.com", "hi", "") is None
    assert called["extract"] == 0   # never even reached the LLM


def test_record_turn_noop_for_empty_email(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    db = str(tmp_path / "t.db")
    assert aa.record_turn(db, threading.Lock(), "", "hi", "") is None


def test_record_turn_persists_and_accumulates(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    db = str(tmp_path / "t.db")
    seq = iter([
        {"dimensions": {"symptoms": {"state": "opened", "excerpt": "knee aches", "notes": "AM"}},
         "summary": "One."},
        {"dimensions": {"symptoms": {"state": "deep", "excerpt": "ignored", "notes": "night pain"}},
         "summary": "Two."},
    ])
    monkeypatch.setattr(am, "_haiku_extract", lambda *a, **k: next(seq))
    aa.record_turn(db, threading.Lock(), "u@x.com", "my knee aches", "")
    cx = sqlite3.connect(db)
    assert am.get(cx, "u@x.com")["dimensions"]["symptoms"]["state"] == "opened"
    aa.record_turn(db, threading.Lock(), "u@x.com", "worse at night", "")
    m = am.get(cx, "u@x.com")
    assert m["dimensions"]["symptoms"]["state"] == "deep"          # deepened
    assert m["dimensions"]["symptoms"]["opened_excerpt"] == "knee aches"  # set-once preserved
    assert "AM" in m["dimensions"]["symptoms"]["notes"] and "night pain" in m["dimensions"]["symptoms"]["notes"]
    assert m["summary"] == "Two."
    cx.close()


def test_record_turn_fail_open_on_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    monkeypatch.setattr(am, "_haiku_extract", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    # extract raising must be swallowed — record_turn returns None, no exception
    assert aa.record_turn(str(tmp_path / "t.db"), threading.Lock(), "u@x.com", "hi", "") is None


def test_record_turn_does_not_hold_lock_during_extract(tmp_path, monkeypatch):
    monkeypatch.setenv("ASH_ALLY_ENABLED", "1")
    db = str(tmp_path / "t.db")
    lock = threading.Lock()
    observed = {}

    def fake_extract(memory, user_text, ally_text=""):
        # The lock MUST be free while the (slow) extract runs.
        got = lock.acquire(blocking=False)
        observed["lock_free_during_extract"] = got
        if got:
            lock.release()
        return {"dimensions": {"mind": {"state": "opened", "excerpt": "x", "notes": "y"}}, "summary": "s"}

    monkeypatch.setattr(am, "_haiku_extract", fake_extract)
    aa.record_turn(db, lock, "u@x.com", "hi", "")
    assert observed["lock_free_during_extract"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ash_ally.py -k record_turn -v`
Expected: FAIL — `AttributeError: module 'dashboard.ash_ally' has no attribute 'record_turn'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/ash_ally.py
def record_turn(db_path, lock, subject_email: str, user_text: str, ally_text: str = "") -> None:
    """Update a person's ASH memory from one turn. Designed to be dispatched off the
    request path (a daemon thread). No-op when disabled / no email. Fail-open: never
    raises. Lock-split: the slow Haiku extract runs with NO lock held; only the fast
    sqlite read and write hold `lock`."""
    try:
        if not ENABLED() or not (subject_email or "").strip():
            return
        # (1) locked read of current memory (for extract context)
        with lock:
            with sqlite3.connect(db_path) as cx:
                memory = ash_map.get(cx, subject_email)
        # (2) UNLOCKED slow LLM call
        extracted = ash_map._haiku_extract(memory, user_text, ally_text)
        # (3) locked merge + persist (re-reads under the lock so concurrent same-email
        #     turns converge; merge_turn is forward-only)
        with lock:
            with sqlite3.connect(db_path) as cx:
                ash_map.persist_extract(cx, subject_email, extracted)
    except Exception:
        return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ash_ally.py -v`
Expected: PASS (all Task 2 + 5 new record_turn tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/ash_ally.py tests/test_ash_ally.py
git commit -m "feat(ash_ally): lock-split record_turn (extract runs unlocked)"
```

---

### Task 4: Wire `/chat`

Thin, uniform 2-touch edit. Logic is fully tested in Tasks 1-3; this just connects it. Behavioral proof is the go-live render-verify (Task 8) — route-level SSE unit tests are not practical in this app.

**Files:**
- Modify: `app.py` (overlay after line 3355; record after the `log_query` at ~3479; add `from dashboard import ash_ally` import near the other dashboard imports if not present)

**Interfaces:**
- Consumes: `ash_ally.ally_overlay(LOG_DB, subject_email)`, `ash_ally.record_turn(LOG_DB, _db_lock, subject_email, query, answer)`. Subject = `email or _member_email`.

- [ ] **Step 1: Confirm `ash_ally` is importable in app context**

Run: `grep -n "from dashboard import ash_ally\|import ash_ally" app.py`
If absent, add `from dashboard import ash_ally` alongside the other top-level `from dashboard import ...` lines. (Many dashboard modules are already imported at module top; follow that pattern.)

- [ ] **Step 2: Add the overlay touch**

After the member-overlay block ends (line 3358, the `# ── end member overlay ──` comment) and before `_system = get_system_prompt(level)` (line 3430), the `_system` is assembled at 3430. Change the assembly to prepend the ally overlay. Replace:

```python
        _system = get_system_prompt(level)
```

with:

```python
        _system = get_system_prompt(level)
        _ally_ov = ash_ally.ally_overlay(LOG_DB, email or _member_email)
        if _ally_ov:
            _system = _ally_ov + "\n\n" + _system
```

- [ ] **Step 3: Add the record touch**

Immediately after the `log_query(...)` call returns `log_id` (after line 3479, the closing `)` of the `log_query(...)` call), add the background record dispatch:

```python
        try:
            import threading as _t
            _t.Thread(target=ash_ally.record_turn,
                      args=(LOG_DB, _db_lock, (email or _member_email), query, _clean),
                      daemon=True).start()
        except Exception:
            pass
```

(`_clean` is the directive-stripped final answer; `email or _member_email` is the subject.)

- [ ] **Step 4: Verify no syntax break**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
Expected: prints `ok`.

Run: `grep -n "ash_ally.ally_overlay\|ash_ally.record_turn" app.py`
Expected: both lines present in the `/chat` region (~3431 and ~3480).

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(chat): wire ASH ally overlay + record into /chat"
```

---

### Task 5: Wire `/begin/match/chat`

Same two touches; subject is guarded by `for_whom`.

**Files:**
- Modify: `app.py` (overlay after `_match_system` assembly ~line 3802; record after the `log_query` at ~3827)

**Interfaces:**
- Consumes: `ash_ally.ally_overlay`, `ash_ally.record_turn`. Subject = `email if for_whom != "someone-else" else ""`.

- [ ] **Step 1: Add the overlay touch**

The system prompt is `_match_system` (assembled at line 3802). Replace:

```python
        _match_system = _REMEDY_MATCH_SYSTEM
        if not _member and _is_gated_question(query):
            _match_system = _REMEDY_MATCH_SYSTEM + _EDUCATE_ONLY_POLICY
            yield sse({"gate": True})
```

with:

```python
        _match_system = _REMEDY_MATCH_SYSTEM
        if not _member and _is_gated_question(query):
            _match_system = _REMEDY_MATCH_SYSTEM + _EDUCATE_ONLY_POLICY
            yield sse({"gate": True})
        _ally_subject = email if for_whom != "someone-else" else ""
        _ally_ov = ash_ally.ally_overlay(LOG_DB, _ally_subject)
        if _ally_ov:
            _match_system = _ally_ov + "\n\n" + _match_system
```

- [ ] **Step 2: Add the record touch**

After the `log_query(...)` call (the `except Exception: pass` block ending ~line 3829), add:

```python
        try:
            import threading as _t
            _t.Thread(target=ash_ally.record_turn,
                      args=(LOG_DB, _db_lock,
                            (email if for_whom != "someone-else" else ""), query, answer),
                      daemon=True).start()
        except Exception:
            pass
```

(`answer` is `_clean` here; subject re-applies the `for_whom` guard.)

- [ ] **Step 3: Verify no syntax break**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
Expected: `ok`.

Run: `grep -n "_ally_subject\|for_whom != \"someone-else\"" app.py`
Expected: the guard appears in BOTH the overlay and record touches in the match region.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(match): wire ASH ally into /begin/match/chat (for-whom guarded)"
```

---

### Task 6: Wire `/begin/concierge/chat`

System prompt is the module constant `_CONCIERGE_SYSTEM`, passed directly to `.stream(...)` at line 7184. Build a per-request prompt var that prepends the overlay.

**Files:**
- Modify: `app.py` (overlay computed before `generate()` ~after line 7163; used at the `.stream(...)` call ~7184; record after `answer` is assembled ~7189)

**Interfaces:**
- Consumes: `ash_ally.ally_overlay`, `ash_ally.record_turn`. Subject = `email`.

- [ ] **Step 1: Add the overlay touch (compute per-request system var)**

After the RAG `context_str` block (ends ~line 7164, the `except Exception as e: print(...)` for retrieval) and before `generate()` is defined, add:

```python
    _ally_ov = ash_ally.ally_overlay(LOG_DB, email)
    _sys_concierge = (_ally_ov + "\n\n" + _CONCIERGE_SYSTEM) if _ally_ov else _CONCIERGE_SYSTEM
```

Then change the stream call at line 7183-7184 from:

```python
            with _cl.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=700,
                                     system=_CONCIERGE_SYSTEM, messages=messages) as stream:
```

to:

```python
            with _cl.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=700,
                                     system=_sys_concierge, messages=messages) as stream:
```

- [ ] **Step 2: Add the record touch**

After `answer = "".join(full)` (line 7189), add:

```python
        try:
            import threading as _t
            _t.Thread(target=ash_ally.record_turn,
                      args=(LOG_DB, _db_lock, email, query, answer),
                      daemon=True).start()
        except Exception:
            pass
```

- [ ] **Step 3: Verify no syntax break**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
Expected: `ok`.

Run: `grep -n "_sys_concierge" app.py`
Expected: defined once and used in the `.stream(...)` system arg (2 occurrences).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(concierge): wire ASH ally into /begin/concierge/chat"
```

---

### Task 7: Wire `/api/portal/<token>/chat`

System prompt is `_sys` (built by `portal_concierge.system_prompt(ctx)` at line 12115). Prepend the overlay; subject = the portal record's email.

**Files:**
- Modify: `app.py` (overlay after line 12115; record after `answer` is assembled ~12142)

**Interfaces:**
- Consumes: `ash_ally.ally_overlay`, `ash_ally.record_turn`. Subject = `email` (line 12104, from `portal.get("email")`).

- [ ] **Step 1: Add the overlay touch**

After `_sys = _pcz.system_prompt(ctx)` (line 12115), add:

```python
    _ally_ov = ash_ally.ally_overlay(LOG_DB, email)
    if _ally_ov:
        _sys = _ally_ov + "\n\n" + _sys
```

(`_sys` is already used as the `system=` arg at the `.stream(...)` call on line 12137 — no change needed there.)

- [ ] **Step 2: Add the record touch**

After `answer = "".join(full)` (line 12142), add:

```python
        try:
            import threading as _t
            _t.Thread(target=ash_ally.record_turn,
                      args=(LOG_DB, _db_lock, email, query, answer),
                      daemon=True).start()
        except Exception:
            pass
```

- [ ] **Step 3: Verify no syntax break**

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"`
Expected: `ok`.

Run: `grep -n "ash_ally" app.py | wc -l`
Expected: at least 8 (import + 2 touches × 4 surfaces ≈ 9).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(portal): wire ASH ally into /api/portal/<token>/chat"
```

---

### Task 8: Full-suite green + go-live verification doc

No new production code. Confirm the helper/seam suite is green, the app still parses, and write the go-live render-verify checklist into the spec's verification section so Glen can enable + verify per surface.

**Files:**
- Test: run-only (no new test code)
- Modify: `docs/superpowers/specs/2026-06-29-sp2b-1-ash-ally-cross-surface-memory.md` (append a "Go-live checklist" subsection if not already precise)

- [ ] **Step 1: Run the helper + seam suite**

Run: `python3 -m pytest tests/test_ash_ally.py tests/test_ash_map.py -v`
Expected: ALL passing (SP2a's tests + Task 1's 3 + Tasks 2-3's helper tests). Report the exact count.

- [ ] **Step 2: Confirm pure-module + app parse**

Run: `python3 -c "import dashboard.ash_ally as a; print(a.ENABLED())"`
Expected: prints `False` (flag off by default), no import error (proves no app/Flask/network-at-import dependency).

Run: `python3 -c "import ast; ast.parse(open('app.py').read()); print('app ok')"`
Expected: `app ok`.

- [ ] **Step 3: Confirm all 4 surfaces wired**

Run: `grep -n "ash_ally.ally_overlay\|ash_ally.record_turn" app.py`
Expected: 8 lines — an overlay + a record in each of the 4 SSE regions (~3431/3480, ~3805/3830, ~7165/7190, ~12116/12143).

- [ ] **Step 4: Write the go-live checklist**

Append to the spec's Verification section (if not already there) the exact go-live procedure:
1. Merge with the flag dark (`ASH_ALLY_ENABLED` unset). The layer is inert on prod (overlay `""`, record no-ops) — nothing changes for any user.
2. Set `ASH_ALLY_ENABLED=1` in Render. Redeploy/restart.
3. Render-verify each surface in a headless browser as an identified user who has prior `ash_map` memory: `/chat`, `/begin/match/chat` (with `for_whom=me`), `/begin/concierge/chat` (as a gated member), a `/portal/<token>` page. For each: confirm the reply reflects continuity (no re-asking known facts), assert zero console errors, send a follow-up turn, then confirm via `ash_map.get` (or a DB read) that the stored map advanced.
4. Confirm anonymous use of each surface is unchanged (no overlay, no errors, no extra Haiku call).
5. If any surface misbehaves, unset the flag (instant rollback — no code change).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-06-29-sp2b-1-ash-ally-cross-surface-memory.md
git commit -m "docs(sp2b-1): go-live render-verify checklist"
```

---

## Self-Review

**Spec coverage:**
- Helper `ash_ally.py` (`ENABLED`, `ally_overlay`, `record_turn`, fail-open, flag-gated) → Tasks 2-3 ✓
- Lock discipline (extract unlocked; sqlite ops locked) → Task 3 + its dedicated lock-free-during-extract test ✓
- `ash_map.persist_extract` seam → Task 1 ✓
- Overlay framing block (use naturally, don't recite, no "dimensions") → Task 2 `FRAME_HEADER`/`FRAME_FOOTER` ✓
- 4 SSE surfaces, two uniform touches, subject-email rules (incl. `for_whom` guard, `email or _member_email`) → Tasks 4-7 ✓
- Background daemon-thread dispatch, try/except-wrapped → Tasks 4-7 ✓
- `ASH_ALLY_ENABLED` dark + per-surface render-verify → Task 8 ✓
- Anonymous traffic untouched / no extra cost → enforced by empty-subject no-op (Tasks 2-3), verified Task 8 ✓
- Out of scope (scoped_reply surfaces, practitioner search, scan-analysis, Glendalf voice/video) → not in any task ✓

**Placeholder scan:** none — every code/test step carries full content. Task 8 is run-only + a documented checklist (acceptable: it's the go-live procedure, not deferred code).

**Type consistency:** `ally_overlay(db_path, subject_email)`, `record_turn(db_path, lock, subject_email, user_text, ally_text="")`, `ash_map.persist_extract(cx, email, extracted)`, `ENABLED()`, `_is_blank(memory)`, `FRAME_HEADER`/`FRAME_FOOTER` — consistent across tasks. App touches consistently call `ash_ally.ally_overlay(LOG_DB, <subject>)` and `ash_ally.record_turn(LOG_DB, _db_lock, <subject>, query, <answer>)`.

**Note on wiring-task testing:** Tasks 4-7 edit SSE routes in a 27k-line app with no existing route-level test harness and network-at-import; their gate is `ast.parse` + grep-assert of the two touches, with behavioral correctness proven by the Task 8 go-live render-verify. The substantive logic they invoke is fully unit-tested in Tasks 1-3. This is a deliberate, called-out testing boundary, not an omission.
