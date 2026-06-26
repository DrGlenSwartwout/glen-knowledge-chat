# Biofield Balancing B3b-2a — Email-Feedback Mining — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold a client's recent email feedback (`personal_email_feedback`, AI-distilled) into the existing recent-comms mining so it becomes `source='comm'` stresses, via two pure-function extensions only.

**Architecture:** Add a windowed, email-joined `recent_feedback` section to `recent_comms` (prod-deployed via the unchanged `/api/people/recent-comms` endpoint) and include it in `comms_to_text` (local). No new endpoint, route, button, or `add_stress` change — the additive dict key auto-flows through the B3b pipeline.

**Tech Stack:** Python 3.11, sqlite3, json, pytest.

## Global Constraints

- Only two files change: `dashboard/recent_comms.py` and `dashboard/biofield_comms.py`. No endpoint/route/UI/`add_stress` changes.
- `recent_comms` is pure (takes a connection) and offline-testable; `comms_to_text` is pure.
- Mine the **distilled** fields only: `ai_summary` + `extracted_topics` + `extracted_conditions`. **Do NOT select or surface `raw_text`.**
- Window the feedback to the last `days_window` days via `received_at > datetime('now', ?)` with `f"-{int(days_window)} days"` (same int-guard as B3b). Join `users.email → user_id`, case-insensitive.
- Best-effort: a missing `personal_email_feedback`/`users` table, or bad/empty JSON in the list columns, yields an empty section — never raises.
- Back-compat: a context dict WITHOUT `recent_feedback` must make `comms_to_text` behave exactly as today (B3b `comms_to_text` tests stay green).
- The `recent_comms` change deploys to prod (the endpoint imports it) — additive key, read-only, no behavior change to the endpoint or the other three sections.
- Run tests: `cd /tmp/wt-deploy-chat-82bd74c2 && ~/.venvs/deploy-chat311/bin/python -m pytest <path> -v`. B3b tests (`tests/test_recent_comms.py`, `tests/test_biofield_comms.py`) must stay green.

---

### Task 1: `recent_comms` — `recent_feedback` section

**Files:**
- Modify: `dashboard/recent_comms.py`
- Test: `tests/test_recent_comms.py` (add cases)

**Interfaces:**
- Produces: `recent_comms(...)` return dict gains `"recent_feedback": [ {"summary": str, "topics": [str], "conditions": [str], "received_at": str} ]`, windowed + email-joined, JSON-parsed, `raw_text` excluded. Adds module helper `_json_list(s) -> list[str]`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_recent_comms.py`:

```python
import json as _json
from datetime import datetime, timezone, timedelta


def _fb_db(tmp_path):
    import sqlite3
    cx = sqlite3.connect(str(tmp_path / "fb.db"))
    cx.executescript("""
        CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT);
        CREATE TABLE personal_email_feedback(id INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at TEXT, user_id INTEGER, raw_text TEXT, ai_summary TEXT,
            extracted_topics TEXT, extracted_conditions TEXT);
    """)
    cx.execute("INSERT INTO users(id,email) VALUES(1,'j@x.com')")
    cx.execute("INSERT INTO users(id,email) VALUES(2,'other@x.com')")
    return cx


def _ago(days):
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def test_recent_feedback_windowed_joined_parsed(tmp_path):
    cx = _fb_db(tmp_path)
    # in-window row for j@x.com
    cx.execute("INSERT INTO personal_email_feedback(received_at,user_id,raw_text,ai_summary,"
               "extracted_topics,extracted_conditions) VALUES(?,?,?,?,?,?)",
               (_ago(2), 1, "SECRET RAW REPLY", "feels exhausted lately",
                _json.dumps(["sleep", "energy"]), _json.dumps(["adrenal fatigue"])))
    # out-of-window row for j@x.com
    cx.execute("INSERT INTO personal_email_feedback(received_at,user_id,ai_summary) VALUES(?,?,?)",
               (_ago(60), 1, "old feedback"))
    # other user's row
    cx.execute("INSERT INTO personal_email_feedback(received_at,user_id,ai_summary) VALUES(?,?,?)",
               (_ago(1), 2, "not jane's"))
    cx.commit()
    out = recent_comms(cx, "J@X.com", days_window=7)               # case-insensitive join
    fb = out["recent_feedback"]
    assert len(fb) == 1
    assert fb[0]["summary"] == "feels exhausted lately"
    assert fb[0]["topics"] == ["sleep", "energy"]
    assert fb[0]["conditions"] == ["adrenal fatigue"]
    # raw_text must never be exposed
    assert "raw_text" not in fb[0] and "SECRET RAW REPLY" not in str(out)


def test_recent_feedback_bad_json_and_missing_table(tmp_path):
    cx = _fb_db(tmp_path)
    cx.execute("INSERT INTO personal_email_feedback(received_at,user_id,ai_summary,"
               "extracted_topics,extracted_conditions) VALUES(?,?,?,?,?)",
               (_ago(1), 1, "sum", "not json", ""))
    cx.commit()
    out = recent_comms(cx, "j@x.com")
    assert out["recent_feedback"][0]["topics"] == [] and out["recent_feedback"][0]["conditions"] == []
    # missing tables entirely -> empty section, no raise
    import sqlite3
    bare = sqlite3.connect(str(tmp_path / "bare.db"))
    assert recent_comms(bare, "j@x.com")["recent_feedback"] == []


def test_empty_email_includes_feedback_key(tmp_path):
    cx = _fb_db(tmp_path)
    assert recent_comms(cx, "")["recent_feedback"] == []
```

- [ ] **Step 2: Run** → FAIL (`recent_feedback` key absent).

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_recent_comms.py -v`

- [ ] **Step 3: Implement** — in `dashboard/recent_comms.py`:

(a) Add `"recent_feedback": []` to the initial `out` dict:

```python
    out = {"intake_summary": "", "recent_inquiries": [], "recent_queries": [],
           "recent_feedback": []}
```

(b) Add the helper (near `_intake_summary`):

```python
def _json_list(s):
    try:
        v = json.loads(s or "[]")
    except Exception:
        return []
    return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []
```

(c) Add the feedback section just before `return out` (after the queries block):

```python
    try:                                                  # email feedback: windowed, joined
        out["recent_feedback"] = [
            {"summary": r["ai_summary"] or "",
             "topics": _json_list(r["extracted_topics"]),
             "conditions": _json_list(r["extracted_conditions"]),
             "received_at": r["received_at"]}
            for r in cx.execute(
                "SELECT pf.ai_summary, pf.extracted_topics, pf.extracted_conditions, "
                "pf.received_at FROM personal_email_feedback pf "
                "JOIN users u ON u.id = pf.user_id "
                "WHERE lower(u.email)=lower(?) AND pf.received_at > datetime('now', ?) "
                "ORDER BY pf.received_at DESC", (email, win)).fetchall()]
    except Exception:
        pass
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_recent_comms.py -v` → PASS (new + existing B3b recent_comms tests green).

- [ ] **Step 5: Commit**

```bash
git add dashboard/recent_comms.py tests/test_recent_comms.py
git commit -m "feat(biofield-b3b2a): recent_comms email-feedback section (distilled, windowed)"
```

---

### Task 2: `comms_to_text` — include `recent_feedback`

**Files:**
- Modify: `dashboard/biofield_comms.py`
- Test: `tests/test_biofield_comms.py` (add cases)

**Interfaces:**
- Consumes: the `recent_feedback` shape from Task 1.
- Produces: `comms_to_text` appends each feedback item's `summary` + joined `topics`+`conditions` to the blob, after the existing sections. A context without `recent_feedback` is unchanged.

- [ ] **Step 1: Write the failing test** — append to `tests/test_biofield_comms.py`:

```python
def test_includes_recent_feedback():
    ctx = {"intake_summary": "", "recent_inquiries": [], "recent_queries": [],
           "recent_feedback": [{"summary": "feels exhausted",
                                "topics": ["sleep"], "conditions": ["adrenal fatigue"]}]}
    t = comms_to_text(ctx)
    assert "feels exhausted" in t and "sleep" in t and "adrenal fatigue" in t


def test_no_feedback_key_unchanged():
    # context without the key behaves exactly as before (back-compat)
    ctx = {"intake_summary": "Jane", "recent_inquiries": [], "recent_queries": []}
    assert comms_to_text(ctx) == "Jane"
```

- [ ] **Step 2: Run** → FAIL (feedback not in blob).

- [ ] **Step 3: Implement** — in `dashboard/biofield_comms.py`, append to `comms_to_text` after the `recent_queries` loop (before `return`):

```python
    for fb in context.get("recent_feedback") or []:
        s2 = (fb.get("summary") or "").strip()
        if s2:
            parts.append(s2)
        tc = ", ".join([*(fb.get("topics") or []), *(fb.get("conditions") or [])])
        if tc.strip():
            parts.append(tc)
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_comms.py -v` → PASS (new + existing B3b comms tests green).

- [ ] **Step 5: Run B3b + mining tests + commit**

```bash
~/.venvs/deploy-chat311/bin/python -m pytest \
  tests/test_recent_comms.py tests/test_biofield_comms.py \
  tests/test_biofield_mine_comms_routes.py -q
git add dashboard/biofield_comms.py tests/test_biofield_comms.py
git commit -m "feat(biofield-b3b2a): comms_to_text includes email feedback"
```

---

## Self-Review

**Spec coverage:**
- `recent_feedback` section (windowed, email-joined, distilled, raw_text excluded, JSON-parsed best-effort) → Task 1. ✓
- `comms_to_text` includes it, back-compatible → Task 2. ✓
- No endpoint/route/UI/`add_stress` change → confirmed (only 2 files). ✓
- Flows through existing pipeline as `source='comm'` → inherited from B3b (the endpoint returns the additive key, fetch_recent_comms passes it, comms_to_text now reads it, mine-comms adds it). ✓

**Placeholder scan:** No TBDs; complete code in every step.

**Type consistency:** `recent_feedback` items `{summary, topics, conditions, received_at}` (Task 1) match what `comms_to_text` reads (Task 2: `summary`/`topics`/`conditions`). `_json_list` defined + used in Task 1.

## Verification (manual, after both tasks + merge/deploy)

1. Local tests green.
2. After merge to main + Render redeploy: curl the existing endpoint for a client who has email feedback —
   `doppler run -p remedy-match -c prd -- sh -c 'curl -s -H "X-Console-Key: $CONSOLE_SECRET" "https://illtowell.com/api/people/recent-comms?q=<email>"'` →
   confirm a populated `"recent_feedback"` array (and that `raw_text` is absent).
3. In the local tool, "Mine recent comms" for that client now also yields stresses drawn from their email feedback.
