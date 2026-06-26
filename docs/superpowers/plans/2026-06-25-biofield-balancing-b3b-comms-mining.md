# Biofield Balancing B3b — Recent-Communication Mining — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mine a client's recent communications (last-7-day chat + inquiries + latest ScoreApp intake) into intake stresses, via a read-only prod endpoint the local tool calls.

**Architecture:** A pure, connection-based `recent_comms(cx, email, days_window=7)` (offline-testable); a thin console-gated `GET /api/people/recent-comms` in app.py that calls it (deploys to prod); a local `comms_to_text` + `fetch_recent_comms` + `_mine_comms` → `interpret_stresses` → `add_stress(source='comm')`, merged, with a button + run-once hook.

**Tech Stack:** Python 3.11, Flask, sqlite3, urllib, pytest.

## Global Constraints

- B3b CHANGES PROD: `app.py` gets one read-only, console-gated GET endpoint that merges to main and auto-deploys to Render. Additive, read-only — no existing prod behavior changes. The local tool changes stay local.
- `recent_comms` is pure (takes a connection) so it is offline-testable; the app.py route wrapper is NOT offline-testable (importing `app` needs prod creds) — its logic is covered by `recent_comms` tests + a syntax check + post-deploy manual verification.
- Window: chat (`query_log`) + `inquiries` filtered to the last `days_window` days; the latest ScoreApp intake is included regardless of age.
- Comm-derived stresses: `source='comm'`, `balance='required'`, merged by normalized label across all sources (reuses B3a `add_stress`).
- Best-effort everywhere: a prod/network/parse failure returns `{}`/an error dict and never blocks intake. `fetch_recent_comms` makes no network call when `CONSOLE_SECRET` is unset.
- Run tests: `cd /tmp/wt-deploy-chat-82bd74c2 && ~/.venvs/deploy-chat311/bin/python -m pytest <path> -v` (pure-module; no Doppler). B1–B4/B3a biofield tests must stay green.

---

### Task 1: `recent_comms` — windowed aggregation (pure, connection-based)

**Files:**
- Create: `dashboard/recent_comms.py`
- Test: `tests/test_recent_comms.py`

**Interfaces:**
- Produces: `recent_comms(cx, email, *, days_window=7, query_log_n=20) -> {"intake_summary": str, "recent_inquiries": [{"main_challenge","main_goal","created_at"}], "recent_queries": [{"question","ts"}]}`. Takes an open sqlite connection. Intake = latest `inbound_leads` (age-agnostic); inquiries + queries filtered to the last `days_window` days. Best-effort per section (missing table → empty, no raise).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recent_comms.py
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from dashboard.recent_comms import recent_comms


def _db(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    cx.executescript("""
        CREATE TABLE inbound_leads(id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT,
            source TEXT, first_name TEXT, raw_json TEXT);
        CREATE TABLE inquiries(id INTEGER PRIMARY KEY AUTOINCREMENT, client_email TEXT,
            main_challenge TEXT, main_goal TEXT, created_at TEXT);
        CREATE TABLE query_log(id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT,
            query TEXT, ts TEXT);
    """)
    return cx


def _iso(days_ago):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def test_windows_chat_and_inquiries_keeps_latest_intake(tmp_path):
    cx = _db(tmp_path)
    # intake (old, but always kept)
    quiz = {"data": {"total_score": {"percent": 80},
                     "quiz_questions": [{"question": "Energy?", "answers": [{"answer": "Low"}]}]}}
    cx.execute("INSERT INTO inbound_leads(email,source,first_name,raw_json) VALUES(?,?,?,?)",
               ("j@x.com", "scoreapp", "Jane", json.dumps(quiz)))
    # inquiries: one recent, one old
    cx.execute("INSERT INTO inquiries(client_email,main_challenge,main_goal,created_at) VALUES(?,?,?,?)",
               ("j@x.com", "fatigue", "more energy", _iso(2)))
    cx.execute("INSERT INTO inquiries(client_email,main_challenge,main_goal,created_at) VALUES(?,?,?,?)",
               ("j@x.com", "ancient", "old", _iso(60)))
    # queries: one recent, one old
    cx.execute("INSERT INTO query_log(email,query,ts) VALUES(?,?,?)", ("j@x.com", "why tired", _iso(1)))
    cx.execute("INSERT INTO query_log(email,query,ts) VALUES(?,?,?)", ("j@x.com", "stale q", _iso(30)))
    cx.commit()
    out = recent_comms(cx, "j@x.com", days_window=7)
    assert "Energy?: Low" in out["intake_summary"] and "Jane" in out["intake_summary"]
    assert [i["main_challenge"] for i in out["recent_inquiries"]] == ["fatigue"]   # old excluded
    assert [q["question"] for q in out["recent_queries"]] == ["why tired"]          # old excluded


def test_question_column_fallback(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    cx.executescript("""CREATE TABLE query_log(id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT, question TEXT, ts TEXT);""")
    cx.execute("INSERT INTO query_log(email,question,ts) VALUES(?,?,?)",
               ("j@x.com", "from question col", _iso(1)))
    cx.commit()
    out = recent_comms(cx, "j@x.com")
    assert [q["question"] for q in out["recent_queries"]] == ["from question col"]


def test_empty_email_and_missing_tables(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))   # no tables at all
    assert recent_comms(cx, "") == {"intake_summary": "", "recent_inquiries": [], "recent_queries": []}
    assert recent_comms(cx, "j@x.com") == {"intake_summary": "", "recent_inquiries": [], "recent_queries": []}
```

- [ ] **Step 2: Run** → FAIL (module missing).

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_recent_comms.py -v`

- [ ] **Step 3: Implement**

```python
# dashboard/recent_comms.py
"""Windowed recent-communications aggregation for the Biofield Intake balancing loop
(B3b). Pure: takes an open sqlite connection so it is testable offline. Mirrors the
queries in app.py:_member_context_for_email but adds a time window and takes a cx."""
import json
import sqlite3


def _intake_summary(first_name, raw_json_str):
    parts = []
    if first_name:
        parts.append(f"first name: {first_name}")
    try:
        payload = json.loads(raw_json_str or "{}")
        data = payload.get("data", payload) or {}
        score = (data.get("total_score") or {}).get("percent") or data.get("score")
        if score:
            parts.append(f"assessment score: {score}%")
        for q in (data.get("quiz_questions") or [])[:6]:
            qt = (q.get("question") or "").strip()
            ans = ", ".join((a.get("answer") or "").strip()
                            for a in (q.get("answers") or []) if a.get("answer"))
            if qt and ans:
                parts.append(f"  {qt}: {ans}")
    except Exception:
        pass
    return "\n".join(parts)


def recent_comms(cx, email, *, days_window=7, query_log_n=20):
    out = {"intake_summary": "", "recent_inquiries": [], "recent_queries": []}
    email = (email or "").strip()
    if not email:
        return out
    cx.row_factory = sqlite3.Row
    win = f"-{int(days_window)} days"                      # int() guards against injection
    try:                                                  # intake: latest, age-agnostic
        row = cx.execute(
            "SELECT first_name, raw_json FROM inbound_leads WHERE email=? "
            "AND source IN ('scoreapp','practice-better','concierge') "
            "ORDER BY id DESC LIMIT 1", (email,)).fetchone()
        if row:
            out["intake_summary"] = _intake_summary(row["first_name"], row["raw_json"])
    except Exception:
        pass
    try:                                                  # inquiries: windowed
        out["recent_inquiries"] = [
            {"main_challenge": r["main_challenge"], "main_goal": r["main_goal"],
             "created_at": r["created_at"]}
            for r in cx.execute(
                "SELECT main_challenge, main_goal, created_at FROM inquiries "
                "WHERE client_email=? AND created_at > datetime('now', ?) "
                "ORDER BY created_at DESC", (email, win)).fetchall()]
    except Exception:
        pass
    try:                                                  # queries: windowed, col fallback
        try:
            rows = cx.execute(
                "SELECT question, ts FROM query_log WHERE email=? AND ts > datetime('now', ?) "
                "ORDER BY id DESC LIMIT ?", (email, win, int(query_log_n))).fetchall()
            out["recent_queries"] = [{"question": r["question"], "ts": r["ts"]} for r in rows]
        except Exception:
            rows = cx.execute(
                "SELECT query, ts FROM query_log WHERE email=? AND ts > datetime('now', ?) "
                "ORDER BY id DESC LIMIT ?", (email, win, int(query_log_n))).fetchall()
            out["recent_queries"] = [{"question": r["query"], "ts": r["ts"]} for r in rows]
    except Exception:
        pass
    return out
```

- [ ] **Step 4: Run** → PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/recent_comms.py tests/test_recent_comms.py
git commit -m "feat(biofield-b3b): recent_comms windowed aggregation (pure, cx-based)"
```

---

### Task 2: `comms_to_text` — flatten for the extractor

**Files:**
- Create: `dashboard/biofield_comms.py`
- Test: `tests/test_biofield_comms.py`

**Interfaces:**
- Produces: `comms_to_text(context) -> str` — flattens a `recent_comms` dict (intake_summary + each inquiry's challenge/goal + each query question) into one newline-joined blob. Empty/None → "".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_comms.py
from dashboard.biofield_comms import comms_to_text


def test_flattens_all_sections():
    ctx = {"intake_summary": "first name: Jane\n  Energy?: Low",
           "recent_inquiries": [{"main_challenge": "fatigue", "main_goal": "more energy"}],
           "recent_queries": [{"question": "why tired"}, {"question": "what helps sleep"}]}
    t = comms_to_text(ctx)
    assert "Jane" in t and "fatigue" in t and "more energy" in t
    assert "why tired" in t and "what helps sleep" in t


def test_empty_context():
    assert comms_to_text({}) == ""
    assert comms_to_text(None) == ""
```

- [ ] **Step 2: Run** → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# dashboard/biofield_comms.py
"""Flatten a recent_comms dict into a single text blob for the stress extractor (B3b)."""


def comms_to_text(context):
    context = context or {}
    parts = []
    s = (context.get("intake_summary") or "").strip()
    if s:
        parts.append(s)
    for inq in context.get("recent_inquiries") or []:
        ch = (inq.get("main_challenge") or "").strip()
        g = (inq.get("main_goal") or "").strip()
        if ch:
            parts.append("challenge: " + ch)
        if g:
            parts.append("goal: " + g)
    for q in context.get("recent_queries") or []:
        qq = (q.get("question") or "").strip()
        if qq:
            parts.append(qq)
    return "\n".join(parts)
```

- [ ] **Step 4: Run** → PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_comms.py tests/test_biofield_comms.py
git commit -m "feat(biofield-b3b): comms_to_text flatten"
```

---

### Task 3: Prod endpoint `GET /api/people/recent-comms` (app.py)

**Files:**
- Modify: `app.py` (add the route near `/api/people`, ~line 16319)

**Interfaces:**
- Consumes: `recent_comms` (Task 1); module-level `LOG_DB`, `CONSOLE_SECRET`, `sqlite3`, `jsonify`, `request`.
- Produces: `GET /api/people/recent-comms?q=<email>` → `recent_comms(cx, email)` as JSON; console-gated like `/api/people`.

**NOTE — no offline pytest:** importing `app` requires prod credentials (Pinecone validates at import), so this route cannot be unit-tested offline. Its logic lives in `recent_comms` (Task 1, fully tested). Verification here = a syntax/parse check + post-deploy manual curl. The reviewer reviews the route against the `/api/people` auth pattern.

- [ ] **Step 1: Add the route** — in `app.py`, immediately after the `get_people()` route (`@app.route("/api/people", methods=["GET"])`, ~line 16319-16341), add:

```python
@app.route("/api/people/recent-comms", methods=["GET"])
def get_recent_comms():
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    from dashboard.recent_comms import recent_comms
    email = (request.args.get("q", "") or "").strip()
    with sqlite3.connect(LOG_DB) as cx:
        return jsonify(recent_comms(cx, email))
```

IMPORTANT placement: it must be defined BEFORE the `@app.route("/api/people/<int:person_id>")` route is not a concern (different path), but DO place `/api/people/recent-comms` so Flask's router matches the static path — Flask matches static rules before `<int:person_id>` converters, so order is safe; still, put it directly after `get_people()` for clarity.

- [ ] **Step 2: Syntax/parse check**

Run: `~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"`
Expected: `OK`

Also confirm the route + auth are present:
Run: `grep -n "recent-comms" app.py`
Expected: shows the new `@app.route("/api/people/recent-comms"...)` line.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(biofield-b3b): read-only GET /api/people/recent-comms endpoint"
```

(Manual post-deploy verification — after this branch merges to main and Render redeploys:
`doppler run -p remedy-match -c prd -- sh -c 'curl -s -H "X-Console-Key: $CONSOLE_SECRET" "https://illtowell.com/api/people/recent-comms?q=<email>"'`
should return the windowed `{intake_summary, recent_inquiries, recent_queries}` JSON.)

---

### Task 4: Local fetch + mine + route + hook

**Files:**
- Modify: `biofield_local_app.py`
- Test: `tests/test_biofield_mine_comms_routes.py`

**Interfaces:**
- Consumes: `comms_to_text` (Task 2), `interpret_stresses` (B2), `add_stress` (B3a), `_report_for`, `interpret_complete`, `_seed_stresses` (B3a).
- Produces: `create_app(..., fetch_recent_comms=None)` (new kwarg, default `_default_fetch_recent_comms`); `_mine_comms(cx, test_id)`; `POST /author/<test_id>/mine-comms` → `{"added": n}`/`{"added":0,"error":...}`; a best-effort run-once call to `_mine_comms` from `_seed_stresses`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_mine_comms_routes.py
import sqlite3
import pytest
from biofield_local_app import create_app


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}
_COMMS = {"intake_summary": "first name: Jane\n  Energy?: Low",
          "recent_inquiries": [{"main_challenge": "fatigue", "main_goal": "more energy"}],
          "recent_queries": [{"question": "why tired"}]}


def _app(db, comms, stresses):
    import json as _j
    return create_app(db, scan_lookup=lambda e: _NONE,
                      fetch_profile=lambda e: {},
                      fetch_recent_comms=lambda e: comms if e == "j@x.com" else {},
                      interpret_complete=lambda s, u: _j.dumps({"stresses": stresses}))


def _new(client, email):
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"name": "J", "email": email, "date": "2026-06-25"})
    return tid


def test_mine_comms_adds_comm_stresses(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, _COMMS, ["Chronic fatigue", "Poor sleep"]).test_client()
    tid = _new(client, "j@x.com")
    j = client.post(f"/author/{tid}/mine-comms", json={}).get_json()
    assert "error" not in j
    data = client.get(f"/author/{tid}/stresses").get_json()["data"]
    labels = {x["label"] for x in data["active"] + data["balanced"]}
    sources = {x["source"] for x in data["active"] + data["balanced"]}
    assert {"Chronic fatigue", "Poor sleep"} <= labels and "comm" in sources


def test_mine_comms_no_email(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, _COMMS, ["X"]).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    j = client.post(f"/author/{tid}/mine-comms", json={}).get_json()
    assert j["added"] == 0 and "error" in j


def test_mine_comms_empty(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, {}, []).test_client()
    tid = _new(client, "nobody@x.com")
    assert client.post(f"/author/{tid}/mine-comms", json={}).get_json()["added"] == 0


def test_header_save_mines_comms_run_once(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, _COMMS, ["Chronic fatigue"]).test_client()
    tid = _new(client, "j@x.com")   # header-save hook should have mined comms already
    data = client.get(f"/author/{tid}/stresses").get_json()["data"]
    labels = {x["label"] for x in data["active"] + data["balanced"]}
    assert "Chronic fatigue" in labels
```

- [ ] **Step 2: Run** → FAIL (`fetch_recent_comms` kwarg unknown / route 404).

- [ ] **Step 3: Implement** — in `biofield_local_app.py`:

(a) Module-level default fetcher, next to `_default_fetch_profile`:

```python
def _default_fetch_recent_comms(email):
    """Best-effort: pull a client's windowed recent comms from the prod endpoint.
    Returns {} on any failure (incl. missing CONSOLE_SECRET -> no network call)."""
    import json as _json
    import urllib.parse
    import urllib.request
    email = (email or "").strip()
    if not email:
        return {}
    try:
        key = os.environ["CONSOLE_SECRET"]
        base = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
        url = (f"{base}/api/people/recent-comms?key=" + urllib.parse.quote(key)
               + "&q=" + urllib.parse.quote(email))
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        return _json.load(urllib.request.urlopen(req, timeout=20)) or {}
    except Exception:
        return {}
```

(b) Add `fetch_recent_comms=None` to the `create_app(...)` signature (after `fetch_profile=None`), and resolve it next to `fetch_profile`:

```python
    fetch_recent_comms = fetch_recent_comms or _default_fetch_recent_comms
```

(c) `_mine_comms` closure, next to `_mine_profile`:

```python
    def _mine_comms(cx, test_id):
        """Mine the client's recent communications into comm stresses. Best-effort."""
        from dashboard.biofield_interpret import interpret_stresses
        from dashboard.biofield_comms import comms_to_text
        from dashboard import biofield_stress as _st
        rep = _report_for(cx, test_id)
        email = ((rep.get("client") or {}).get("email") or "").strip()
        if not email:
            return {"added": 0, "error": "No client selected yet"}
        try:
            ctx = fetch_recent_comms(email) or {}
            text = comms_to_text(ctx)
            labels = interpret_stresses(text, interpret_complete) if text.strip() else []
            added = sum(1 for label in labels if _st.add_stress(cx, test_id, label, source="comm"))
        except Exception as e:
            return {"added": 0, "error": str(e)[:200]}
        return {"added": added}
```

(d) Route, next to `author_mine_profile`:

```python
    @app.route("/author/<test_id>/mine-comms", methods=["POST"])
    def author_mine_comms(test_id):
        with sqlite3.connect(db_path) as cx:
            return _mine_comms(cx, test_id)
```

(e) In `_seed_stresses`, after the existing profile-mining block (the `if not ... source='tag' ...: _mine_profile` block), add a parallel comms block:

```python
        if not cx.execute(
                "SELECT 1 FROM biofield_auth_stress WHERE test_id=? AND source='comm' LIMIT 1",
                (int(str(test_id).lstrip("a") or 0),)).fetchone():
            try:
                _mine_comms(cx, test_id)
            except Exception:
                pass
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_mine_comms_routes.py tests/test_biofield_mine_profile_routes.py tests/test_biofield_stress_routes.py -v` → PASS (B3a mine-profile + B1 stress routes green).

- [ ] **Step 5: Commit**

```bash
git add biofield_local_app.py tests/test_biofield_mine_comms_routes.py
git commit -m "feat(biofield-b3b): fetch_recent_comms + mine-comms route + always-on hook"
```

---

### Task 5: UI — "Mine recent comms → stresses" button

**Files:**
- Modify: `dashboard/biofield_report_html.py`
- Test: `tests/test_biofield_mine_comms_button.py`

**Interfaces:**
- Consumes: existing `__TID__`/`post()`/`rstat()`/`loadStress()` conventions.
- Produces: a "Mine recent comms → stresses" button + `mineComms()` JS (POST `/author/__TID__/mine-comms` → `loadStress()`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_mine_comms_button.py
from dashboard.biofield_report_html import render_author_html


def _html():
    rep = {"test_id": "a7", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [], "schedule": []}
    return render_author_html(rep, [], "")


def test_mine_comms_button_and_handler():
    h = _html()
    assert "Mine recent comms" in h
    assert "function mineComms" in h
    assert "/author/a7/mine-comms" in h
    assert "loadStress()" in h
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — in `dashboard/biofield_report_html.py`:

(a) In `_AUTHOR_JS`, add `mineComms()` next to `mineProfile()` (mirrors it exactly):

```javascript
async function mineComms(){rstat('Mining recent comms for stresses\\u2026');
 var j=await post('/author/__TID__/mine-comms',{});
 if(j.error){rstat('Mine comms: '+j.error);return}
 rstat('Added '+j.added+' comm stress(es).');loadStress()}
```

(b) Add the button to the existing "Mine profile" button row (the `<div class=btnrow ...>` that contains the `mineProfile()` button, just before `<div id=stresspanel></div>`). Add a second button in that same row:

```python
                 "<button class='btn ghost' onclick=mineProfile()>Mine profile &rarr; stresses</button>"
                 "<button class='btn ghost' onclick=mineComms()>Mine recent comms &rarr; stresses</button>"
```
(Insert the mineComms button line immediately after the existing mineProfile button line, inside the same `btnrow` div. Preserve all surrounding markup.)

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_mine_comms_button.py tests/test_biofield_author_html.py tests/test_biofield_mine_profile_button.py -v` → PASS.

- [ ] **Step 5: Run the B3b + adjacent suite + commit**

```bash
~/.venvs/deploy-chat311/bin/python -m pytest \
  tests/test_recent_comms.py tests/test_biofield_comms.py \
  tests/test_biofield_mine_comms_routes.py tests/test_biofield_mine_comms_button.py \
  tests/test_biofield_mine_profile_routes.py tests/test_biofield_stress_routes.py \
  tests/test_biofield_stress_seed.py tests/test_biofield_stress_derive.py \
  tests/test_biofield_author_html.py tests/test_biofield_mine_profile_button.py -q
git add dashboard/biofield_report_html.py tests/test_biofield_mine_comms_button.py
git commit -m "feat(biofield-b3b): Mine-recent-comms button + mineComms()"
```

---

## Self-Review

**Spec coverage:**
- Windowed aggregation (7d chat/inquiries, latest intake) → Task 1. ✓
- Prod read-only console-gated endpoint → Task 3. ✓
- comms_to_text flatten → Task 2. ✓
- fetch + mine + source='comm' merged + route + always-on run-once hook → Task 4. ✓
- Button → Task 5. ✓
- Reachable trio only; email/GHL/PB deferred → not built (correct). ✓

**Placeholder scan:** No TBDs. Task 3 explicitly states no offline pytest is possible (app import needs creds) and gives a syntax check + manual post-deploy curl — honest, not a placeholder.

**Type consistency:** `recent_comms(cx, email, *, days_window, query_log_n) -> {intake_summary, recent_inquiries, recent_queries}` (T1) consumed by T3 endpoint + (via the dict shape) T2/T4; `comms_to_text(context) -> str` (T2) consumed by T4; `fetch_recent_comms(email) -> dict` (T4) defaulted to `_default_fetch_recent_comms`; `add_stress(..., source='comm')` reused from B3a. Consistent.

## Verification (manual, after all tasks + after merge/deploy)

1. Local suite green (Tasks 1,2,4,5). 
2. After merge to main + Render redeploy, curl the prod endpoint (Task 3 note) — confirm the windowed JSON.
3. `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- python3 biofield_local_app.py`; open a test for a client with recent chat/inquiries: "Mine recent comms → stresses" (and header-save) add comm stresses to the Active list, deduped against scan/voice/tag.
