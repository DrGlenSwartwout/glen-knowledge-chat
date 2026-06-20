# Begin Page #3 — Entry Points into One Record + Card Progress — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Capture ScoreApp quiz + E4L scan completions into `journey_state` by email, evolve the journey map to a per-card fractional fill, add a smart Scan door (portal vs signup), and rename Earn -> Give.

**Architecture:** Additive webhook wire-ins write entry-point completions to the one record via a new idempotent `_record_entry_unlock`. `begin_funnel.journey_map` is rewritten from binary status to per-card 2-step sub-steps with a `fill` fraction; `/begin/state` computes a small `signals` dict (predicate-derived Ambassador / Bring-a-friend / has-E4L) and passes it in. PB course/intake/masterclass completions ride the existing `/webhook/practice-better` via an editable event->gate map.

**Tech Stack:** Flask (Python 3.11), vanilla JS in `static/begin.html`, SQLite `journey_state` via `begin_funnel`, pytest + Flask test client.

## Global Constraints

- No emoji, no em dashes. Live page, no feature flag (`main` auto-deploys; manual visual pass required before launch).
- One record: all writes via `record_unlock` -> `journey_state` (email-union). Webhooks ALWAYS return 200; wire-ins wrapped, never alter existing behavior.
- New gates additive to `begin_funnel.VALID_TRIGGERS` (`unlocked_gates` is a JSON list — no migration): `course_ww`, `intake`, `masterclass`, `biofield`.
- `journey_map(state, ref, signals=None)` is pure; predicate steps are not-done and Scan routes to signup when `signals is None`.
- XSS-safe front-end: card text via `textContent`; the fill bar width set via `.style.width` only. No value interpolated into `innerHTML` except clearing.
- Copy provisional (BNSN later); keep strings in `JOURNEY_STEPS` + the JS constants.
- Test harness: `importlib.import_module("app")`/`"begin_funnel"` with repo_root on `sys.path` (skip if not importable); `monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path/"chat_log.db"))`; `begin_funnel.init_journey_tables(cx)`; `app_module.app.test_client()`; set `amg_session` cookie; mock `ghl_onboard_contact` + `_capture_concierge_referral` on free-tier transitions.
- Run tests: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. Spec: `docs/superpowers/specs/2026-06-19-begin-entry-points-one-record-design.md`.

---

### Task 1: Entry-capture helpers + Part A (quiz/scan -> one record)

**Files:** Modify `app.py`; Create `tests/test_begin_entry_points.py`.

**Interfaces produced:** `_entry_session_id(email) -> str`; `_record_entry_unlock(trigger, email, first_name="", last_name="", ref_slug="") -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_begin_entry_points.py
"""Begin #3 - entry points into the one record + predicates + PB wiring."""
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    monkeypatch.setattr(app_module, "ghl_onboard_contact", lambda *a, **k: {"contact_id": "x"})
    monkeypatch.setattr(app_module, "_capture_concierge_referral", lambda *a, **k: None)
    return db


def test_record_entry_unlock_writes_quiz_by_email(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    app_module._record_entry_unlock("quiz", "Ann@Example.com", first_name="Ann")
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="ann@example.com")
    assert "quiz" in st["unlocked_gates"]
    assert st["first_name"] == "Ann"


def test_record_entry_unlock_idempotent(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    app_module._record_entry_unlock("scan", "b@x.com")
    app_module._record_entry_unlock("scan", "b@x.com")
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM journey_events WHERE trigger='scan' AND email='b@x.com'").fetchone()[0]
    assert n == 1


def test_entry_unions_with_real_session(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.record_unlock(cx, session_id="sessA", trigger="name",
                                   email="c@x.com", first_name="Cee")
    app_module._record_entry_unlock("scan", "c@x.com")
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, session_id="sessA", email="c@x.com")
    assert "scan" in st["unlocked_gates"]


def test_record_entry_unlock_never_raises(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    import begin_funnel
    monkeypatch.setattr(begin_funnel, "record_unlock",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    app_module._record_entry_unlock("scan", "d@x.com")  # must not raise


def test_e4l_freshness_ingest_records_scan(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CONSOLE_SECRET", "k")
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "k", raising=False)
    client = app_module.app.test_client()
    r = client.post("/api/e4l/scan-freshness", json={"rows": [{"email": "e@x.com", "last_scan_date": "2026-06-19"}]},
                    headers={"X-Console-Key": "k"})
    assert r.status_code == 200
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="e@x.com")
    assert "scan" in st["unlocked_gates"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_entry_points.py -v`
Expected: FAIL (`_record_entry_unlock` not defined). NOTE: the e4l test also needs the Task-1 wire-in; it may stay failing until Step 3 wires the endpoint.

- [ ] **Step 3: Add the helpers + wire the two endpoints**

In `app.py`, add near the other journey helpers (e.g. just below `is_member`):

```python
def _entry_session_id(email):
    import hashlib
    return "entry:" + hashlib.sha1((email or "").strip().lower().encode()).hexdigest()[:16]


def _record_entry_unlock(trigger, email, first_name="", last_name="", ref_slug=""):
    """Write an entry-point completion to the one record by email. Idempotent
    (skips an already-present gate); never raises into the caller."""
    email = (email or "").strip().lower()
    if not email:
        return
    try:
        sid = _entry_session_id(email)
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            begin_funnel.init_journey_tables(cx)
            row = cx.execute(
                "SELECT unlocked_gates FROM journey_state WHERE session_id=?",
                (sid,)).fetchone()
            if row and trigger in set(json.loads(row[0] or "[]")):
                return  # already recorded - no duplicate event
            begin_funnel.record_unlock(
                cx, session_id=sid, trigger=trigger, email=email,
                first_name=first_name, last_name=last_name, ref_slug=ref_slug)
    except Exception as e:
        print(f"[entry-unlock] {trigger} {e!r}", flush=True)
```

In `scoreapp_webhook` (the `if utm_source:` referral block area, before `return jsonify(...)`), add after the referral logging:

```python
    _record_entry_unlock("quiz", email, first, last, utm_source)
```

In `api_e4l_scan_freshness`, after `_sf.upsert(cx, rows)` (and outside that `with`), add:

```python
    for _r in rows:
        if (_r.get("last_scan_date") or _r.get("scan_date") or "").strip():
            _record_entry_unlock("scan", (_r.get("email") or ""))
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_entry_points.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_begin_entry_points.py
git commit -m "feat: begin #3 entry-point completions into the one record"
```

---

### Task 2: journey_map fractional fill + new gates + Give rename

**Files:** Modify `begin_funnel.py`; Update `tests/test_begin_journey_map.py`.

**Interfaces produced:** `JOURNEY_STEPS` (per-card `steps[]`); `journey_map(state, ref="", signals=None) -> [{key,label,paren,href,status,fill,steps:[{key,label,done}]}]`.

- [ ] **Step 1: Update the existing journey tests to the new contract**

Replace the body of `tests/test_begin_journey_map.py` status/href tests with:

```python
def test_no_signals_scan_is_next():
    bf = _bf()
    m = bf.journey_map(_state([]), "")
    assert [c["key"] for c in m] == ["scan", "find", "heal", "give"]
    assert m[0]["status"] == "next" and m[0]["fill"] == 0.0
    assert all(c["fill"] == 0.0 for c in m)


def test_scan_gate_half_fills_scan():
    bf = _bf()
    m = bf.journey_map(_state(["scan"]), "")
    by = {c["key"]: c for c in m}
    assert by["scan"]["fill"] == 0.5 and by["scan"]["status"] == "next"


def test_scan_complete_advances_next_to_find():
    bf = _bf()
    m = bf.journey_map(_state(["scan", "course_ww"]), "")
    by = {c["key"]: c for c in m}
    assert by["scan"]["fill"] == 1.0 and by["scan"]["status"] == "done"
    assert by["find"]["status"] == "next"


def test_give_label_and_predicate_ambassador():
    bf = _bf()
    m = bf.journey_map(_state([]), "", {"ambassador": True})
    by = {c["key"]: c for c in m}
    assert by["give"]["label"] == "Give"
    assert by["give"]["fill"] == 0.5
    assert by["give"]["steps"][0]["done"] is True


def test_smart_scan_href():
    bf = _bf()
    sign = bf.journey_map(_state([]), "slug", {"has_e4l": False})
    nos = {c["key"]: c for c in sign}
    assert nos["scan"]["href"] == "https://truly.vip/E4L"
    have = bf.journey_map(_state([]), "slug", {"has_e4l": True})
    yes = {c["key"]: c for c in have}
    assert yes["scan"]["href"] == "https://portal.e4l.com"
```

Keep `test_thread_href_external_threads_utm` and `test_card_href_unchanged_after_refactor` unchanged. Delete the old `test_labels_and_internal_hrefs` / `test_*_done_*` that assumed the binary shape (they are replaced above).

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_journey_map.py -v`
Expected: FAIL (new shape not implemented).

- [ ] **Step 3: Add gates, rewrite JOURNEY_STEPS + journey_map**

In `begin_funnel.py`, add to `VALID_TRIGGERS` the four names: `"course_ww", "intake", "masterclass", "biofield"`.

Replace `JOURNEY_STEPS` and `journey_map` with:

```python
JOURNEY_STEPS = [
    {"key": "scan", "label": "Scan", "paren": "Your Biofield", "steps": [
        {"key": "voice_scan", "label": "Voice scan",          "src": ("gate", "scan"),       "href": None},
        {"key": "ww_course",  "label": "Wellness Whispering", "src": ("gate", "course_ww"),  "href": "https://truly.vip/GetWell"}]},
    {"key": "find", "label": "Find", "paren": "Your Remedy Match", "steps": [
        {"key": "match_chat", "label": "Match via chat",      "src": ("gate", "question"),   "href": "/begin/match"},
        {"key": "biofield",   "label": "Biofield interpretation", "src": ("gate", "biofield"), "href": "/begin/match"}]},
    {"key": "heal", "label": "Heal", "paren": "the root causes", "steps": [
        {"key": "intake",      "label": "Intake form",        "src": ("gate", "intake"),      "href": "https://truly.vip/Join"},
        {"key": "masterclass", "label": "ASH MasterClass",    "src": ("gate", "masterclass"), "href": "https://truly.vip/Intro"}]},
    {"key": "give", "label": "Give", "paren": "lift others", "steps": [
        {"key": "ambassador",   "label": "Be an Ambassador",  "src": ("predicate", "ambassador"),     "href": "/affiliate/apply"},
        {"key": "bring_friend", "label": "Bring a friend",    "src": ("predicate", "referred_friend"), "href": "/begin/path"}]},
]


def _step_done(step, gates, signals):
    kind, name = step["src"]
    if kind == "gate":
        return name in gates
    return bool((signals or {}).get(name))


def _scan_first_href(signals, ref):
    base = "https://portal.e4l.com" if (signals or {}).get("has_e4l") else "https://truly.vip/E4L"
    return _thread_href(base, ref, "begin-journey-scan")


def journey_map(state, ref="", signals=None):
    """Per-card fractional progress. Each card has an ordered sub-step list;
    fill = done/total; status = done(>=1.0) / next(first<1.0) / available.
    href = the first undone step's destination (smart for Scan). Pure."""
    gates = set((state or {}).get("unlocked_gates") or ())
    out = []
    next_assigned = False
    for card in JOURNEY_STEPS:
        steps_out = []
        done_count = 0
        first_undone_href = None
        for step in card["steps"]:
            done = _step_done(step, gates, signals)
            if done:
                done_count += 1
            elif first_undone_href is None:
                if card["key"] == "scan" and step["key"] == "voice_scan":
                    first_undone_href = _scan_first_href(signals, ref)
                else:
                    first_undone_href = _thread_href(step["href"], ref, f"begin-journey-{card['key']}")
            steps_out.append({"key": step["key"], "label": step["label"], "done": done})
        total = len(card["steps"])
        fill = round(done_count / total, 3) if total else 0.0
        if fill >= 1.0:
            status = "done"
        elif not next_assigned:
            status = "next"; next_assigned = True
        else:
            status = "available"
        if first_undone_href is None:  # all steps done -> link to the card's entry dest
            if card["key"] == "scan":
                first_undone_href = _scan_first_href(signals, ref)
            else:
                first_undone_href = _thread_href(card["steps"][0]["href"], ref, f"begin-journey-{card['key']}")
        out.append({"key": card["key"], "label": card["label"], "paren": card["paren"],
                    "href": first_undone_href, "status": status, "fill": fill, "steps": steps_out})
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_journey_map.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add begin_funnel.py tests/test_begin_journey_map.py
git commit -m "feat: begin #3 journey_map fractional fill + new gates + Give rename"
```

---

### Task 3: /begin/state computes signals + passes to journey_map

**Files:** Modify `app.py` (`begin_state` + 3 helpers); add tests to `tests/test_begin_entry_points.py`.

**Interfaces consumed:** `begin_funnel.journey_map(state, ref, signals)` (Task 2).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_begin_entry_points.py`:

```python
def test_state_predicates_light_give(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS affiliate_signups (email TEXT, slug TEXT, status TEXT)")
        cx.execute("INSERT INTO affiliate_signups (email, slug, status) VALUES (?,?,?)",
                   ("amb@x.com", "amb", "approved"))
        cx.commit()
    client = app_module.app.test_client(); client.set_cookie("amg_session", "s1")
    # activate so email is on the session row, then read state with that email
    app_module._record_entry_unlock("quiz", "amb@x.com", first_name="Amb")
    with sqlite3.connect(db) as cx:
        import begin_funnel
        begin_funnel.record_unlock(cx, session_id="s1", trigger="tos",
                                   email="amb@x.com", tos=True)
    body = client.get("/begin/state").get_json()
    give = [c for c in body["journey_map"] if c["key"] == "give"][0]
    assert give["steps"][0]["done"] is True


def test_state_scan_gate_routes_to_portal(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client(); client.set_cookie("amg_session", "s2")
    app_module._record_entry_unlock("scan", "p@x.com")
    with sqlite3.connect(db) as cx:
        import begin_funnel
        begin_funnel.record_unlock(cx, session_id="s2", trigger="tos",
                                   email="p@x.com", tos=True)
    body = client.get("/begin/state").get_json()
    scan = [c for c in body["journey_map"] if c["key"] == "scan"][0]
    assert scan["href"] == "https://portal.e4l.com"
```

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_entry_points.py -k "state_" -v`
Expected: FAIL (signals not wired; Scan routes to signup).

- [ ] **Step 3: Add predicate helpers + wire begin_state**

In `app.py`, add near `_record_entry_unlock`:

```python
def _is_ambassador(cx, email):
    if not email:
        return False
    try:
        return cx.execute("SELECT 1 FROM affiliate_signups WHERE LOWER(email)=? AND status='approved' LIMIT 1",
                          (email.lower(),)).fetchone() is not None
    except Exception:
        return False


def _has_referred_friend(cx, email):
    if not email:
        return False
    try:
        return cx.execute("SELECT 1 FROM referral_redemptions WHERE LOWER(owner_email)=? LIMIT 1",
                          (email.lower(),)).fetchone() is not None
    except Exception:
        return False


def _has_e4l(cx, email, state):
    if "scan" in set(state.get("unlocked_gates") or ()):
        return True
    if not email:
        return False
    try:
        from dashboard import scan_freshness as _sf
        _sf.init_table(cx)
        return cx.execute("SELECT 1 FROM scan_freshness WHERE LOWER(email)=? LIMIT 1",
                          (email.lower(),)).fetchone() is not None
    except Exception:
        return False
```

In `begin_state`, replace the journey_map injection line with a signals-computing block (the route already has `state`, `email`, `ref_slug`):

```python
    with sqlite3.connect(LOG_DB) as _cx:
        signals = {
            "ambassador": _is_ambassador(_cx, email),
            "referred_friend": _has_referred_friend(_cx, email),
            "has_e4l": _has_e4l(_cx, email, state),
        }
    payload["journey_map"] = begin_funnel.journey_map(state, ref_slug, signals)
```

- [ ] **Step 4: Run to verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_entry_points.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_begin_entry_points.py
git commit -m "feat: begin #3 /begin/state signals (ambassador, referred-friend, has-e4l)"
```

---

### Task 4: Front-end fill bar + Give + honest question signal

**Files:** Modify `static/begin.html`; update serve assertions in `tests/test_begin_journey_map.py`.

**Interfaces consumed:** `STATE.journey_map[].fill/steps/status` (Task 3); existing `renderJourney`/`refreshJourney`/`unlock` from #2/#1.

- [ ] **Step 1: Add fill-bar CSS**

Before `</style>`, add:

```css
  .journey-card .jc-fill { margin-top: 10px; height: 5px; border-radius: 3px; background: rgba(255,255,255,0.10); overflow: hidden; }
  .journey-card .jc-fill > i { display: block; height: 100%; width: 0; background: var(--gold); border-radius: 3px; transition: width .5s ease; }
```

- [ ] **Step 2: Rewrite renderJourney + constants + drop click-gate**

Replace the `JOURNEY_TRIGGER` constant and the `renderJourney` body. Set `JOURNEY_FALLBACK` to:

```javascript
  var JOURNEY_FALLBACK = [
    { key:'scan', label:'Scan', paren:'Your Biofield',     href:'https://truly.vip/E4L', status:'available', fill:0, steps:[] },
    { key:'find', label:'Find', paren:'Your Remedy Match', href:'/begin/match',  status:'available', fill:0, steps:[] },
    { key:'heal', label:'Heal', paren:'the root causes',   href:'https://truly.vip/Join', status:'available', fill:0, steps:[] },
    { key:'give', label:'Give', paren:'lift others',       href:'/affiliate/apply', status:'available', fill:0, steps:[] }
  ];
```

Delete the `var JOURNEY_TRIGGER = {...};` line. Replace `renderJourney` with:

```javascript
  function renderJourney(){
    var wrap = document.getElementById('journey-cards');
    if (!wrap) return;
    var steps = (STATE && STATE.journey_map) || JOURNEY_FALLBACK;
    wrap.innerHTML = '';
    steps.forEach(function(s){
      var a = document.createElement('a');
      a.className = 'journey-card status-' + (s.status || 'available');
      a.href = s.href || '#';
      var lab = document.createElement('span');
      lab.className = 'jc-label'; lab.textContent = s.label;
      if (s.status === 'done') {
        var ck = document.createElement('span');
        ck.className = 'jc-check'; ck.setAttribute('aria-hidden', 'true'); ck.textContent = ' [v]';
        lab.appendChild(ck);
      }
      var par = document.createElement('span');
      par.className = 'jc-paren'; par.textContent = s.paren;
      var fillWrap = document.createElement('span');
      fillWrap.className = 'jc-fill';
      var fillBar = document.createElement('i');
      fillBar.style.width = Math.round((s.fill || 0) * 100) + '%';
      fillWrap.appendChild(fillBar);
      a.appendChild(lab); a.appendChild(par); a.appendChild(fillWrap);
      if (s.status === 'next') {
        var tg = document.createElement('span');
        tg.className = 'jc-tag'; tg.textContent = 'your next step';
        a.appendChild(tg);
      }
      wrap.appendChild(a);   // cards navigate via href; completions are real signals now
    });
  }
```

- [ ] **Step 3: Fire `question` from the hero chat's first message**

In the hero chat `send()` (the block that does `heroAppend('user', q); heroHistory.push(...); heroExchanges += 1;`), add a one-time `question` unlock so Find step 1 reflects real engagement. Right after `heroExchanges += 1;` add:

```javascript
      if (heroExchanges === 1 && typeof unlock === 'function') {
        try { unlock('question'); } catch (_) {}
      }
```

- [ ] **Step 4: Update serve assertions**

In `tests/test_begin_journey_map.py`, update `test_begin_serves_journey_strip` (or the relevant serve test): assert the page contains `"Give"`, `jc-fill`, and does NOT contain `JOURNEY_TRIGGER`. Concretely add:

```python
def test_begin_serves_give_and_fill(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    assert "Give" in html
    assert "jc-fill" in html
    assert "JOURNEY_TRIGGER" not in html
```

(`_load_app` already exists in this file from #2.)

- [ ] **Step 5: Run the tests**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_journey_map.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add static/begin.html tests/test_begin_journey_map.py
git commit -m "feat: begin #3 card fill bar + Give label + question-on-first-message"
```

Note for the reviewer: manual visual pass required (fill bars per card, Give label, smart Scan link target).

---

### Task 5: Practice Better completion wiring (Increment 2)

**Files:** Modify `app.py` (`pb_webhook` + `PB_EVENT_GATES`); add a test to `tests/test_begin_entry_points.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_begin_entry_points.py`:

```python
def test_pb_completion_sets_gate(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "WEBHOOK_SECRET", "", raising=False)
    client = app_module.app.test_client()
    r = client.post("/webhook/practice-better",
                    json={"event_type": "wellness-whispering.completed", "email": "g@x.com", "name": "Gee Aitch"})
    assert r.status_code == 200
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="g@x.com")
    assert "course_ww" in st["unlocked_gates"]


def test_pb_unmapped_event_sets_no_gate(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "WEBHOOK_SECRET", "", raising=False)
    client = app_module.app.test_client()
    r = client.post("/webhook/practice-better",
                    json={"event_type": "client.created", "email": "h@x.com", "name": "H"})
    assert r.status_code == 200
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="h@x.com")
    assert "course_ww" not in st["unlocked_gates"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_entry_points.py -k "pb_" -v`
Expected: FAIL (no gate set).

- [ ] **Step 3: Add PB_EVENT_GATES + wire pb_webhook**

In `app.py`, add a module-level map just above `pb_webhook`:

```python
# PB internal automations POST completion events; each maps to a journey gate.
# Confirm the exact event identifiers once the PB automations are configured.
PB_EVENT_GATES = {
    "wellness-whispering.completed": "course_ww",
    "intake.completed":             "intake",
    "ash-masterclass.completed":    "masterclass",
}
```

In `pb_webhook`, after the existing `pb_events` insert block (before the signup branch), add:

```python
    if event_type in PB_EVENT_GATES and pb_email:
        _parts = pb_name.split(" ", 1) if pb_name else ["", ""]
        _record_entry_unlock(PB_EVENT_GATES[event_type], pb_email,
                             _parts[0], _parts[1] if len(_parts) > 1 else "")
```

- [ ] **Step 4: Run to verify pass + the begin sweep**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_entry_points.py tests/test_begin_journey_map.py -v`
Then: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k "begin" -v`
Expected: all PASS; no regressions in `test_begin_routes`/`test_begin_funnel`/`test_begin_hero_identity`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_begin_entry_points.py
git commit -m "feat: begin #3 Practice Better completion -> journey gates"
```

---

## Self-Review

**1. Spec coverage:** Part A capture (T1); fractional fill + new gates + Give (T2); signals/predicates + smart routing in the payload (T3); fill-bar render + Give + honest `question` (T4); PB completions (T5). The deferred `biofield` gate is added (T2 VALID_TRIGGERS) and appears as an undone Find step; nothing sets it (correct — #4). Smart Scan routing: T2 (journey_map) + T3 (has_e4l signal).

**2. Placeholder scan:** No TBD/handle-edge-cases. The PB event identifiers in `PB_EVENT_GATES` are concrete strings flagged as confirm-on-config in one editable map (a real Glen-side config dependency, not a code placeholder).

**3. Type consistency:** `_record_entry_unlock(trigger, email, first_name, last_name, ref_slug)` used identically in T1/T5. `journey_map(state, ref, signals)` returns `{key,label,paren,href,status,fill,steps}` consumed by T3 tests and T4 render. `signals` keys `ambassador/referred_friend/has_e4l` match T2 predicate-src names and T3 helpers. Card keys `scan/find/heal/give` consistent across `JOURNEY_STEPS`, tests, and `JOURNEY_FALLBACK`. New gates `course_ww/intake/masterclass/biofield` consistent in `VALID_TRIGGERS`, `JOURNEY_STEPS`, and `PB_EVENT_GATES`.
