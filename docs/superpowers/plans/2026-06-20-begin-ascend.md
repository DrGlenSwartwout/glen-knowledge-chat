# Begin #5 - Personalized Ascend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/begin/ascend` a goal-driven, personalized high-ticket escalation - a short "where do you want to go?" question picks a track, the member's one-record state picks the rung within it, that rung is the hero, and "Book your consultation" becomes a real capture-and-notify inquiry - all behind `ASCEND_PERSONALIZED_ENABLED` (default off).

**Architecture:** A pure `recommend_ascend(goal, reached)` in `begin_funnel.py` maps the 3 goals onto the `TIER_CATALOG` rungs. A `GET /begin/ascend/recommend` endpoint resolves the member, derives which rungs they have reached, and returns the recommended rung + the full ladder. A `POST /begin/ascend/inquire` endpoint (member-gated) records a consultation inquiry, tags GHL, emails Glen, and sets the `ascend` one-record gate. The `begin-ascend.html` page renders the goal chooser + personalized hero + inquiry capture when the flag is on, and the existing static ladder when off.

**Tech Stack:** Python 3.11 / Flask (single `app.py`), `begin_funnel.py`, SQLite (`LOG_DB`), pytest. Front-end is vanilla JS in `static/begin-ascend.html`.

## Global Constraints

- No emoji, no em dashes (code, comments, commit messages).
- Behind `ASCEND_PERSONALIZED_ENABLED` (default off). When off: `/begin/ascend` serves the existing static ladder unchanged, `recommend` -> `{enabled:false}`, `inquire` -> `{ok:false}`. No behavior change when off.
- NOT a money path: the high-ticket tiers sell via consultation. `inquire` is capture-and-notify only (no charge, no Stripe).
- Member-gated: `inquire` requires `is_member` (ToS); non-member -> 403 `{need_optin:true}` (the existing OptinGate pattern).
- All outward effects (GHL tag, Glen email) are best-effort and wrapped; the inquiry succeeds as long as its row is written; endpoints never 500 except a final catch.
- `recommend_ascend` is pure and total - any bad input yields the heal entry rung (`biofield-analysis`), never an exception.
- Reuse, do not duplicate: `_send_inquiry_email`, `ghl_onboard_contact`, `_record_entry_unlock`, `is_member`, `_active_membership_for_email`, `get_authenticated_user`, `begin_funnel.get_state`, `TIER_CATALOG`.
- Test runner: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. Tests skip if app/begin_funnel not importable. Toggle the flag via `monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True/False, raising=False)`. tmp `LOG_DB` via `monkeypatch.setattr(app_module, "LOG_DB", db)`.

---

## Critical files

- `begin_funnel.py`
  - Add `"ascend"` to `VALID_TRIGGERS` (~line 81) so `_record_entry_unlock("ascend", ...)` is accepted (no schema change; `unlocked_gates` is a JSON list).
  - Add `ASCEND_TRACKS` + `recommend_ascend(goal, reached=())` near `TIER_CATALOG` (~line 738-769).
- `app.py`
  - Add `ASCEND_PERSONALIZED_ENABLED` near the other feature flags (search `BIOFIELD_CART_ENABLED`).
  - Add `_ascend_reached(cx, email, state)` + `GET /begin/ascend/recommend` + `POST /begin/ascend/inquire` + `_init_ascend_inquiries(cx)`. Place the new routes next to the existing ascend routes (`begin_ascend` ~1376-1408).
- `static/begin-ascend.html`: goal chooser + personalized hero + inquiry capture + confirmation (flag-on); static ladder unchanged (flag-off).
- Tests: `tests/test_begin_ascend.py` (new).

---

## Task 1: `recommend_ascend` + `ASCEND_TRACKS` + `ascend` trigger (begin_funnel.py)

**Files:**
- Modify: `begin_funnel.py` (add `"ascend"` to `VALID_TRIGGERS`; add `ASCEND_TRACKS` + `recommend_ascend`)
- Test: `tests/test_begin_ascend.py`

**Interfaces:**
- Produces:
  - `ASCEND_TRACKS: dict[str, list[str]]` mapping `"heal"|"learn"|"build"` to ordered `TIER_CATALOG` slug lists.
  - `recommend_ascend(goal: str, reached=()) -> str` - the recommended `TIER_CATALOG` slug. Pure, total.
  - `"ascend"` is now in `VALID_TRIGGERS`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_begin_ascend.py`:

```python
# tests/test_begin_ascend.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_bf():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("begin_funnel")
    except Exception as e:
        pytest.skip(f"begin_funnel not importable: {e}")


def test_recommend_heal_entry():
    bf = _load_bf()
    assert bf.recommend_ascend("heal") == "biofield-analysis"


def test_recommend_learn_entry():
    bf = _load_bf()
    assert bf.recommend_ascend("learn") == "certification"


def test_recommend_learn_certified_bumps():
    bf = _load_bf()
    assert bf.recommend_ascend("learn", reached={"certification"}) == "one-to-one"


def test_recommend_build_entry():
    bf = _load_bf()
    assert bf.recommend_ascend("build") == "one-to-one"


def test_recommend_build_practitioner_bumps():
    bf = _load_bf()
    assert bf.recommend_ascend("build", reached={"one-to-one"}) == "healing-oasis-tools"


def test_recommend_unknown_goal_falls_back_to_heal():
    bf = _load_bf()
    assert bf.recommend_ascend("nonsense") == "biofield-analysis"
    assert bf.recommend_ascend("") == "biofield-analysis"
    assert bf.recommend_ascend(None) == "biofield-analysis"


def test_recommend_all_reached_returns_track_top():
    bf = _load_bf()
    allslugs = set(bf.TIER_CATALOG.keys())
    assert bf.recommend_ascend("build", reached=allslugs) == "consultant-package"


def test_recommend_returns_valid_catalog_slug():
    bf = _load_bf()
    for goal in ("heal", "learn", "build", "x"):
        assert bf.recommend_ascend(goal) in bf.TIER_CATALOG


def test_ascend_is_valid_trigger():
    bf = _load_bf()
    assert "ascend" in bf.VALID_TRIGGERS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_ascend.py -v`
Expected: FAIL (`AttributeError: module 'begin_funnel' has no attribute 'recommend_ascend'`, and `"ascend" not in VALID_TRIGGERS`).

- [ ] **Step 3: Add `"ascend"` to `VALID_TRIGGERS`**

In `begin_funnel.py`, change the `VALID_TRIGGERS` set (~line 81-87) to include `"ascend"`:

```python
VALID_TRIGGERS = {
    "load", "video", "scroll", "question", "name", "email", "tos",
    "voice", "scan", "quiz", "paid_fork", "purchase", "share_video",
    "deep_link",
    "course_ww", "intake", "masterclass", "biofield", "ascend",
}
```

- [ ] **Step 4: Add `ASCEND_TRACKS` + `recommend_ascend`**

In `begin_funnel.py`, immediately after `tier_for` (~line 769), add:

```python
# Goal -> ordered high-ticket track (slugs into TIER_CATALOG). The recommended
# rung is the lowest one the member has not reached; default to the entry rung.
ASCEND_TRACKS = {
    "heal":  ["biofield-analysis"],
    "learn": ["certification", "one-to-one"],
    "build": ["one-to-one", "healing-oasis-tools", "hawaii-immersion", "consultant-package"],
}


def recommend_ascend(goal, reached=()):
    """Recommended TIER_CATALOG slug for a goal + the set of rungs already reached.
    Pure and total: the first rung in the goal's track not in `reached`; if all are
    reached, the track's top rung; an unknown/missing goal falls back to the heal
    track (entry rung biofield-analysis)."""
    track = ASCEND_TRACKS.get((goal or "").strip().lower()) or ASCEND_TRACKS["heal"]
    reached = set(reached or ())
    for slug in track:
        if slug not in reached:
            return slug
    return track[-1]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_ascend.py -v`
Expected: PASS (all 9 tests).

- [ ] **Step 6: Commit**

```bash
git add begin_funnel.py tests/test_begin_ascend.py
git commit -m "feat: begin #5 recommend_ascend track mapping + ascend trigger"
```

---

## Task 2: Flag + signal builder + `GET /begin/ascend/recommend` (app.py)

**Files:**
- Modify: `app.py` (add `ASCEND_PERSONALIZED_ENABLED`; add `_ascend_reached`; add the recommend route)
- Test: `tests/test_begin_ascend.py`

**Interfaces:**
- Consumes: `begin_funnel.recommend_ascend`, `begin_funnel.TIER_CATALOG`, `begin_funnel.get_state`, `get_authenticated_user(request)`, `_active_membership_for_email(email)`, `_db_lock`, `LOG_DB`.
- Produces:
  - `ASCEND_PERSONALIZED_ENABLED: bool`
  - `_ascend_reached(cx, email, state) -> set[str]` - the rungs the member has reached (v1: `{"biofield-analysis"}` when a paid member or a biofield/paid_fork/purchase gate is present; else `set()`). Never raises.
  - `GET /begin/ascend/recommend?goal=<heal|learn|build>` -> `{ok, enabled, goal, recommended, ladder, is_member}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_begin_ascend.py` (these load `app`, distinct from the `begin_funnel`-only helper above):

```python
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
        cx.commit()
    return db


def test_recommend_endpoint_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", False, raising=False)
    r = app_module.app.test_client().get("/begin/ascend/recommend?goal=heal")
    body = r.get_json()
    assert body["ok"] is True and body["enabled"] is False


def test_recommend_endpoint_returns_hero_and_ladder(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    r = app_module.app.test_client().get("/begin/ascend/recommend?goal=learn")
    body = r.get_json()
    assert body["ok"] is True and body["enabled"] is True
    assert body["recommended"]["slug"] == "certification"
    # full ladder, ordered by n
    ns = [t["n"] for t in body["ladder"]]
    assert ns == sorted(ns) and len(body["ladder"]) == len(app_module.begin_funnel.TIER_CATALOG)


def test_recommend_endpoint_member_reaches_biofield(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    # A paid member asking the heal track has already reached biofield-analysis ->
    # the only heal rung is reached, so the recommendation is the track top (still biofield-analysis).
    r = app_module.app.test_client().get("/begin/ascend/recommend?goal=heal")
    assert r.get_json()["recommended"]["slug"] == "biofield-analysis"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_ascend.py -k recommend_endpoint -v`
Expected: FAIL (route 404 / flag missing).

- [ ] **Step 3: Add the flag**

In `app.py`, next to the `BIOFIELD_CART_ENABLED = ...` line, add:

```python
ASCEND_PERSONALIZED_ENABLED = os.environ.get("ASCEND_PERSONALIZED_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 4: Add `_ascend_reached` + the recommend route**

In `app.py`, immediately after `begin_ascend_tier_data` (~line 1408), add:

```python
def _ascend_reached(cx, email, state):
    """Rungs this member has already reached (v1). A paid member or a
    biofield/paid_fork/purchase gate marks the $300 Biofield rung reached.
    Never raises. (Practitioner-track signals are a future extension.)"""
    reached = set()
    try:
        if email and _active_membership_for_email(email):
            reached.add("biofield-analysis")
    except Exception:
        pass
    try:
        gates = set((state or {}).get("unlocked_gates") or ())
        if gates & {"biofield", "paid_fork", "purchase"}:
            reached.add("biofield-analysis")
    except Exception:
        pass
    return reached


@app.route("/begin/ascend/recommend")
def begin_ascend_recommend():
    """Personalized rung recommendation + the full ladder. Never raises."""
    if not ASCEND_PERSONALIZED_ENABLED:
        return jsonify({"ok": True, "enabled": False})
    try:
        goal = (request.args.get("goal") or "heal").strip().lower()
        session_id = (request.cookies.get("amg_session") or "").strip()
        auth_user = get_authenticated_user(request)
        email = (auth_user["email"] if auth_user else "") or ""
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            state = begin_funnel.get_state(cx, session_id=session_id, email=email)
        resolved_email = (state.get("email") or email or "").strip().lower()
        with sqlite3.connect(LOG_DB) as cx:
            reached = _ascend_reached(cx, resolved_email, state)
        slug = begin_funnel.recommend_ascend(goal, reached)
        ladder = sorted(begin_funnel.TIER_CATALOG.values(), key=lambda t: t.get("n", 0))
        is_member_now = bool(is_member(session_id, resolved_email))
        return jsonify({"ok": True, "enabled": True, "goal": goal,
                        "recommended": begin_funnel.TIER_CATALOG.get(slug),
                        "ladder": ladder, "is_member": is_member_now})
    except Exception as e:
        print(f"[ascend-recommend] {e!r}", flush=True)
        return jsonify({"ok": True, "enabled": True, "goal": "heal",
                        "recommended": begin_funnel.TIER_CATALOG.get("biofield-analysis"),
                        "ladder": sorted(begin_funnel.TIER_CATALOG.values(), key=lambda t: t.get("n", 0)),
                        "is_member": False})
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_ascend.py -k recommend -v`
Expected: PASS (the recommend endpoint tests plus the Task 1 recommend tests).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_begin_ascend.py
git commit -m "feat: begin #5 ASCEND_PERSONALIZED_ENABLED + recommend endpoint"
```

---

## Task 3: `POST /begin/ascend/inquire` + `ascend_inquiries` table (app.py)

**Files:**
- Modify: `app.py` (add `_init_ascend_inquiries` + the inquire route)
- Test: `tests/test_begin_ascend.py`

**Interfaces:**
- Consumes: `ASCEND_PERSONALIZED_ENABLED`, `begin_funnel.TIER_CATALOG`, `is_member`, `get_authenticated_user`, `begin_funnel.get_state`, `ghl_onboard_contact(email, first_name, last_name, phone, source_tag, extra_tags)`, `_send_inquiry_email(to_email, subject, body, reply_to=None)`, `_record_entry_unlock("ascend", email)`, `_db_lock`, `LOG_DB`.
- Produces:
  - `_init_ascend_inquiries(cx)` - idempotent `CREATE TABLE IF NOT EXISTS ascend_inquiries (email TEXT, slug TEXT, goal TEXT, note TEXT, created_at TEXT, PRIMARY KEY(email, slug))`.
  - `POST /begin/ascend/inquire` -> `{ok}` (member-gated; records the inquiry; tags GHL; emails Glen; sets the `ascend` gate).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_begin_ascend.py`:

```python
def _seed_member(app_module, monkeypatch):
    # ToS member (ordering gate) for the inquire path.
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "t@x.com"})


def test_inquire_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", False, raising=False)
    r = app_module.app.test_client().post("/begin/ascend/inquire", json={"slug": "biofield-analysis", "goal": "heal"})
    assert r.get_json().get("ok") is False


def test_inquire_non_member_403(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "t@x.com"})
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": False)
    r = app_module.app.test_client().post("/begin/ascend/inquire", json={"slug": "biofield-analysis", "goal": "heal"})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_inquire_unknown_slug_400(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    _seed_member(app_module, monkeypatch)
    r = app_module.app.test_client().post("/begin/ascend/inquire", json={"slug": "not-a-tier", "goal": "heal"})
    assert r.status_code == 400 and r.get_json().get("ok") is False


def test_inquire_member_records_and_notifies(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    _seed_member(app_module, monkeypatch)
    calls = {"ghl": 0, "email": 0}
    monkeypatch.setattr(app_module, "ghl_onboard_contact", lambda *a, **k: calls.__setitem__("ghl", calls["ghl"] + 1))
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: calls.__setitem__("email", calls["email"] + 1) or True)
    r = app_module.app.test_client().post("/begin/ascend/inquire", json={"slug": "certification", "goal": "learn", "note": "ready"})
    assert r.get_json() == {"ok": True}
    with sqlite3.connect(db) as cx:
        rows = cx.execute("SELECT email, slug, goal, note FROM ascend_inquiries").fetchall()
    assert rows == [("t@x.com", "certification", "learn", "ready")]
    assert calls["ghl"] == 1 and calls["email"] == 1


def test_inquire_best_effort_email_failure_still_ok(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    _seed_member(app_module, monkeypatch)
    def _boom(*a, **k):
        raise RuntimeError("smtp down")
    monkeypatch.setattr(app_module, "ghl_onboard_contact", _boom)
    monkeypatch.setattr(app_module, "_send_inquiry_email", _boom)
    r = app_module.app.test_client().post("/begin/ascend/inquire", json={"slug": "biofield-analysis", "goal": "heal"})
    assert r.get_json() == {"ok": True}  # row written despite GHL/email failure
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM ascend_inquiries").fetchone()[0] == 1


def test_inquire_idempotent_per_email_slug(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    _seed_member(app_module, monkeypatch)
    monkeypatch.setattr(app_module, "ghl_onboard_contact", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    c = app_module.app.test_client()
    c.post("/begin/ascend/inquire", json={"slug": "certification", "goal": "learn", "note": "first"})
    c.post("/begin/ascend/inquire", json={"slug": "certification", "goal": "build", "note": "second"})
    with sqlite3.connect(db) as cx:
        rows = cx.execute("SELECT goal, note FROM ascend_inquiries WHERE email='t@x.com' AND slug='certification'").fetchall()
    assert rows == [("build", "second")]  # single row, updated
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_ascend.py -k inquire -v`
Expected: FAIL (route 404).

- [ ] **Step 3: Add the table init + the inquire route**

In `app.py`, immediately after `begin_ascend_recommend` (from Task 2), add:

```python
def _init_ascend_inquiries(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS ascend_inquiries "
        "(email TEXT, slug TEXT, goal TEXT, note TEXT, created_at TEXT, "
        "PRIMARY KEY(email, slug))")


@app.route("/begin/ascend/inquire", methods=["POST"])
def begin_ascend_inquire():
    """Record a consultation inquiry for a high-ticket rung (capture-and-notify).
    Member-gated; not a charge. Best-effort GHL tag + Glen email; never 500s."""
    if not ASCEND_PERSONALIZED_ENABLED:
        return jsonify({"ok": False}), 200
    try:
        body = request.get_json(silent=True) or {}
        slug = (body.get("slug") or "").strip()
        if slug not in begin_funnel.TIER_CATALOG:
            return jsonify({"ok": False, "error": "unknown tier"}), 400
        goal = (body.get("goal") or "").strip().lower()
        note = (body.get("note") or "").strip()[:2000]
        session_id = (request.cookies.get("amg_session") or "").strip()
        auth_user = get_authenticated_user(request)
        auth_email = ((auth_user["email"] if auth_user else "") or "").strip().lower()
        # Resolve the email from the one record: prefer the authenticated email,
        # else the journey state (a session-only ToS member still has an email).
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            state = begin_funnel.get_state(cx, session_id=session_id, email=auth_email)
        email = (auth_email or (state.get("email") or "")).strip().lower()
        if not is_member(session_id, email):
            return jsonify({"ok": False, "need_optin": True,
                            "error": "Please agree to our Terms to request a consultation."}), 403
        # Record the inquiry (one record per email+rung).
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            _init_ascend_inquiries(cx)
            cx.execute(
                "INSERT INTO ascend_inquiries (email, slug, goal, note, created_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(email, slug) DO UPDATE SET goal=excluded.goal, note=excluded.note, created_at=excluded.created_at",
                (email, slug, goal, note, datetime.utcnow().isoformat() + "Z"))
            cx.commit()
        tier = begin_funnel.TIER_CATALOG.get(slug) or {}
        # Best-effort GHL tag.
        try:
            ghl_onboard_contact(email, source_tag="ascend", extra_tags=[f"ascend:inquiry:{slug}"])
        except Exception as _ge:
            print(f"[ascend-inquire] ghl {_ge!r}", flush=True)
        # Best-effort Glen notification.
        try:
            to = os.environ.get("ASCEND_NOTIFY_EMAIL", "drglenswartwout@gmail.com")
            subject = f"Ascend inquiry: {tier.get('title', slug)} ({email})"
            note_line = f"\nNote: {note}" if note else ""
            _send_inquiry_email(to, subject,
                f"{email} requested a consultation.\nRung: {tier.get('title', slug)} ({tier.get('price', '')})\nGoal: {goal or '-'}{note_line}",
                reply_to=email)
        except Exception as _ee:
            print(f"[ascend-inquire] email {_ee!r}", flush=True)
        # One-record gate (idempotent, wrapped inside the helper).
        _record_entry_unlock("ascend", email)
        return jsonify({"ok": True})
    except Exception as e:
        app.logger.exception("ascend inquire failed")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_ascend.py -v`
Expected: PASS (the whole #5 suite).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_begin_ascend.py
git commit -m "feat: begin #5 ascend consultation inquiry capture + notify (member-gated)"
```

---

## Task 4: Front-end - goal chooser + personalized hero + inquiry capture (begin-ascend.html)

**Files:**
- Modify: `static/begin-ascend.html` (goal chooser + hero + inquiry form + confirmation when enabled; static ladder unchanged when not)
- Test: `tests/test_begin_ascend.py` (serve assertion)

**Interfaces:**
- Consumes: `GET /begin/ascend/recommend?goal=` (returns `{ok, enabled, goal, recommended, ladder, is_member}`) and `POST /begin/ascend/inquire` (`{slug, goal, note?}` -> `{ok}` / 403 `{need_optin}`).
- Produces: a personalized Ascend page when `enabled` is true; the existing static ladder otherwise.

- [ ] **Step 1: Write the failing serve test**

Add to `tests/test_begin_ascend.py`:

```python
def test_ascend_page_ships_personalized_wiring(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    # The page is static HTML; the personalization is client-side JS that calls the
    # endpoints. Assert the page ships the goal chooser markers + endpoint paths.
    html = app_module.app.test_client().get("/begin/ascend").get_data(as_text=True)
    assert "ascend/recommend" in html and "ascend/inquire" in html
    assert "data-goal" in html  # the goal chooser buttons
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_ascend.py::test_ascend_page_ships_personalized_wiring -v`
Expected: FAIL (`ascend/recommend` not in the static page).

- [ ] **Step 3: Add the personalization layer to `static/begin-ascend.html`**

Add a goal chooser, a hero mount point, an inquiry form, and a confirmation region to the page body (above or around the existing `.cards` ladder), plus a script that fetches `recommend` and wires `inquire`. Insert this block just before the closing `</body>` (and add a `<div id="ascend-personalized" hidden></div>` mount near the top of `.shell`). Use only `textContent`/`setAttribute` for dynamic data (no `innerHTML` of server data):

```html
<div id="ascend-goal" class="goal-chooser" hidden>
  <button type="button" data-goal="heal">Heal myself deeper</button>
  <button type="button" data-goal="learn">Learn the method</button>
  <button type="button" data-goal="build">Build my own practice</button>
</div>
<div id="ascend-hero" hidden></div>
<div id="ascend-confirm" hidden>Dr. Glen and Rae will reach out to book your consultation.</div>
<script>
(function () {
  var GOAL_KEY = "ascend_goal";
  function token() { return ""; }
  function getGoal() { try { return localStorage.getItem(GOAL_KEY) || "heal"; } catch (e) { return "heal"; } }
  function setGoal(g) { try { localStorage.setItem(GOAL_KEY, g); } catch (e) {} }

  function renderHero(rec, goal) {
    var hero = document.getElementById("ascend-hero");
    hero.textContent = "";
    if (!rec) { hero.hidden = true; return; }
    var title = document.createElement("div"); title.className = "hero-title"; title.textContent = rec.title || "";
    var price = document.createElement("div"); price.className = "hero-price"; price.textContent = (rec.price || "") + (rec.value ? " - " + rec.value : "");
    var incl = document.createElement("p"); incl.className = "hero-incl"; incl.textContent = rec.included || "";
    var btn = document.createElement("button"); btn.type = "button"; btn.textContent = rec.cta_label || "Book your consultation";
    btn.addEventListener("click", function () { inquire(rec.slug, goal, btn); });
    hero.appendChild(title); hero.appendChild(price); hero.appendChild(incl); hero.appendChild(btn);
    hero.hidden = false;
  }

  function load(goal) {
    fetch("/begin/ascend/recommend?goal=" + encodeURIComponent(goal))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d || !d.enabled) { return; }  // flag off -> leave the static ladder as-is
        document.getElementById("ascend-goal").hidden = false;
        renderHero(d.recommended, d.goal);
      }).catch(function () {});
  }

  function inquire(slug, goal, btn) {
    if (btn) { btn.disabled = true; }
    fetch("/begin/ascend/inquire", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slug: slug, goal: goal })
    }).then(function (r) { return r.json().then(function (b) { return { status: r.status, body: b }; }); })
      .then(function (res) {
        if (res.body && res.body.ok) {
          document.getElementById("ascend-confirm").hidden = false;
          if (btn) { btn.textContent = "Requested"; }
        } else if (res.status === 403 && res.body && res.body.need_optin) {
          window.location.href = "/begin";  // send them to agree to Terms first
        } else if (btn) { btn.disabled = false; alert((res.body && res.body.error) || "Could not send your request."); }
      }).catch(function () { if (btn) { btn.disabled = false; } });
  }

  var chooser = document.getElementById("ascend-goal");
  chooser.addEventListener("click", function (e) {
    var g = e.target && e.target.getAttribute("data-goal");
    if (!g) { return; }
    setGoal(g); load(g);
  });
  load(getGoal());
})();
</script>
```

(Place the `<div id="ascend-goal">` / `<div id="ascend-hero">` near the top of the content, above `.cards`, so the personalized hero sits above the full ladder. Keep the existing `.cards` ladder intact - it is the always-shown full ladder and the flag-off fallback.)

- [ ] **Step 4: Run the serve test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_ascend.py -v`
Expected: PASS (the serve assertion plus the whole #5 suite).

- [ ] **Step 5: Commit**

```bash
git add static/begin-ascend.html tests/test_begin_ascend.py
git commit -m "feat: begin #5 ascend page goal chooser + personalized hero + inquiry capture"
```

---

## Verification

- Per task: the named `pytest` target passes (doppler + venv).
- Full sweep after Task 4: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_ascend.py -v` all green; then a `-k "begin or journey"` regression run for no funnel regressions (the new `ascend` trigger is additive).
- Final Opus whole-branch review (focus: `recommend_ascend` pure/total and the track maps match the spec; flag-off fully dark on the page + both endpoints; `inquire` is member-gated BEFORE any record/notify, best-effort outward steps never fail the request, never a charge path; the `ascend_inquiries` upsert is idempotent per email+slug; XSS-safe front-end - textContent/setAttribute only; no emoji/em-dash; `"ascend"` added to VALID_TRIGGERS so `_record_entry_unlock` is accepted).
- Manual visual pass (live, after flag flip): the goal chooser switches the hero, the recommended rung shows above the full ladder, "Book your consultation" records + confirms, a non-member is routed to agree to Terms first, the static ladder still renders with the flag off.
- Ship via PR + merge to `main` (auto-deploys dark behind `ASCEND_PERSONALIZED_ENABLED`); gentle `/begin/ascend` + `/begin/ascend/recommend` probe per the warm-up rule (flag off -> static page + `{enabled:false}`); update memory.

## Build order
Task 1 (pure recommend + trigger) -> Task 2 (flag + recommend endpoint) -> Task 3 (inquire capture) -> Task 4 (front-end). Tasks 2-4 depend on Task 1; Task 4 depends on 2 and 3. Go-live = flip `ASCEND_PERSONALIZED_ENABLED=true` in Doppler `remedy-match/prd` (optionally set `ASCEND_NOTIFY_EMAIL`), after the manual visual pass.
