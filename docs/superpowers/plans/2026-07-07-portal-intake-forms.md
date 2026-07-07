# Portal Intake Forms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a client fill Dr. Glen's clinical intake in the portal (required before booking a Biofield Consult); store it in prod and bridge it into the local clinical tagger.

**Architecture:** A declarative form definition + pure logic module in deploy-chat (`dashboard/intake.py`), portal-token-gated routes, a precondition added to the existing consult booking routes, a console read panel, and a vault-side local puller that feeds submissions into the existing clinical tagger (`~/AI-Training/02 Skills/clinical_tagger.py`) and `e4l.db`.

**Tech Stack:** Python 3 stdlib + Flask (deploy-chat), sqlite3, pytest. No new dependencies. Frontend is vanilla JS injected into existing static HTML pages.

**Spec:** `docs/superpowers/specs/2026-07-07-portal-intake-forms-design.md`

## Global Constraints

- Additive only: new table, new routes, one new section in the consult card, one new console panel. Do not change existing booking/consult byte-paths beyond the added intake precondition.
- Copy rules (all user-facing + console text): no em dashes, no ALL-CAPS words, no "Hook:" labels.
- Auth: portal routes gate via the existing `_evox_ident(cx, token)` (returns an `Identity` with `.email`, or `None` → 404 `not_found`). Console routes gate via `_portal_console_ok()`. Token/console check ALWAYS precedes body validation.
- All DB writes run inside `with _db_lock, sqlite3.connect(LOG_DB) as cx:`; reads may skip the lock. Every route calls the lazy `intake.init_intake_table(cx)` before touching the table.
- Tasks 1-5 are in the deploy-chat worktree (`/tmp/wt-deploy-chat-d5d50811`, branch `sess/d5d50811-intake`). **Task 6 is in a DIFFERENT repo — the vault at `~/AI-Training`** (the tagger + `e4l.db` are local-only). Commit Task 6 in the vault, not the worktree.
- `LOG_DB`, `_db_lock`, `_evox_ident`, `_portal_console_ok`, `PUBLIC_BASE_URL` are existing module-level names in `app.py`.

---

### Task 1: `dashboard/intake.py` — form definition + pure logic

**Files:**
- Create: `dashboard/intake.py`
- Test: `tests/test_intake.py`

**Interfaces:**
- Produces:
  - `INTAKE_FORM: dict` — `{"version": "2026-07-07", "sections": [{"id","title","fields":[field,...]}, ...]}`. A `field` is `{"id","type","label", "help"?, "required"?, "options"?, "columns"?, "maps_to"?}`.
  - `init_intake_table(cx) -> None` — lazy `CREATE TABLE IF NOT EXISTS intake_responses`.
  - `validate_response(answers: dict) -> list[str]` — returns the ids of required-but-missing or invalid fields (empty list = valid).
  - `is_submitted(cx, email: str) -> bool`
  - `save_draft(cx, email: str, answers: dict, now: str) -> None` — upsert with `status='draft'`.
  - `submit(cx, email: str, answers: dict, now: str) -> None` — upsert with `status='submitted'`, stamps `submitted_at`. Caller guarantees validated + not already submitted.
  - `get_response(cx, email: str) -> dict | None` — row as dict with `answers` parsed from JSON.
  - `list_submitted(cx) -> list[dict]` — all submitted rows as dicts (email, form_version, submitted_at, answers).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_intake.py
import sqlite3
from dashboard import intake


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    intake.init_intake_table(cx)
    return cx


def test_form_structure_integrity():
    form = intake.INTAKE_FORM
    assert form["version"]
    ids = []
    dim_fields = []
    for sec in form["sections"]:
        assert sec["id"] and sec["title"]
        for f in sec["fields"]:
            assert f["id"] and f["type"]
            ids.append(f["id"])
            if f.get("maps_to"):
                dim_fields.append(f["maps_to"])
            if f["type"] == "scale":
                assert f["options"] and all("value" in o and "label" in o for o in f["options"])
            if f["type"] == "table":
                assert f["columns"] and all("id" in c and "type" in c for c in f["columns"])
    assert len(ids) == len(set(ids)), "field ids must be unique"
    assert sorted(dim_fields) == ["commitment", "penetration", "response", "terrain", "tissue_layer"]


def test_validate_missing_required():
    errors = intake.validate_response({})
    for req in ("first_name", "last_name", "email", "dob", "terrain", "terms"):
        assert req in errors


def test_validate_scale_out_of_range():
    errors = intake.validate_response({"terrain": 9})
    assert "terrain" in errors


def test_validate_consent_unsigned():
    errors = intake.validate_response({"terms": {"agreed": False, "signature": "", "date": ""}})
    assert "terms" in errors


def test_validate_valid_minimal():
    answers = {
        "first_name": "Steven", "last_name": "Fox", "email": "s@x.com", "dob": "1960-06-17",
        "terrain": 1, "penetration": 5, "tissue_layer": 3, "response": 3, "commitment": 8,
        "terms": {"agreed": True, "signature": "Steven Fox", "date": "2026-07-02"},
    }
    assert intake.validate_response(answers) == []


def test_draft_then_submit_transitions_status():
    cx = _cx()
    intake.save_draft(cx, "s@x.com", {"first_name": "Steven"}, "2026-07-07T00:00:00")
    assert intake.is_submitted(cx, "s@x.com") is False
    assert intake.get_response(cx, "s@x.com")["status"] == "draft"
    intake.submit(cx, "s@x.com", {"first_name": "Steven"}, "2026-07-07T01:00:00")
    assert intake.is_submitted(cx, "s@x.com") is True
    row = intake.get_response(cx, "s@x.com")
    assert row["status"] == "submitted" and row["submitted_at"] == "2026-07-07T01:00:00"


def test_list_submitted_only_returns_submitted():
    cx = _cx()
    intake.save_draft(cx, "draft@x.com", {"a": 1}, "2026-07-07T00:00:00")
    intake.submit(cx, "done@x.com", {"a": 2}, "2026-07-07T01:00:00")
    rows = intake.list_submitted(cx)
    assert [r["email"] for r in rows] == ["done@x.com"]
    assert rows[0]["answers"] == {"a": 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && env DATA_DIR=/tmp/intake-t python -m pytest tests/test_intake.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard.intake'` (or attribute errors).

- [ ] **Step 3: Write the module**

```python
# dashboard/intake.py
"""Client clinical intake: a declarative form (brought home from Practice Better)
plus pure store logic. No Flask, no network. The form definition is the single
source of truth for the questions; the local tagger consumes the `maps_to` hints.

Response shape: answers is a dict field_id -> value. Scalars for text/number/
scale/single_choice; a list of row-dicts for `table` fields; `terms` is
{"agreed": bool, "signature": str, "date": str}."""
import json

# --- scale option builders (labels are Glen's exact PB wording) ---
def _scale(pairs):
    return [{"value": v, "label": l} for v, l in pairs]

_TERRAIN = _scale([
    (1, "Cancer, Degeneration, Viral or Low Energy"),
    (2, "Rapid Aging, Bacterial, or Parasitic"),
    (3, "Fungal, Deposition, Slow Metabolism, or Low Body Temperature"),
    (4, "Allergy or Toxicity"),
    (5, "Stress or Hormonal Imbalance"),
])
_PENETRATION = _scale([
    (1, "Genetic or epigenetic expression"),
    (2, "Cell metabolism or mitochondrial dysfunction"),
    (3, "Connective tissue, immunity, autonomic or other nerve challenges"),
    (4, "Circulation, lymph drainage issues"),
    (5, "Poor digestion, dysbiosis, or other gut concerns"),
])
_TISSUE = _scale([
    (1, "Urogenital or Muscle"),
    (2, "Connective Tissue, Immune, or Cardiovascular"),
    (3, "Digestive or Respiratory"),
    (4, "Neuroendocrine"),
    (5, "Skin"),
])
_RESPONSE = _scale([
    (1, "No change"),
    (2, "Feel worse before better"),
    (3, "Mixed: some symptoms worse, but others better"),
    (4, "Some gradual improvement"),
    (5, "Rapid improvement"),
])
_COMMITMENT = _scale([(n, str(n)) for n in range(1, 11)])

INTAKE_FORM = {
    "version": "2026-07-07",
    "sections": [
        {"id": "personal", "title": "Personal Information", "fields": [
            {"id": "first_name", "type": "text", "label": "Legal first name", "required": True},
            {"id": "last_name", "type": "text", "label": "Last name", "required": True},
            {"id": "street", "type": "text", "label": "Street"},
            {"id": "unit", "type": "text", "label": "Unit"},
            {"id": "city", "type": "text", "label": "City"},
            {"id": "state", "type": "text", "label": "State"},
            {"id": "postal_code", "type": "text", "label": "Postal code"},
            {"id": "country", "type": "text", "label": "Country"},
            {"id": "email", "type": "email", "label": "Email address", "required": True},
            {"id": "home_phone", "type": "tel", "label": "Home phone"},
            {"id": "mobile_phone", "type": "tel", "label": "Mobile phone"},
            {"id": "dob", "type": "date", "label": "Date of birth", "required": True},
            {"id": "relationship_status", "type": "single_choice", "label": "Relationship status",
             "options": ["Single", "Partnered", "Married", "Divorced", "Widowed", "Prefer not to say"]},
            {"id": "gender", "type": "single_choice", "label": "Gender",
             "options": ["Woman/Girl", "Man/Boy", "Nonbinary", "Prefer not to say"]},
            {"id": "occupation", "type": "text", "label": "Occupation"},
            {"id": "hours_per_week", "type": "number", "label": "Hours per week"},
            {"id": "referred_by", "type": "text", "label": "Referred by"},
            {"id": "favorite_color", "type": "text", "label": "Describe your favorite color"},
        ]},
        {"id": "goals", "title": "Top Health Goals", "fields": [
            {"id": "health_concerns", "type": "table",
             "label": "List your current health concerns in order of importance",
             "help": "Rate how important each concern is to you from 1 to 10.",
             "columns": [
                 {"id": "concern", "type": "text", "label": "Health concern"},
                 {"id": "rating", "type": "number", "label": "Rating (1-10)"},
                 {"id": "years_since_onset", "type": "number", "label": "Years since onset"},
             ]},
        ]},
        {"id": "dimensions", "title": "Key Dimensions of the Clinical Theory of Everything",
         "fields": [
            {"id": "terrain", "type": "scale", "maps_to": "terrain", "required": True,
             "label": "Dominant Terrain",
             "help": "Select the lowest number that applies to current issues.",
             "options": _TERRAIN},
            {"id": "penetration", "type": "scale", "maps_to": "penetration", "required": True,
             "label": "Penetration of the Body Sanctuary",
             "help": "Select the lowest number that applies to current issues.",
             "options": _PENETRATION},
            {"id": "tissue_layer", "type": "scale", "maps_to": "tissue_layer", "required": True,
             "label": "Dominant Embryological Tissue Layer",
             "help": "Select the lowest number that applies to your current issues.",
             "options": _TISSUE},
            {"id": "response", "type": "scale", "maps_to": "response", "required": True,
             "label": "Dominant Healing Response",
             "help": "Your most typical response to well-selected natural therapies.",
             "options": _RESPONSE},
            {"id": "commitment", "type": "scale", "maps_to": "commitment", "required": True,
             "label": "Level of commitment to improving your health",
             "help": "1 is lowest, 10 is highest.", "options": _COMMITMENT},
            {"id": "obstacles", "type": "textarea",
             "label": "Is there anything that will get in the way of following a plan?"},
            {"id": "budget_monthly", "type": "number", "label": "Current budget",
             "help": "Estimated USD per month available to invest in better health."},
        ]},
        {"id": "history", "title": "Personal Health History", "fields": [
            {"id": "sleep", "type": "textarea",
             "label": "Do you have trouble falling asleep, staying asleep, or wake frequently?"},
            {"id": "dental", "type": "textarea", "label": "Dental issues: any amalgams or root canals?"},
            {"id": "vaccinations", "type": "textarea",
             "label": "Vaccinations: any COVID or other recent vaccinations?"},
            {"id": "supplements", "type": "table", "label": "Supplements you take now",
             "help": "Include vitamins, herbs, minerals. Rate how certain you are each is needed, 1 to 10.",
             "columns": [
                 {"id": "brand", "type": "text", "label": "Brand name"},
                 {"id": "name", "type": "text", "label": "Supplement name"},
                 {"id": "reason", "type": "text", "label": "Reason"},
                 {"id": "need", "type": "number", "label": "Need (1-10)"},
             ]},
            {"id": "diagnoses", "type": "table", "label": "Medical diagnoses", "columns": [
                 {"id": "diagnosis", "type": "text", "label": "Diagnosis"},
                 {"id": "current", "type": "single_choice", "label": "Status", "options": ["Current", "Past"]},
                 {"id": "age_onset", "type": "number", "label": "Age at onset"},
            ]},
            {"id": "medications", "type": "table", "label": "Medications you are currently taking",
             "columns": [
                 {"id": "medication", "type": "text", "label": "Medication"},
                 {"id": "reason", "type": "text", "label": "Reason"},
             ]},
            {"id": "surgeries", "type": "table", "label": "Past hospitalizations or surgeries",
             "columns": [
                 {"id": "procedure", "type": "text", "label": "Hospitalization or surgery"},
                 {"id": "reason", "type": "text", "label": "Reason"},
                 {"id": "age", "type": "number", "label": "Age"},
             ]},
            {"id": "allergies", "type": "table",
             "label": "Food or environmental allergies or sensitivities", "columns": [
                 {"id": "sensitivity", "type": "text", "label": "Sensitivity"},
                 {"id": "reaction", "type": "text", "label": "Reaction"},
            ]},
            {"id": "portrait", "type": "textarea",
             "label": "Portrait photo",
             "help": "Link to a photo for our clinical database, or note that one was sent."},
        ]},
        {"id": "consent", "title": "Consent", "fields": [
            {"id": "terms", "type": "consent", "required": True,
             "label": "I agree to the terms of service for Wellness Services at "
                      "remedymatch.com/info/terms-and-conditions."},
        ]},
    ],
}

# --- flat field index for validation ---
def _fields():
    for sec in INTAKE_FORM["sections"]:
        for f in sec["fields"]:
            yield f


def validate_response(answers):
    """Return the ids of required-but-missing or invalid fields (empty = valid).
    Tables are optional in v1 (a client may legitimately have none)."""
    errors = []
    for f in _fields():
        fid, ftype, req = f["id"], f["type"], f.get("required", False)
        val = answers.get(fid)
        if ftype == "scale":
            allowed = {o["value"] for o in f["options"]}
            if val is None:
                if req:
                    errors.append(fid)
            elif val not in allowed:
                errors.append(fid)
        elif ftype == "consent":
            ok = isinstance(val, dict) and val.get("agreed") is True and str(val.get("signature") or "").strip()
            if req and not ok:
                errors.append(fid)
        elif ftype == "table":
            continue  # optional in v1
        else:
            if req and not str(val or "").strip():
                errors.append(fid)
    return errors


def init_intake_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS intake_responses ("
        " email TEXT PRIMARY KEY,"
        " form_version TEXT NOT NULL,"
        " status TEXT NOT NULL,"          # 'draft' | 'submitted'
        " answers_json TEXT NOT NULL,"
        " created_at TEXT NOT NULL,"
        " submitted_at TEXT)")


def _upsert(cx, email, answers, status, now, submitted_at):
    email = (email or "").strip().lower()
    cx.execute(
        "INSERT INTO intake_responses (email, form_version, status, answers_json, created_at, submitted_at)"
        " VALUES (?,?,?,?,?,?)"
        " ON CONFLICT(email) DO UPDATE SET"
        "   form_version=excluded.form_version, status=excluded.status,"
        "   answers_json=excluded.answers_json,"
        "   submitted_at=COALESCE(excluded.submitted_at, intake_responses.submitted_at)",
        (email, INTAKE_FORM["version"], status, json.dumps(answers), now, submitted_at))
    cx.commit()


def save_draft(cx, email, answers, now):
    _upsert(cx, email, answers, "draft", now, None)


def submit(cx, email, answers, now):
    _upsert(cx, email, answers, "submitted", now, now)


def is_submitted(cx, email):
    row = cx.execute("SELECT status FROM intake_responses WHERE email=?",
                     ((email or "").strip().lower(),)).fetchone()
    return bool(row) and row[0] == "submitted"


def _row_to_dict(row):
    d = dict(row)
    d["answers"] = json.loads(d.pop("answers_json"))
    return d


def get_response(cx, email):
    row = cx.execute("SELECT * FROM intake_responses WHERE email=?",
                     ((email or "").strip().lower(),)).fetchone()
    return _row_to_dict(row) if row else None


def list_submitted(cx):
    rows = cx.execute(
        "SELECT * FROM intake_responses WHERE status='submitted' ORDER BY submitted_at").fetchall()
    return [_row_to_dict(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && env DATA_DIR=/tmp/intake-t python -m pytest tests/test_intake.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-d5d50811
git add dashboard/intake.py tests/test_intake.py
git commit -m "feat(intake): declarative intake form + pure store logic"
```

---

### Task 2: Intake routes

**Files:**
- Modify: `app.py` (add four routes near the consult routes, findable via `grep -n "def consult_state" app.py`)
- Test: `tests/test_intake_routes.py`

**Interfaces:**
- Consumes: `intake.INTAKE_FORM`, `init_intake_table`, `validate_response`, `is_submitted`, `save_draft`, `submit`, `get_response`; `_evox_ident`, `_db_lock`, `LOG_DB`, `_hst_now` (existing ISO-timestamp helper).
- Produces: `GET /api/intake/form`, `GET /api/intake/state`, `POST /api/intake/save-draft`, `POST /api/intake/submit`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_intake_routes.py
import os
os.environ.setdefault("DATA_DIR", "/tmp/intake-rt")
import json, importlib, sqlite3, pytest


@pytest.fixture
def client(monkeypatch):
    import app as appmod
    importlib.reload(appmod)
    from dashboard import intake

    class _Ident:  # stand-in for portal_identity.Identity
        def __init__(self, email): self.email = email

    # any token "good" resolves to a fixed member; "" or "bad" -> None
    def fake_ident(cx, token):
        return _Ident("member@x.com") if token == "good" else None
    monkeypatch.setattr(appmod, "_evox_ident", fake_ident)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_form_endpoint_returns_sections(client):
    r = client.get("/api/intake/form")
    assert r.status_code == 200
    assert r.get_json()["version"]
    assert any(s["id"] == "dimensions" for s in r.get_json()["sections"])


def test_state_bad_token_404(client):
    r = client.get("/api/intake/state?token=bad")
    assert r.status_code == 404 and r.get_json()["error"] == "not_found"


def test_token_gate_precedes_body_on_submit(client):
    r = client.post("/api/intake/submit?token=bad", json={"garbage": True})
    assert r.status_code == 404  # token wins over validation


def test_save_draft_then_state(client):
    client.post("/api/intake/save-draft?token=good", json={"answers": {"first_name": "Ann"}})
    r = client.get("/api/intake/state?token=good")
    body = r.get_json()
    assert body["status"] == "draft" and body["submitted"] is False
    assert body["answers"]["first_name"] == "Ann"


def test_submit_validation_error_lists_fields(client):
    r = client.post("/api/intake/submit?token=good", json={"answers": {}})
    assert r.status_code == 400
    assert "first_name" in r.get_json()["errors"]


def test_submit_success_then_double_submit_409(client):
    good = {"answers": {
        "first_name": "Ann", "last_name": "Lee", "email": "a@x.com", "dob": "1970-01-01",
        "terrain": 1, "penetration": 5, "tissue_layer": 3, "response": 3, "commitment": 8,
        "terms": {"agreed": True, "signature": "Ann Lee", "date": "2026-07-07"}}}
    assert client.post("/api/intake/submit?token=good", json=good).status_code == 200
    assert client.get("/api/intake/state?token=good").get_json()["submitted"] is True
    r2 = client.post("/api/intake/submit?token=good", json=good)
    assert r2.status_code == 409 and r2.get_json()["error"] == "already_submitted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && env DATA_DIR=/tmp/intake-rt python -m pytest tests/test_intake_routes.py -q`
Expected: FAIL (404 on `/api/intake/form` — routes not defined).

- [ ] **Step 3: Add the routes**

Insert immediately before `def consult_state():` in `app.py`:

```python
@app.route("/api/intake/form")
def intake_form():
    from dashboard import intake as _intake
    return jsonify(_intake.INTAKE_FORM)


@app.route("/api/intake/state")
def intake_state():
    from dashboard import intake as _intake
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _intake.init_intake_table(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        row = _intake.get_response(cx, ident.email)
    return jsonify({
        "submitted": bool(row) and row["status"] == "submitted",
        "status": row["status"] if row else "none",
        "answers": row["answers"] if row else {},
    })


@app.route("/api/intake/save-draft", methods=["POST"])
def intake_save_draft():
    from dashboard import intake as _intake
    answers = (request.get_json(force=True) or {}).get("answers") or {}
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _intake.init_intake_table(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        _intake.save_draft(cx, ident.email, answers, _hst_now().isoformat())
    return jsonify({"ok": True})


@app.route("/api/intake/submit", methods=["POST"])
def intake_submit():
    from dashboard import intake as _intake
    answers = (request.get_json(force=True) or {}).get("answers") or {}
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _intake.init_intake_table(cx)
        ident = _evox_ident(cx, request.args.get("token", ""))
        if ident is None:
            return jsonify({"error": "not_found"}), 404
        if _intake.is_submitted(cx, ident.email):
            return jsonify({"error": "already_submitted"}), 409
        errors = _intake.validate_response(answers)
        if errors:
            return jsonify({"error": "invalid", "errors": errors}), 400
        _intake.submit(cx, ident.email, answers, _hst_now().isoformat())
    return jsonify({"ok": True})
```

Note: if `_hst_now()` returns a `datetime`, `.isoformat()` gives the ISO string the store expects. Verify with `grep -n "def _hst_now" app.py`; if it already returns a string, drop `.isoformat()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && env DATA_DIR=/tmp/intake-rt python -m pytest tests/test_intake_routes.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-d5d50811
git add app.py tests/test_intake_routes.py
git commit -m "feat(intake): portal-token-gated form/state/save-draft/submit routes"
```

---

### Task 3: Consult gate — require intake before booking

**Files:**
- Modify: `app.py` — `consult_availability`, `consult_book`, `consult_state` (find via `grep -n "def consult_availability\|def consult_book\|def consult_state" app.py`)
- Test: `tests/test_consult_intake_gate.py`

**Interfaces:**
- Consumes: `intake.init_intake_table`, `intake.is_submitted`, `intake.submit`; existing `_consult.consult_is_ready`.
- Produces: a 409 `intake_required` from `consult_availability`/`consult_book` when intake not submitted; `intake_submitted: bool` added to the `consult_state` JSON.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_consult_intake_gate.py
import os
os.environ.setdefault("DATA_DIR", "/tmp/intake-gate")
import importlib, sqlite3, pytest


@pytest.fixture
def client(monkeypatch):
    import app as appmod
    importlib.reload(appmod)

    class _Ident:
        def __init__(self, email): self.email = email
    monkeypatch.setattr(appmod, "_evox_ident",
                        lambda cx, token: _Ident("m@x.com") if token == "good" else None)
    # consult is "ready" for our member so we exercise the intake gate, not the ready gate
    from dashboard import consult as _c
    monkeypatch.setattr(_c, "consult_is_ready", lambda cx, email: True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_availability_blocked_until_intake(client):
    r = client.get("/api/consult/availability?token=good")
    assert r.status_code == 409 and r.get_json()["error"] == "intake_required"


def test_book_blocked_until_intake(client):
    r = client.post("/api/consult/book?token=good", json={"start_ts": "2026-07-10T09:00:00"})
    assert r.status_code == 409 and r.get_json()["error"] == "intake_required"


def test_state_reports_intake_submitted_flag(client):
    r = client.get("/api/consult/state?token=good")
    assert r.get_json().get("intake_submitted") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && env DATA_DIR=/tmp/intake-gate python -m pytest tests/test_consult_intake_gate.py -q`
Expected: FAIL (availability returns 200/slots, no `intake_required`; state lacks the flag).

- [ ] **Step 3: Add the precondition**

In `consult_availability`, right after the `consult_is_ready` check block, add:

```python
        from dashboard import intake as _intake
        _intake.init_intake_table(cx)
        if not _intake.is_submitted(cx, ident.email):
            return jsonify({"error": "intake_required"}), 409
```

In `consult_book`, add the identical block right after its `consult_is_ready` check.

In `consult_state`, the return happens inside the `with` block:
```python
        stages = {"member": _is_paid_member(ident.email),
                  "test_paid": _consult.has_paid_purchase(cx, ident.email, _consult.CONSULT["test_slug"]),
                  "ready": ready}
        return jsonify({"ready": ready, "booked": booked, "stages": stages})
```
Replace those last two statements with:
```python
        stages = {"member": _is_paid_member(ident.email),
                  "test_paid": _consult.has_paid_purchase(cx, ident.email, _consult.CONSULT["test_slug"]),
                  "ready": ready}
        from dashboard import intake as _intake
        _intake.init_intake_table(cx)
        intake_done = _intake.is_submitted(cx, ident.email)
        return jsonify({"ready": ready, "booked": booked, "stages": stages,
                        "intake_submitted": intake_done})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && env DATA_DIR=/tmp/intake-gate python -m pytest tests/test_consult_intake_gate.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the existing consult suite to prove no regression**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && env DATA_DIR=/tmp/intake-gate python -m pytest tests/ -q -k "consult"`
Expected: PASS (pre-existing consult tests still green; if any assert a ready member can immediately fetch availability, update that fixture to submit intake first).

- [ ] **Step 6: Commit**

```bash
cd /tmp/wt-deploy-chat-d5d50811
git add app.py tests/test_consult_intake_gate.py
git commit -m "feat(intake): require submitted intake before consult booking"
```

---

### Task 4: Console read endpoint + panel

**Files:**
- Modify: `app.py` — add `GET /api/console/intake/<email>` and `GET /api/console/intake-submissions` near `_portal_console_ok` usages (find via `grep -n "console-biofield-portal\|_portal_console_ok" app.py | head`)
- Modify: `static/console-biofield-portal.html` — read-only intake panel
- Test: `tests/test_intake_console.py`

**Interfaces:**
- Consumes: `intake.get_response`, `intake.list_submitted`, `_portal_console_ok`.
- Produces: `GET /api/console/intake/<email>` → the one response; `GET /api/console/intake-submissions` → list (feeds Task 6's puller).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_intake_console.py
import os
os.environ.setdefault("DATA_DIR", "/tmp/intake-con")
import importlib, sqlite3, pytest


@pytest.fixture
def client(monkeypatch):
    import app as appmod
    importlib.reload(appmod)
    monkeypatch.setattr(appmod, "_portal_console_ok",
                        lambda: bool(__import__("flask").request.args.get("key") == "K"))
    # seed a submitted intake
    from dashboard import intake
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        intake.init_intake_table(cx)
        intake.submit(cx, "seed@x.com", {"first_name": "Seed"}, "2026-07-07T00:00:00")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_console_intake_requires_key(client):
    assert client.get("/api/console/intake/seed@x.com").status_code == 401


def test_console_intake_returns_response(client):
    r = client.get("/api/console/intake/seed@x.com?key=K")
    assert r.status_code == 200 and r.get_json()["answers"]["first_name"] == "Seed"


def test_console_submissions_list(client):
    r = client.get("/api/console/intake-submissions?key=K")
    assert any(x["email"] == "seed@x.com" for x in r.get_json()["submissions"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && env DATA_DIR=/tmp/intake-con python -m pytest tests/test_intake_console.py -q`
Expected: FAIL (404, routes undefined).

- [ ] **Step 3: Add the endpoints**

```python
@app.route("/api/console/intake/<path:email>")
def console_intake(email):
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import intake as _intake
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _intake.init_intake_table(cx)
        row = _intake.get_response(cx, email)
    if not row:
        return jsonify({"error": "not_found"}), 404
    return jsonify(row)


@app.route("/api/console/intake-submissions")
def console_intake_submissions():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import intake as _intake
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _intake.init_intake_table(cx)
        subs = _intake.list_submitted(cx)
    return jsonify({"submissions": subs})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && env DATA_DIR=/tmp/intake-con python -m pytest tests/test_intake_console.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Add the console panel**

In `static/console-biofield-portal.html`, next to the consult-ready control (find via `grep -n "consult-ready\|consultReady" static/console-biofield-portal.html`), add a small panel that, given an email already in scope on that page, fetches `/api/console/intake/<email>?key=<consoleKey>` and renders the answers read-only. Render every value with `textContent` (never innerHTML) to stay XSS-inert; show scale answers as "label (value)"; render table answers as simple rows. Use the page's existing console-key variable. Keep copy plain (no em dashes).

- [ ] **Step 6: Render-verify the panel**

Follow the render-verify approach ([[feedback_render_verify_not_just_inject]]): load `console-biofield-portal.html` in a headless browser against a seeded submitted intake and confirm the answers render with zero console errors.

- [ ] **Step 7: Commit**

```bash
cd /tmp/wt-deploy-chat-d5d50811
git add app.py static/console-biofield-portal.html tests/test_intake_console.py
git commit -m "feat(intake): console read endpoint + read-only intake panel"
```

---

### Task 5: Portal intake card (client-facing)

**Files:**
- Modify: `static/client-portal.html` — intake section inside the consult card

**Interfaces:**
- Consumes: `GET /api/intake/form`, `GET /api/intake/state`, `POST /api/intake/save-draft`, `POST /api/intake/submit`, and `GET /api/consult/state`'s new `intake_submitted` flag.

This task is UI wiring with no pytest unit; it is verified by render + a live click-through.

- [ ] **Step 1: Build the intake renderer**

In the consult card block of `static/client-portal.html` (find via `grep -n "consult" static/client-portal.html | head`), add JS that:
1. On consult-card load, fetches `/api/consult/state?token=`. If `intake_submitted` is false, render the intake form BEFORE the slot-picker and keep the slot-picker hidden.
2. Fetches `/api/intake/form` and generically renders each section and field by `type`: `text/email/tel/date` → `<input>`; `number` → `<input type=number>`; `textarea` → `<textarea>`; `single_choice` → `<select>`; `scale` → radio buttons showing `value` and `label`; `table` → a repeating row group with an "Add row" and per-row "Remove" button, one input per column; `consent` → a checkbox + a signature text input + a date input.
3. Prefills from `/api/intake/state`'s `answers`.
4. Autosaves: on any field change, debounce 1200ms then POST `/api/intake/save-draft?token=` with `{answers}`.
5. Submit button POSTs `/api/intake/submit?token=`. On 400, highlight the returned `errors` field ids and show "Please complete the highlighted fields." On 200, hide the form and reveal the slot-picker (re-run the consult-card init). On 409 `already_submitted`, treat as submitted.
6. All rendering uses `textContent` / `createElement` (no innerHTML with answer data).

- [ ] **Step 2: Render-verify the flow**

Follow [[feedback_render_verify_not_just_inject]]: in a headless browser with a valid portal token for a consult-ready member, confirm: intake renders, a table row adds/removes, autosave fires (network shows save-draft), submit with missing required fields highlights them, a full submit reveals the slot-picker, and there are zero console errors.

- [ ] **Step 3: Commit**

```bash
cd /tmp/wt-deploy-chat-d5d50811
git add static/client-portal.html
git commit -m "feat(intake): portal intake card gating the consult slot-picker"
```

---

### Task 6: Local tagging bridge (VAULT repo, not the worktree)

**Files (in `~/AI-Training`, the vault — commit here, NOT in the deploy-chat worktree):**
- Create: `02 Skills/intake_pull.py`
- Modify: `02 Skills/clinical-tags-sweep-run.sh` — add a pull step
- Test: `02 Skills/test_intake_pull.py`

**Interfaces:**
- Consumes (from `clinical_tagger.py`): `ensure_schema(cx)`, `apply_intake_dimensions(cx, client_id, **selections)`, `extract_freetext_tags(texts, source, use_llm=None)`, `diff_and_write(cx, client_id, computed)`. Reads prod via `GET /api/console/intake-submissions` with the console key (curl + real UA, per [[reference_ghl_api_ua_403]] the same UA discipline; here it is our own endpoint).
- Produces: an idempotent CLI that ingests each new submitted intake into `e4l.db`.

**Design of the puller:**
- Resolve email → e4l client_id via `SELECT client_id FROM e4l_clients WHERE lower(email)=?` (skip + log if no match).
- The five dimension answers (`terrain/penetration/tissue_layer/response/commitment`) → `apply_intake_dimensions(cx, cid, terrain=…, response=…, tissue_layer=…, penetration=…, commitment=…)` (only pass the keys present).
- Free-text: join `obstacles`, `sleep`, `dental`, `vaccinations`, and every `health_concerns[].concern` into one list of strings → `extract_freetext_tags(texts, source="pb-intake")` → `diff_and_write(cx, cid, computed)`.
- Idempotency: keep a local marker table `intake_ingested(email, submitted_at)` in `e4l.db`; skip a submission whose `(email, submitted_at)` is already recorded.

- [ ] **Step 1: Write the failing test**

```python
# 02 Skills/test_intake_pull.py
import importlib.util, os, sqlite3, sys

HERE = os.path.dirname(__file__)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_ingest_maps_dimensions_and_dedups(tmp_path, monkeypatch):
    pull = _load("intake_pull")
    db = tmp_path / "e4l.db"
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE e4l_clients (client_id INTEGER, name TEXT, email TEXT, archived_at TEXT)")
    cx.execute("INSERT INTO e4l_clients VALUES (42,'Ann','a@x.com',NULL)")
    cx.commit()
    sub = {"email": "a@x.com", "submitted_at": "2026-07-07T00:00:00", "form_version": "2026-07-07",
           "answers": {"terrain": 1, "response": 3, "tissue_layer": 3, "penetration": 5,
                       "commitment": 8, "obstacles": "cost", "health_concerns": [{"concern": "cataracts"}]}}
    n1 = pull.ingest_submissions(cx, [sub])
    n2 = pull.ingest_submissions(cx, [sub])  # idempotent second run
    assert n1 == 1 and n2 == 0
    tags = [r[0] for r in cx.execute(
        "SELECT tag FROM client_clinical_tags WHERE client_id=42 AND axis='dimension'").fetchall()]
    assert "terrain-phase:recharge" in tags  # terrain 1 -> recharge (per map_intake_dimensions)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/AI-Training/"02 Skills" && python3 -m pytest test_intake_pull.py -q`
Expected: FAIL (no module `intake_pull`).

- [ ] **Step 3: Write the puller**

```python
#!/usr/bin/env python3
# 02 Skills/intake_pull.py
"""Pull submitted portal intakes from prod and feed them into the clinical tagger
(local e4l.db). The portal (deploy-chat on Render) captures the intake; the tagger
+ ledger are local-only, so ingestion runs here on the local cadence.

  CONSOLE_KEY=... python3 intake_pull.py            # pull from prod + ingest
  python3 intake_pull.py --dry-run                  # show what would ingest
Reads GET {BASE}/api/console/intake-submissions?key=... (BASE default prod)."""
import argparse, importlib.util, json, os, sqlite3, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.expanduser("~/AI-Training/e4l.db")
BASE = os.environ.get("INTAKE_PROD_BASE", "https://glen-knowledge-chat.onrender.com")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def _load_tagger():
    spec = importlib.util.spec_from_file_location(
        "clinical_tagger", os.path.join(HERE, "clinical_tagger.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


ct = _load_tagger()


def _ensure_marker(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS intake_ingested (email TEXT, submitted_at TEXT,"
               " PRIMARY KEY (email, submitted_at))")


def _client_id_for(cx, email):
    row = cx.execute("SELECT client_id FROM e4l_clients WHERE lower(email)=? AND archived_at IS NULL",
                     ((email or "").strip().lower(),)).fetchone()
    return row[0] if row else None


def ingest_submissions(cx, submissions):
    """Ingest each new submission into e4l.db; returns the count actually ingested."""
    ct.ensure_schema(cx)
    _ensure_marker(cx)
    done = 0
    for sub in submissions:
        email, sat = sub.get("email"), sub.get("submitted_at")
        if cx.execute("SELECT 1 FROM intake_ingested WHERE email=? AND submitted_at=?",
                      (email, sat)).fetchone():
            continue
        cid = _client_id_for(cx, email)
        if cid is None:
            print(f"  skip (no e4l client): {email}")
            continue
        a = sub.get("answers") or {}
        dims = {k: a[k] for k in ("terrain", "response", "tissue_layer", "penetration", "commitment")
                if a.get(k) is not None}
        try:
            if dims:
                ct.apply_intake_dimensions(cx, cid, **dims)
            texts = [str(a.get(k) or "") for k in ("obstacles", "sleep", "dental", "vaccinations")]
            texts += [str(r.get("concern") or "") for r in (a.get("health_concerns") or [])]
            texts = [t for t in texts if t.strip()]
            if texts:
                computed = ct.extract_freetext_tags(texts, source="pb-intake")
                ct.diff_and_write(cx, cid, computed)
            cx.execute("INSERT OR IGNORE INTO intake_ingested (email, submitted_at) VALUES (?,?)",
                       (email, sat))
            cx.commit()
            done += 1
            print(f"  ingested: {email} -> client {cid}")
        except Exception as e:
            print(f"  ERROR {email}: {e}")
    return done


def _fetch_submissions(key):
    url = f"{BASE}/api/console/intake-submissions?key={key}"
    out = subprocess.run(["curl", "-sS", "-H", f"User-Agent: {UA}", url],
                         capture_output=True, text=True, timeout=30).stdout
    return (json.loads(out) or {}).get("submissions", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    key = os.environ.get("CONSOLE_KEY") or os.environ.get("CONSOLE_SECRET")
    if not key:
        print("ERROR: set CONSOLE_KEY (or CONSOLE_SECRET) to read the prod endpoint")
        return
    subs = _fetch_submissions(key)
    print(f"fetched {len(subs)} submitted intake(s) from prod")
    if a.dry_run:
        for s in subs:
            print(f"  would ingest: {s.get('email')} @ {s.get('submitted_at')}")
        return
    cx = sqlite3.connect(DB); cx.row_factory = sqlite3.Row
    try:
        n = ingest_submissions(cx, subs)
        print(f"ingested {n} new intake(s) into e4l.db")
    finally:
        cx.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/AI-Training/"02 Skills" && python3 -m pytest test_intake_pull.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Wire into the weekly sweep**

In `02 Skills/clinical-tags-sweep-run.sh`, after the GHL write-back step (step 2), add a step 3 that runs the puller when doppler is present (so `CONSOLE_SECRET` is in the env):

```bash
    # Pull new portal intake submissions into the tagger (prod -> local e4l.db).
    echo "--- intake pull $(date '+%H:%M:%S %Z') ---"
    doppler run -p remedy-match -c prd -- \
      /opt/homebrew/bin/python3 "$SKILLS/intake_pull.py"
```

- [ ] **Step 6: Commit (in the vault)**

```bash
cd ~/AI-Training
git add "02 Skills/intake_pull.py" "02 Skills/test_intake_pull.py" "02 Skills/clinical-tags-sweep-run.sh"
git commit -m "feat(intake): local puller bridges portal intakes into the clinical tagger"
```

Note: the vault's hourly auto-snapshot may also pick these up; an explicit commit here keeps the message meaningful.

---

## Finishing

After Task 6, run the full deploy-chat suite once from the worktree:

Run: `cd /tmp/wt-deploy-chat-d5d50811 && env DATA_DIR=/tmp/intake-fin python -m pytest tests/ -q -k "intake or consult"`
Expected: all intake + consult tests green.

Then use **superpowers:finishing-a-development-branch** to open the deploy-chat PR (branch `sess/d5d50811-intake`) and complete the work. The vault-side Task 6 lands via the vault's normal snapshot/commit flow, not the deploy-chat PR.

## Go-live notes

- No new prod env vars for Tasks 1-5. The puller reuses `CONSOLE_SECRET` (already in Doppler `remedy-match/prd`) to read the prod endpoint.
- After merge + deploy: flip a test client "consult ready," confirm the intake card blocks the slot-picker until submitted, submit it, confirm the slot-picker unlocks, then run `intake_pull.py --dry-run` to confirm the submission is visible before the first real ingest.
- Existing `glen-evox-reminders` cron is untouched; the intake pull rides the weekly `com.glen.clinical-tags-sweep` job.
