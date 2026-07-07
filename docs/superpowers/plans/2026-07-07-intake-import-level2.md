# Level 2 Intake Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import a client's existing Practice Better intake into the portal as a real structured record: paste the export on the local :8011 tool, LLM-parse it into the `INTAKE_FORM` shape, review/edit every field, and on approval write a submitted `intake_responses` row in prod (which the existing puller tags).

**Architecture:** A pure parse module (`dashboard/intake_parse.py`) turns pasted PB text into `INTAKE_FORM`-keyed answers; `dashboard/intake.py` gains `import_response`; `app.py` gains a console-gated `POST /api/console/intake-import`; the local `biofield_local_app.py` gets a paste/parse/review/import page. Tagging rides the existing `02 Skills/intake_pull.py` (no new tagger code).

**Tech Stack:** Python 3 stdlib + Flask, sqlite3, pytest, OpenAI JSON-mode (`openai_json`). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-07-intake-import-level2-design.md`

## Global Constraints

- Runs in the deploy-chat worktree `/tmp/wt-deploy-chat-d5d50811`, branch `sess/d5d50811-intake-import`.
- Additive only. Do not change the consult gate, the portal intake card, or the level-1 `mark_on_file`/`clear_intake`.
- Auth: prod routes gate on `_portal_console_ok()` (401), auth before body. Writes under `with _db_lock, sqlite3.connect(LOG_DB) as cx:`; call `intake.init_intake_table(cx)` first.
- Copy rules (any human-facing string): no em dashes, no ALL-CAPS words, no "Hook:" label.
- The imported answers MUST use `INTAKE_FORM` field ids so the existing puller (`intake_pull.py`) ingests them: the five dimension keys are `terrain`, `response`, `tissue_layer`, `penetration`, `commitment`; free-text the puller reads are `obstacles`, `sleep`, `dental`, `vaccinations`, and each `health_concerns[].concern`.
- Test env: `import app` needs real keys, so run app-importing tests via `doppler run -p remedy-match -c dev -- env DATA_DIR=/tmp/<dir> python3 -m pytest …` (mkdir the dir first). Pure-module tests (`intake_parse`, `intake`) run with plain `python3 -m pytest`.
- `_hst_now()` returns a naive `datetime`; call `.isoformat()`.

---

### Task 1: `dashboard/intake_parse.py` — pure PB-export parser

**Files:**
- Create: `dashboard/intake_parse.py`
- Test: `tests/test_intake_parse.py`

**Interfaces:**
- Consumes: `intake.INTAKE_FORM` (passed in as `form`, not imported, so tests stay pure).
- Produces:
  - `build_parse_prompt(form, pasted_text) -> str`
  - `coerce_parsed(form, raw) -> dict` — keep only known field ids; scale → int in the field's option values (else dropped); table → list of row dicts limited to known column ids; consent skipped; other types kept as scalar.
  - `parse(form, pasted_text, complete) -> dict` — `complete(system, user) -> str` returns model JSON; parse json-loads then `coerce_parsed`; any failure returns `{}`.
  - `SYSTEM: str` constant.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_intake_parse.py
from dashboard import intake, intake_parse

FORM = intake.INTAKE_FORM


def test_prompt_includes_sections_and_scale_options():
    p = intake_parse.build_parse_prompt(FORM, "PASTED CLIENT TEXT")
    assert "PASTED CLIENT TEXT" in p
    assert "terrain" in p and "health_concerns" in p
    # a scale option label appears so the model maps to the integer
    assert "Rapid Aging" in p


def test_coerce_drops_unknown_and_bad_scale():
    raw = {
        "first_name": "Steven", "bogus_field": "x",
        "terrain": "2", "penetration": 99,  # 2 valid (coerced to int), 99 out of range -> dropped
        "commitment": 8,
        "health_concerns": [{"concern": "cataracts", "rating": 10, "junk": "drop"},
                            "not-a-row"],
        "terms": {"agreed": True},  # consent skipped
    }
    out = intake_parse.coerce_parsed(FORM, raw)
    assert out["first_name"] == "Steven"
    assert "bogus_field" not in out
    assert out["terrain"] == 2 and "penetration" not in out
    assert out["commitment"] == 8
    assert out["health_concerns"] == [{"concern": "cataracts", "rating": 10}]
    assert "terms" not in out


def test_parse_uses_complete_and_coerces():
    canned = '{"first_name":"Ann","terrain":3,"nope":1}'
    out = intake_parse.parse(FORM, "text", lambda system, user: canned)
    assert out == {"first_name": "Ann", "terrain": 3}


def test_parse_bad_json_returns_empty():
    out = intake_parse.parse(FORM, "text", lambda s, u: "not json")
    assert out == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && python3 -m pytest tests/test_intake_parse.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.intake_parse`).

- [ ] **Step 3: Write the module**

```python
# dashboard/intake_parse.py
"""Parse a pasted Practice Better intake export into the declarative INTAKE_FORM
answer shape. Pure + LLM-agnostic: `parse` takes a `complete(system, user)->str`
callable so it is unit-testable without the network. Glen reviews/edits the
result before it is imported, so coercion is lenient-but-safe: unknown fields
and out-of-range scale values are dropped rather than guessed."""
import json

SYSTEM = ("You extract a clinical intake form into JSON. Return ONLY a JSON object "
          "keyed by the given field ids. For a scale field return the selected integer. "
          "For a table field return an array of row objects using the listed column ids. "
          "Use strings for text fields. Omit any field not present in the text. "
          "No commentary, JSON only.")


def _field_index(form):
    idx = {}
    for sec in form["sections"]:
        for f in sec["fields"]:
            idx[f["id"]] = f
    return idx


def build_parse_prompt(form, pasted_text):
    lines = ["Fields to extract (id: type; options for scales):"]
    for sec in form["sections"]:
        lines.append(f"[{sec['id']}] {sec['title']}")
        for f in sec["fields"]:
            if f["type"] == "scale":
                opts = "; ".join(f"{o['value']}={o['label']}" for o in f["options"])
                lines.append(f"  {f['id']} (scale): {opts}")
            elif f["type"] == "table":
                cols = ", ".join(c["id"] for c in f["columns"])
                lines.append(f"  {f['id']} (table rows of: {cols})")
            elif f["type"] == "consent":
                continue
            else:
                lines.append(f"  {f['id']} ({f['type']})")
    schema = "\n".join(lines)
    return f"{schema}\n\nIntake export text:\n\"\"\"\n{pasted_text}\n\"\"\"\n\nReturn the JSON now."


def coerce_parsed(form, raw):
    if not isinstance(raw, dict):
        return {}
    idx = _field_index(form)
    out = {}
    for fid, val in raw.items():
        f = idx.get(fid)
        if not f:
            continue
        t = f["type"]
        if t == "scale":
            try:
                iv = int(val)
            except (TypeError, ValueError):
                continue
            if iv in {o["value"] for o in f["options"]}:
                out[fid] = iv
        elif t == "table":
            if isinstance(val, list):
                cols = {c["id"] for c in f["columns"]}
                rows = [{k: r[k] for k in r if k in cols}
                        for r in val if isinstance(r, dict)]
                rows = [r for r in rows if r]
                if rows:
                    out[fid] = rows
        elif t == "consent":
            continue
        else:
            if val is not None and str(val).strip():
                out[fid] = val if isinstance(val, (str, int, float)) else str(val)
    return out


def parse(form, pasted_text, complete):
    try:
        raw = json.loads(complete(SYSTEM, build_parse_prompt(form, pasted_text)))
    except Exception:
        return {}
    return coerce_parsed(form, raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && python3 -m pytest tests/test_intake_parse.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-d5d50811
git add dashboard/intake_parse.py tests/test_intake_parse.py
git commit -m "feat(intake): pure PB-export parser (prompt + schema-coerce)"
```

---

### Task 2: `intake.import_response` — write a real imported record

**Files:**
- Modify: `dashboard/intake.py` (add after `mark_on_file`/`clear_intake`; find via `grep -n "def mark_on_file" dashboard/intake.py`)
- Test: `tests/test_intake.py`

**Interfaces:**
- Consumes: existing `_upsert(cx, email, answers, status, now, submitted_at)`, `get_response`, `is_submitted`, `INTAKE_FORM`.
- Produces: `import_response(cx, email, answers, now, source="practice-better") -> None` — upsert a `status='submitted'` row whose stored answers are `{**answers, "_imported": source}`. No-clobber guard: if a real submitted row already exists (submitted, and answers have neither `_imported` nor `_external`), return without change.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_intake.py
def test_import_response_writes_real_answers_with_marker():
    cx = _cx()
    intake.import_response(cx, "a@x.com",
                           {"first_name": "Ann", "terrain": 3}, "2026-07-07T00:00:00")
    assert intake.is_submitted(cx, "a@x.com") is True
    a = intake.get_response(cx, "a@x.com")["answers"]
    assert a["first_name"] == "Ann" and a["terrain"] == 3
    assert a["_imported"] == "practice-better"


def test_import_preserves_dimension_keys_for_puller():
    cx = _cx()
    dims = {"terrain": 1, "response": 3, "tissue_layer": 3, "penetration": 5, "commitment": 8}
    intake.import_response(cx, "b@x.com", dict(dims), "2026-07-07T00:00:00")
    a = intake.get_response(cx, "b@x.com")["answers"]
    for k, v in dims.items():
        assert a[k] == v  # keys the puller reads survive the import intact


def test_import_does_not_clobber_a_real_submission():
    cx = _cx()
    real = {"first_name": "Real", "terrain": 2,
            "terms": {"agreed": True, "signature": "Real", "date": "2026-07-07"}}
    intake.submit(cx, "c@x.com", real, "2026-07-07T00:00:00")
    intake.import_response(cx, "c@x.com", {"first_name": "Imported"}, "2026-07-07T01:00:00")
    a = intake.get_response(cx, "c@x.com")["answers"]
    assert a["first_name"] == "Real" and "_imported" not in a  # guard held


def test_import_may_overwrite_an_external_stub():
    cx = _cx()
    intake.mark_on_file(cx, "d@x.com", "2026-07-07T00:00:00")
    intake.import_response(cx, "d@x.com", {"first_name": "Now Real"}, "2026-07-07T01:00:00")
    a = intake.get_response(cx, "d@x.com")["answers"]
    assert a["first_name"] == "Now Real" and a["_imported"] == "practice-better"
```

(Uses the existing `_cx()` helper at the top of `tests/test_intake.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && python3 -m pytest tests/test_intake.py -q`
Expected: FAIL (`AttributeError: module 'dashboard.intake' has no attribute 'import_response'`).

- [ ] **Step 3: Implement**

Add to `dashboard/intake.py`:

```python
def import_response(cx, email, answers, now, source="practice-better"):
    """Write a client's out-of-band intake (parsed from a PB export) as a real
    submitted record. Guard: never overwrite a genuine portal submission (one
    with no _imported / _external marker). An _external level-1 stub is
    overwritable (it holds no real data)."""
    existing = get_response(cx, email)
    if existing and existing["status"] == "submitted":
        a = existing["answers"] or {}
        if not a.get("_imported") and not a.get("_external"):
            return
    payload = {**(answers or {}), "_imported": source}
    _upsert(cx, email, payload, "submitted", now, now)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && python3 -m pytest tests/test_intake.py -q`
Expected: PASS (all, including the 4 new).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-d5d50811
git add dashboard/intake.py tests/test_intake.py
git commit -m "feat(intake): import_response writes a real imported record (guarded)"
```

---

### Task 3: `POST /api/console/intake-import` endpoint

**Files:**
- Modify: `app.py` (add next to `intake_on_file`/`console_intake`; find via `grep -n "intake-on-file\|def console_intake" app.py`)
- Test: `tests/test_intake_console.py`

**Interfaces:**
- Consumes: `intake.init_intake_table`, `intake.import_response`, `intake.is_submitted`, `_portal_console_ok`, `_db_lock`, `LOG_DB`, `_hst_now`.
- Produces: `POST /api/console/intake-import {email, answers}` → `{ok:true}`; 401 unauth; 400 no email or non-dict `answers`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_intake_console.py (reuses the `client` fixture)
def test_intake_import_requires_key(client):
    r = client.post("/api/console/intake-import", json={"email": "x@x.com", "answers": {}})
    assert r.status_code == 401


def test_intake_import_bad_answers_400(client):
    r = client.post("/api/console/intake-import?key=K",
                    json={"email": "x@x.com", "answers": "not-a-dict"})
    assert r.status_code == 400


def test_intake_import_writes_and_gates(client):
    import sqlite3, app as appmod
    from dashboard import intake
    r = client.post("/api/console/intake-import?key=K",
                    json={"email": "imp@x.com", "answers": {"first_name": "Ann", "terrain": 3}})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        intake.init_intake_table(cx)
        assert intake.is_submitted(cx, "imp@x.com") is True
        assert intake.get_response(cx, "imp@x.com")["answers"]["_imported"] == "practice-better"
```

(The `client` fixture in this file already monkeypatches `_portal_console_ok` to accept `?key=K`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && mkdir -p /tmp/intake-imp && doppler run -p remedy-match -c dev -- env DATA_DIR=/tmp/intake-imp python3 -m pytest tests/test_intake_console.py -q`
Expected: FAIL (404 on the new route).

- [ ] **Step 3: Add the route**

```python
@app.route("/api/console/intake-import", methods=["POST"])
def console_intake_import():
    if not _portal_console_ok():
        return jsonify({"error": "unauthorized"}), 401
    from dashboard import intake as _intake
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    answers = body.get("answers")
    if not email:
        return jsonify({"error": "email required"}), 400
    if not isinstance(answers, dict):
        return jsonify({"error": "answers must be an object"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _intake.init_intake_table(cx)
        _intake.import_response(cx, email, answers, _hst_now().isoformat())
    return jsonify({"ok": True})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && doppler run -p remedy-match -c dev -- env DATA_DIR=/tmp/intake-imp python3 -m pytest tests/test_intake_console.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-d5d50811
git add app.py tests/test_intake_console.py
git commit -m "feat(intake): POST /api/console/intake-import endpoint"
```

---

### Task 4: Console panel "imported" note

**Files:**
- Modify: `static/console-biofield-portal.html` (the `renderIntakeAnswers` function added for the read panel; find via `grep -n "renderIntakeAnswers\|_external" static/console-biofield-portal.html`)

This task is a small UI addition; verified by static check + the controller's render-verify.

- [ ] **Step 1: Add the imported-note branch**

In `renderIntakeAnswers`, alongside the existing `_external` branch, add: when `answers._imported` is truthy, prepend a single line built with `createElement`/`textContent` reading "Imported from Practice Better" (do NOT early-return — still render the real fields below it, since an imported record has real answers). Keep all rendering `textContent`/`createElement` (no `innerHTML` with answer data). No em dashes.

- [ ] **Step 2: Static check**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && grep -n "_imported\|Imported from Practice Better" static/console-biofield-portal.html`
Expected: the new branch is present; confirm no `innerHTML` carries answer data in the added lines.

- [ ] **Step 3: Commit**

```bash
cd /tmp/wt-deploy-chat-d5d50811
git add static/console-biofield-portal.html
git commit -m "feat(intake): console panel notes an imported record"
```

---

### Task 5: Local :8011 paste / parse / review / import page

**Files:**
- Modify: `biofield_local_app.py` (add the routes + page; it already imports `dashboard.*` and has `openai_json(system, user)`)

This is local-tool UI wiring. The parse logic is covered by Task 1's tests (it calls `intake_parse.parse(form, text, openai_json)`); the page itself is render-verified by the controller/Glen after pulling and restarting the :8011 server.

- [ ] **Step 1: Add the parse route**

Add a route that accepts pasted text and returns parsed answers, wiring `openai_json` as the `complete` callable:

```python
    @app.route("/intake-import/parse", methods=["POST"])
    def intake_import_parse():
        from dashboard import intake as _intake, intake_parse as _ip
        body = request.get_json(force=True) or {}
        text = (body.get("text") or "").strip()
        if not text:
            return jsonify({"answers": {}, "error": "empty"})
        answers = _ip.parse(_intake.INTAKE_FORM, text, openai_json)
        return jsonify({"answers": answers})
```

- [ ] **Step 2: Add the page + import wiring**

Add a `GET /intake-import` page (console-gated like the other :8011 pages) that:
1. Has a textarea to paste the PB export, an email input (prefilled from the parsed `email` field when available), and a "Parse" button that POSTs to `/intake-import/parse`.
2. On parse, renders a full editable form from `intake.INTAKE_FORM` (imported at the top of the file: build inputs per field type — text/email/tel/date → input, number → number input, textarea → textarea, single_choice → select, scale → radios, table → add/remove rows, consent skipped), pre-filled with the returned answers. Build every field via `createElement`/`textContent` (XSS-inert).
3. An "Import to portal" button collects the edited answers (same shapes as the portal card: scale = integer, table = array of row objects) and POSTs `{email, answers}` to the prod endpoint `{PUBLIC_BASE_URL}/api/console/intake-import` with the `X-Console-Key` header (mirror the existing prod-POST plumbing: `urllib`-based server-side, or a browser `fetch` to the prod URL carrying the key — prefer routing it through a small local proxy route `POST /intake-import/submit` that forwards to prod with the console key, so the key is not exposed to the page JS).
4. Show success/failure status.

Add the local proxy route:

```python
    @app.route("/intake-import/submit", methods=["POST"])
    def intake_import_submit():
        import urllib.request, urllib.parse, json as _json
        body = request.get_json(force=True) or {}
        key = os.environ.get("CONSOLE_SECRET", "")
        base = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
        data = _json.dumps({"email": body.get("email"), "answers": body.get("answers")}).encode()
        req = urllib.request.Request(f"{base}/api/console/intake-import",
                                     data=data, method="POST",
                                     headers={"X-Console-Key": key, "Content-Type": "application/json"})
        try:
            resp = _json.load(urllib.request.urlopen(req, timeout=30))
            return jsonify(resp)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 502
```

Add a link to `/intake-import` from the workflow nav strip or the intake authoring page so it is reachable in the flow.

- [ ] **Step 3: Static check + syntax**

Run: `cd /tmp/wt-deploy-chat-d5d50811 && python3 -c "import ast; ast.parse(open('biofield_local_app.py').read()); print('py ok')"` and extract the page's inline `<script>` and run `node --check` on it.
Expected: both parse clean. Confirm no `innerHTML` carries parsed-answer data.

- [ ] **Step 4: Commit**

```bash
cd /tmp/wt-deploy-chat-d5d50811
git add biofield_local_app.py
git commit -m "feat(intake): local :8011 paste-parse-review-import page for PB intakes"
```

---

## Finishing

Run the full intake suite from the worktree:

Run: `cd /tmp/wt-deploy-chat-d5d50811 && doppler run -p remedy-match -c dev -- env DATA_DIR=/tmp/intake-fin2 python3 -m pytest tests/test_intake.py tests/test_intake_parse.py tests/test_intake_console.py -q` (mkdir the dir first).
Expected: all green.

Then use **superpowers:finishing-a-development-branch** to open the PR.

## Go-live

- Prod auto-deploys the endpoint + `intake.py`/`intake_parse.py` + console panel note on merge.
- The :8011 page ships in `biofield_local_app.py`: after merge, `cd ~/deploy-chat && git pull && launchctl kickstart -k gui/$(id -u)/com.glen.biofield-local-server`.
- Verify: open `/intake-import` on :8011, paste the Steven Fox export, confirm the parse pre-fills fields, edit one, import, confirm the console panel shows the record with the "Imported from Practice Better" note, then run `intake_pull.py` and confirm the dimension tags land in `e4l.db`.
