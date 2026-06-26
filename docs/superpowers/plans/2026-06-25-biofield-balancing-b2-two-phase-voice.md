# Biofield Balancing B2 — Two-Phase Voice Session — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the local intake's live voice session into Phase 1 (capture stresses by voice → master stress list) and Phase 2 (existing balancing), with a derived label-match that clears a stress when a chain row balances it.

**Architecture:** A new `interpret_stresses` parser (beside the existing causal-chain one) feeds a new `add_voice_stress` store function (merge by normalized label). `list_stresses` evolves to also read the live chain rows so a stress shows balanced when a chain row's head normalizes to its label — kept derived/recompute-on-read. The live-session widget gets a Phase 1 ⇄ Phase 2 toggle. Phase 2 (`interpret_transcript` / `author_interpret`) is untouched.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest. Local-only tool (`biofield_local_app.py`).

## Global Constraints

- Local-only; PHI stays on Glen's Mac. No feature flag, no prod deploy.
- Interpreters take an injected `complete(system, user) -> str`; tests stub it and run offline.
- Voice stresses: `source='voice'`, `balance='required'`, **stored with `code = _norm(label)`** (so the B1 `UNIQUE(test_id, source, code)` enforces per-label uniqueness instead of colliding on `code=''`). A synthetic voice code never appears in the scan coverage map, so voice stresses are never code-covered — they clear via label-match or manual only.
- Merge rule: a spoken stress whose normalized label matches ANY existing stress for the test (any source) is a merge → no insert.
- Auto-clear stays DERIVED: `balanced = manual_balanced OR code ∈ covered_codes OR (a chain row with a non-empty remedy has _norm(head) == _norm(stress.label))`. `balanced_by` precedence: covering remedy → label-match remedy → "manual" → "".
- `list_stresses` accepts a MIXED list (plain remedy-name strings OR `{"head","remedy"}` dicts) so B1's existing derive tests (which pass name lists) keep passing unchanged — no test migration needed.
- Run tests: `cd /tmp/wt-deploy-chat-82bd74c2 && ~/.venvs/deploy-chat311/bin/python -m pytest <path> -v` (pure-module tests; no Doppler). B1 + SP-A biofield tests must stay green.

---

### Task 1: `interpret_stresses` — Phase-1 stress-only parser

**Files:**
- Modify: `dashboard/biofield_interpret.py`
- Test: `tests/test_biofield_interpret_stresses.py`

**Interfaces:**
- Produces: `interpret_stresses(transcript, complete) -> list[str]` — distinct spoken stress labels (deduped case-insensitively, original casing kept, stripped); empty/blank transcript → `[]`. Uses the module's existing `_parse_json`. Also `build_stress_prompt(transcript) -> {"system","user"}` and `_STRESS_SYSTEM`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_interpret_stresses.py
from dashboard.biofield_interpret import interpret_stresses


def _c(payload):
    return lambda system, user: payload


def test_extracts_distinct_stress_labels():
    out = interpret_stresses(
        "the stress is liver congestion, also adrenal fatigue, liver congestion again",
        _c('{"stresses": ["Liver congestion", "Adrenal fatigue", "liver congestion"]}'))
    assert out == ["Liver congestion", "Adrenal fatigue"]   # deduped case-insensitively, order kept


def test_empty_transcript_returns_empty():
    assert interpret_stresses("   ", _c('{"stresses": ["x"]}')) == []


def test_handles_garbage_completion():
    assert interpret_stresses("something", _c("not json at all")) == []
```

- [ ] **Step 2: Run** → FAIL (`interpret_stresses` undefined).

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_interpret_stresses.py -v`

- [ ] **Step 3: Implement** — append to `dashboard/biofield_interpret.py`:

```python
_STRESS_SYSTEM = (
    "You read a clinician's spoken biofield-testing transcript (Dr. Glen Swartwout) and extract "
    "ONLY the distinct stress / issue / weakness names they name as present — NOT remedies, layers, "
    'or doses. Return STRICT JSON ONLY, no prose: {"stresses": [str, ...]}.\n'
    "- 'the stress is X', 'I'm seeing X', 'also X', 'there's X here' -> X is a stress.\n"
    "- If a remedy is named (e.g. 'balanced by Neuro Magnesium'), do NOT include the remedy; you MAY "
    "include the stress it balances if that stress is named.\n"
    "- Deduplicate. If nothing is parseable, return an empty list."
)


def build_stress_prompt(transcript):
    return {"system": _STRESS_SYSTEM, "user": "TRANSCRIPT:\n" + (transcript or "")}


def interpret_stresses(transcript, complete):
    """transcript + complete(system,user) -> [stress label, ...] (Phase 1, stresses only)."""
    if not (transcript or "").strip():
        return []
    p = build_stress_prompt(transcript)
    data = _parse_json(complete(p["system"], p["user"]))
    out, seen = [], set()
    for s in (data.get("stresses") or []):
        label = (s if isinstance(s, str) else (s.get("name") if isinstance(s, dict) else "")) or ""
        label = label.strip()
        k = label.lower()
        if label and k not in seen:
            seen.add(k)
            out.append(label)
    return out
```

- [ ] **Step 4: Run** → PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_interpret.py tests/test_biofield_interpret_stresses.py
git commit -m "feat(biofield-b2): interpret_stresses — Phase-1 stress-only parser"
```

---

### Task 2: `_norm` + `add_voice_stress`

**Files:**
- Modify: `dashboard/biofield_stress.py`
- Test: `tests/test_biofield_add_voice_stress.py`

**Interfaces:**
- Produces: `_norm(s) -> str` (lowercase, collapse internal whitespace, strip surrounding non-word chars); `add_voice_stress(cx, tid, label) -> bool` — inserts a `source='voice'`, `balance='required'`, `code=_norm(label)` stress; returns True if inserted, False if it merged into an existing stress (any source) with the same normalized label.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_add_voice_stress.py
import sqlite3
from dashboard.biofield_stress import add_voice_stress, init_stress_tables, seed_from_scan


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    return cx


def test_inserts_voice_stress_required(tmp_path):
    cx = _cx(tmp_path)
    assert add_voice_stress(cx, "a5", "Liver Congestion") is True
    row = cx.execute("SELECT source, balance, code, label FROM biofield_auth_stress "
                     "WHERE test_id=5").fetchone()
    assert row[0] == "voice" and row[1] == "required" and row[2] == "liver congestion"
    assert row[3] == "Liver Congestion"


def test_normalized_duplicate_merges(tmp_path):
    cx = _cx(tmp_path)
    assert add_voice_stress(cx, "a5", "Liver Congestion") is True
    assert add_voice_stress(cx, "a5", "  liver   congestion!! ") is False   # normalized dup
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_stress WHERE test_id=5").fetchone()[0] == 1


def test_two_distinct_voice_stresses_both_insert(tmp_path):
    cx = _cx(tmp_path)
    assert add_voice_stress(cx, "a5", "Liver Congestion") is True
    assert add_voice_stress(cx, "a5", "Adrenal Fatigue") is True   # both code='' would collide pre-fix
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_stress WHERE test_id=5").fetchone()[0] == 2


def test_merges_against_existing_scan_stress(tmp_path):
    cx = _cx(tmp_path)
    seed_from_scan(cx, "a5", [{"code": "ED1", "name": "Liver congestion"}], {})
    assert add_voice_stress(cx, "a5", "liver  congestion") is False   # matches scan label
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_stress WHERE test_id=5").fetchone()[0] == 1
```

- [ ] **Step 2: Run** → FAIL (`add_voice_stress` undefined).

- [ ] **Step 3: Implement** — in `dashboard/biofield_stress.py` add `import re` at the top (next to `import sqlite3`), then append:

```python
def _norm(s):
    """Normalize a stress label for dedup/label-match: lowercase, collapse internal
    whitespace, strip surrounding non-word characters."""
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return re.sub(r"^[^\w]+|[^\w]+$", "", s)


def add_voice_stress(cx, tid, label):
    """Add a voice-captured stress (required) unless its normalized label already
    exists for this test (any source) -> merge. Returns True if inserted."""
    init_stress_tables(cx)
    t = _num(tid)
    n = _norm(label)
    if not n:
        return False
    existing = cx.execute("SELECT label FROM biofield_auth_stress WHERE test_id=?", (t,)).fetchall()
    if any(_norm(r[0]) == n for r in existing):
        return False
    now = _now()
    cx.execute(
        "INSERT INTO biofield_auth_stress(test_id,code,label,source,balance,"
        "manual_balanced,created_at,updated_at) VALUES(?,?,?,'voice','required',0,?,?)",
        (t, n, (label or "").strip(), now, now))
    cx.commit()
    return True
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_add_voice_stress.py tests/test_biofield_stress_seed.py -v` → PASS (seed tests still green).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_stress.py tests/test_biofield_add_voice_stress.py
git commit -m "feat(biofield-b2): add_voice_stress with normalized-label merge"
```

---

### Task 3: `list_stresses` label-match (chain-row aware)

**Files:**
- Modify: `dashboard/biofield_stress.py` (`list_stresses`)
- Test: `tests/test_biofield_stress_labelmatch.py`

**Interfaces:**
- Consumes: `_norm` (Task 2), `covered_codes`/`_coverers` (B1).
- Produces: `list_stresses(cx, tid, chain_rows)` — `chain_rows` is a MIXED list of plain remedy-name strings OR `{"head","remedy"}` dicts. Code-coverage path uses the remedy names (from strings and dicts); label-match path uses dict rows with a non-empty remedy. Return shape unchanged (`{"active":[...],"balanced":[...]}` with `id,code,label,source,balance,balanced,balanced_by`). B1 callers passing name-string lists behave exactly as before.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_stress_labelmatch.py
import sqlite3
from dashboard.biofield_stress import add_voice_stress, init_stress_tables, list_stresses


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    add_voice_stress(cx, "a5", "Liver Congestion")
    return cx


def test_chain_row_head_balances_voice_stress(tmp_path):
    cx = _cx(tmp_path)
    rows = [{"head": "liver  congestion", "remedy": "Hepato Tonic"}]
    res = list_stresses(cx, "a5", rows)
    assert [s["label"] for s in res["balanced"]] == ["Liver Congestion"]
    assert res["balanced"][0]["balanced_by"] == "Hepato Tonic"
    assert res["active"] == []


def test_row_without_remedy_does_not_balance(tmp_path):
    cx = _cx(tmp_path)
    res = list_stresses(cx, "a5", [{"head": "liver congestion", "remedy": ""}])
    assert [s["label"] for s in res["active"]] == ["Liver Congestion"]
    assert res["balanced"] == []


def test_removing_row_reactivates(tmp_path):
    cx = _cx(tmp_path)
    assert list_stresses(cx, "a5", [{"head": "liver congestion", "remedy": "X"}])["active"] == []
    assert [s["label"] for s in list_stresses(cx, "a5", [])["active"]] == ["Liver Congestion"]


def test_string_list_still_works_backcompat(tmp_path):
    cx = _cx(tmp_path)
    # plain remedy-name strings: no head -> no label match -> voice stress stays active
    res = list_stresses(cx, "a5", ["Some Remedy"])
    assert [s["label"] for s in res["active"]] == ["Liver Congestion"]
```

- [ ] **Step 2: Run** → FAIL (dict rows raise / label match absent).

- [ ] **Step 3: Implement** — replace `list_stresses` in `dashboard/biofield_stress.py` with:

```python
def _chain_parts(chain_rows):
    """Split a mixed chain-rows list into (remedy_names, [(norm_head, remedy), ...]).
    Accepts plain remedy-name strings (no head) and {"head","remedy"} dicts."""
    names, heads = [], []
    for r in chain_rows or []:
        if isinstance(r, str):
            if r.strip():
                names.append(r)
        elif isinstance(r, dict):
            rem = (r.get("remedy") or "").strip()
            if rem:
                names.append(rem)
                h = _norm(r.get("head") or "")
                if h:
                    heads.append((h, rem))
    return names, heads


def list_stresses(cx, tid, chain_rows):
    init_stress_tables(cx)
    cx.row_factory = sqlite3.Row
    t = _num(tid)
    remedy_names, head_pairs = _chain_parts(chain_rows)
    covered = covered_codes(cx, tid, remedy_names)
    head_map = {}
    for h, rem in head_pairs:
        head_map.setdefault(h, rem)
    rows = cx.execute(
        "SELECT id, code, label, source, balance, manual_balanced "
        "FROM biofield_auth_stress WHERE test_id=? ORDER BY "
        "CASE balance WHEN 'required' THEN 0 ELSE 1 END, id", (t,)).fetchall()
    active, balanced = [], []
    for r in rows:
        is_cov = r["code"] in covered
        lbl_rem = head_map.get(_norm(r["label"]))
        is_bal = bool(r["manual_balanced"]) or is_cov or (lbl_rem is not None)
        if is_cov:
            cvs = _coverers(cx, tid, r["code"], remedy_names)
            by = cvs[0] if cvs else ""
        elif lbl_rem is not None:
            by = lbl_rem
        elif r["manual_balanced"]:
            by = "manual"
        else:
            by = ""
        item = {"id": r["id"], "code": r["code"], "label": r["label"],
                "source": r["source"], "balance": r["balance"],
                "balanced": is_bal, "balanced_by": by}
        (balanced if is_bal else active).append(item)
    return {"active": active, "balanced": balanced}
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_stress_labelmatch.py tests/test_biofield_stress_derive.py tests/test_biofield_stress_seed.py -v` → PASS (B1 derive tests unchanged + green via the string path).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_stress.py tests/test_biofield_stress_labelmatch.py
git commit -m "feat(biofield-b2): list_stresses label-match auto-clear (chain-row aware)"
```

---

### Task 4: `capture-stresses` route + chain-rows into `author_stresses`

**Files:**
- Modify: `biofield_local_app.py`
- Test: `tests/test_biofield_capture_stresses_routes.py`

**Interfaces:**
- Consumes: `interpret_stresses` (Task 1), `add_voice_stress` (Task 2), `list_stresses(chain_rows)` (Task 3); the app's injected `interpret_complete`, `get_notes`, `_report_for`, `db_path`.
- Produces: `POST /author/<test_id>/capture-stresses` → `{"added": n}` or `{"added": 0, "error": ...}`; `author_stresses` passes `chain_rows` (head+remedy dicts) to `list_stresses`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_capture_stresses_routes.py
import sqlite3
import pytest
from biofield_local_app import create_app


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}


def _app(db, stresses):
    # interpret_complete returns a fixed stresses payload; scan_lookup finds nothing (no seeding)
    import json as _j
    return create_app(db, scan_lookup=lambda e: _NONE,
                      interpret_complete=lambda s, u: _j.dumps({"stresses": stresses}))


def _new(client):
    return client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]


def test_capture_adds_voice_stresses(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, ["Liver Congestion", "Adrenal Fatigue"]).test_client()
    tid = _new(client)
    client.post(f"/author/{tid}/session", json={"transcript": "liver congestion, adrenal fatigue"})
    j = client.post(f"/author/{tid}/capture-stresses", json={}).get_json()
    assert j["added"] == 2
    s = client.get(f"/author/{tid}/stresses").get_json()["data"]
    labels = {x["label"] for x in s["active"] + s["balanced"]}
    assert labels == {"Liver Congestion", "Adrenal Fatigue"}


def test_capture_empty_transcript_errors(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, ["X"]).test_client()
    tid = _new(client)
    j = client.post(f"/author/{tid}/capture-stresses", json={}).get_json()
    assert j["added"] == 0 and "error" in j


def test_capture_then_chain_row_balances_by_label(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, ["Liver Congestion"]).test_client()
    tid = _new(client)
    client.post(f"/author/{tid}/session", json={"transcript": "liver congestion"})
    client.post(f"/author/{tid}/capture-stresses", json={})
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "Liver Congestion", "remedy": "Hepato Tonic"})
    s = client.get(f"/author/{tid}/stresses").get_json()["data"]
    assert [x["label"] for x in s["balanced"]] == ["Liver Congestion"]
    assert s["balanced"][0]["balanced_by"] == "Hepato Tonic"
```

- [ ] **Step 2: Run** → FAIL (route 404 / `author_stresses` ignores heads).

- [ ] **Step 3: Implement** — in `biofield_local_app.py`:

(a) Add the capture route near `author_interpret`:

```python
    @app.route("/author/<test_id>/capture-stresses", methods=["POST"])
    def author_capture_stresses(test_id):
        from dashboard.biofield_interpret import interpret_stresses
        from dashboard import biofield_stress as _st
        with sqlite3.connect(db_path) as cx:
            transcript = get_notes(cx, test_id)
            if not transcript.strip():
                return {"added": 0, "error": "no transcript yet -- record a session first"}
            try:
                labels = interpret_stresses(transcript, interpret_complete)
            except Exception as e:
                return {"added": 0, "error": str(e)[:200]}
            added = sum(1 for label in labels if _st.add_voice_stress(cx, test_id, label))
        return {"added": added}
```

(b) Update `author_stresses` to pass chain rows (head + remedy):

```python
    @app.route("/author/<test_id>/stresses")
    def author_stresses(test_id):
        from dashboard import biofield_stress as _st
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            chain_rows = [{"head": l.get("head"), "remedy": l.get("remedy")}
                          for l in (rep.get("layers") or [])]
            data = _st.list_stresses(cx, test_id, chain_rows)
        return {"data": data, "html": render_stress_panel(data)}
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_capture_stresses_routes.py tests/test_biofield_stress_routes.py -v` → PASS (B1 stress routes still green).

- [ ] **Step 5: Commit**

```bash
git add biofield_local_app.py tests/test_biofield_capture_stresses_routes.py
git commit -m "feat(biofield-b2): capture-stresses route + chain-rows into stresses route"
```

---

### Task 5: UI — Phase 1 ⇄ Phase 2 toggle + `captureStresses()`

**Files:**
- Modify: `dashboard/biofield_report_html.py`
- Test: `tests/test_biofield_phase_toggle.py`

**Interfaces:**
- Consumes: existing `__TID__`/`post()`/`rstat()`/`loadStress()`/`interpret()` JS conventions.
- Produces: the live-session widget renders a Phase 1 ⇄ Phase 2 toggle; a `captureStresses()` JS function POSTs to `/author/__TID__/capture-stresses` then calls `loadStress()`; Phase 2 keeps `interpret()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_phase_toggle.py
from dashboard.biofield_report_html import render_author_html


def _html():
    rep = {"test_id": "a7", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [], "schedule": []}
    return render_author_html(rep, [], "")


def test_phase_toggle_present():
    h = _html()
    assert "Capture stresses" in h and "Balance" in h
    assert "setPhase(" in h            # toggle handler
    assert "function captureStresses" in h


def test_capture_posts_to_route_and_reloads_panel():
    h = _html()
    assert "/author/a7/capture-stresses" in h
    assert "loadStress()" in h         # panel refresh after capture
```

- [ ] **Step 2: Run** → FAIL (no toggle / no `captureStresses`).

- [ ] **Step 3: Implement** — in `dashboard/biofield_report_html.py`:

(a) In `_AUTHOR_JS`, add (next to `interpret`):

```javascript
function setPhase(p){window._phase=p;
 document.getElementById('phaseCap').className=(p==1?'btn':'btn ghost');
 document.getElementById('phaseBal').className=(p==2?'btn':'btn ghost');
 document.getElementById('phaseAct').textContent=(p==1?'Capture stresses → list':'Interpret → fill fields')}
async function phaseRun(){if((window._phase||1)==1){captureStresses()}else{interpret()}}
async function captureStresses(){rstat('Capturing stresses from transcript\\u2026');
 var j=await post('/author/__TID__/capture-stresses',{});
 if(j.error){rstat('Capture: '+j.error);return}
 rstat('Added '+j.added+' stress(es).');loadStress()}
```

(b) In the live-session widget markup (the `session = (...)` block), replace the single "Interpret → fill fields" button with the toggle + a single phase-action button:

```python
    session = (
        "<h2>Live session (voice)</h2>"
        "<div class=btnrow style='margin-bottom:6px'>"
        "<button id=phaseCap class=btn onclick='setPhase(1)'>Phase 1 · Capture stresses</button>"
        "<button id=phaseBal class='btn ghost' onclick='setPhase(2)'>Phase 2 · Balance</button>"
        "</div>"
        # ...existing record button(s)...
        "<button id=phaseAct class=btn onclick=phaseRun()>Capture stresses &rarr; list</button>"
        # ...existing rstat status span + sessText textarea...
    )
```
Keep the existing record button, `rstat` status span, and `sessText` textarea in place; only swap the old `interpret()` button for the toggle + `phaseAct` button. Call `setPhase(1)` once on load (add `setPhase(1);` next to the existing `loadE4L(); loadStress();` calls) so Phase 1 is the default.

NOTE: the exact surrounding markup of the `session` block is in `render_author_html`; preserve every existing element (record control, interim span, transcript textarea, save-on-stop wiring) — this task only adds the toggle and renames the action button to `phaseRun()`.

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_phase_toggle.py tests/test_biofield_author_html.py tests/test_biofield_stress_panel.py -v` → PASS (existing author-html tests green).

- [ ] **Step 5: Run the B2 + adjacent suite + commit**

```bash
~/.venvs/deploy-chat311/bin/python -m pytest \
  tests/test_biofield_interpret_stresses.py tests/test_biofield_add_voice_stress.py \
  tests/test_biofield_stress_labelmatch.py tests/test_biofield_capture_stresses_routes.py \
  tests/test_biofield_phase_toggle.py tests/test_biofield_stress_seed.py \
  tests/test_biofield_stress_derive.py tests/test_biofield_stress_routes.py \
  tests/test_biofield_stress_panel.py tests/test_biofield_interpret.py \
  tests/test_biofield_author_html.py tests/test_biofield_e4l_routes.py -q
git add dashboard/biofield_report_html.py tests/test_biofield_phase_toggle.py
git commit -m "feat(biofield-b2): live-session Phase 1/2 toggle + captureStresses()"
```

---

## Self-Review

**Spec coverage:**
- Phase-1 interpreter → Task 1. ✓
- `add_voice_stress` merge-by-normalized-label (+ the `code=_norm(label)` fix for the UNIQUE) → Task 2. ✓
- Derived label-match auto-clear → Task 3. ✓
- capture-stresses route + chain-rows into stresses route → Task 4. ✓
- Phase toggle UI + `captureStresses` → Task 5. ✓
- Phase 2 unchanged (no task touches `interpret_transcript`/`author_interpret`). ✓
- B1 derive tests unchanged (string back-compat in Task 3). ✓ (Refinement vs spec, which suggested migrating them — back-compat is strictly less churn.)

**Placeholder scan:** No TBDs; every code step is complete. Task 5(b) explicitly preserves the existing session-block elements rather than re-listing markup not in scope.

**Type consistency:** `interpret_stresses -> list[str]` consumed by Task 4; `add_voice_stress(cx,tid,label)->bool` consumed by Task 4; `list_stresses(cx,tid,chain_rows)` (mixed strings/dicts) consumed by Task 4's `author_stresses` and the panel; `_norm` shared by Tasks 2 & 3. Consistent.

## Verification (manual, after all tasks)

```bash
cd ~/deploy-chat && doppler run -p remedy-match -c prd -- python3 biofield_local_app.py
```
Open a test, record a Phase-1 pass naming stresses → "Capture stresses → list" adds them to the Active panel (deduped against scan). Switch to Phase 2, speak balancing → chain rows fill and the matching stresses move to Balanced (by remedy). Delete a chain row → its stress returns to Active.
