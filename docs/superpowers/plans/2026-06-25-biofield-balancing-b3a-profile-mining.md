# Biofield Balancing B3a — Profile / Tag Mining — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mine a client's consolidated `/api/people` profile (tags, conditions, challenges, goals, terrain_concerns, body_systems, notes) into the intake master stress list (merged by normalized label) and weave that profile context into the generated narrative.

**Architecture:** A new injectable `fetch_profile(email)` (HTTP to the existing prod People hub, mirroring `e4l-reveal-push.py:fetch_history`) feeds a new pure `mine_profile_stresses(profile, extract)` that turns discrete fields into labels directly and free-text fields through B2's `interpret_stresses`. Labels land via a generalized `add_stress(..., source='tag')`. The narrative gains a back-compatible `profile=` param. A route + the existing seed hook + a button drive it.

**Tech Stack:** Python 3.11, Flask, sqlite3, urllib, pytest. Local-only tool (`biofield_local_app.py`).

## Global Constraints

- Local-only; PHI stays on Glen's Mac (profile fetched live, used in-process; only derived stress labels persist). No feature flag, no prod deploy.
- `fetch_profile`, the `extract` callable, and `complete`/`interpret_complete` are injected so tests run offline.
- Profile-derived stresses: `source='tag'`, `balance='required'`, stored with `code=_norm(label)`, merged by normalized label across ALL sources (no duplicate vs scan/voice/earlier tag).
- Profile mining is best-effort: any network/profile failure returns an error dict / is swallowed — it never blocks intake. It runs even when there is NO fresh scan.
- Narrative back-compat: with `profile=None` (and no scan) the prompt is byte-identical to today.
- Run tests: `cd /tmp/wt-deploy-chat-82bd74c2 && ~/.venvs/deploy-chat311/bin/python -m pytest <path> -v` (pure-module; no Doppler). B1/B2/SP-A biofield tests must stay green.

---

### Task 1: Generalize the merge-insert — `add_stress`

**Files:**
- Modify: `dashboard/biofield_stress.py` (`add_voice_stress`)
- Test: `tests/test_biofield_add_stress.py`

**Interfaces:**
- Produces: `add_stress(cx, tid, label, *, source='voice', balance='required') -> bool` — inserts a stress (`code=_norm(label)`) unless its normalized label already exists for the test in ANY source (merge → False). `add_voice_stress(cx, tid, label)` delegates: `return add_stress(cx, tid, label, source='voice', balance='required')`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_add_stress.py
import sqlite3
from dashboard.biofield_stress import add_stress, add_voice_stress, init_stress_tables, seed_from_scan


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_stress_tables(cx)
    return cx


def test_add_stress_tag_source(tmp_path):
    cx = _cx(tmp_path)
    assert add_stress(cx, "a5", "Adrenal Fatigue", source="tag") is True
    row = cx.execute("SELECT source, balance, code FROM biofield_auth_stress WHERE test_id=5").fetchone()
    assert row[0] == "tag" and row[1] == "required" and row[2] == "adrenal fatigue"


def test_add_stress_merges_cross_source(tmp_path):
    cx = _cx(tmp_path)
    add_voice_stress(cx, "a5", "Liver Congestion")              # voice
    assert add_stress(cx, "a5", "  liver congestion ", source="tag") is False   # merge
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_stress WHERE test_id=5").fetchone()[0] == 1


def test_add_voice_stress_still_voice(tmp_path):
    cx = _cx(tmp_path)
    assert add_voice_stress(cx, "a5", "Brain Fog") is True
    assert cx.execute("SELECT source FROM biofield_auth_stress WHERE test_id=5").fetchone()[0] == "voice"
```

- [ ] **Step 2: Run** → FAIL (`add_stress` undefined).

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_add_stress.py -v`

- [ ] **Step 3: Implement** — replace `add_voice_stress` in `dashboard/biofield_stress.py` with:

```python
def add_stress(cx, tid, label, *, source="voice", balance="required"):
    """Add a stress unless its normalized label already exists for this test (any
    source) -> merge. Stored with code=_norm(label) so UNIQUE(test_id,source,code)
    never collides. Returns True if inserted."""
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
        "manual_balanced,created_at,updated_at) VALUES(?,?,?,?,?,0,?,?)",
        (t, n, (label or "").strip(), source, balance, now, now))
    cx.commit()
    return True


def add_voice_stress(cx, tid, label):
    """Voice-captured stress (required). Thin wrapper over add_stress."""
    return add_stress(cx, tid, label, source="voice", balance="required")
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_add_stress.py tests/test_biofield_add_voice_stress.py -v` → PASS (B2 voice tests still green).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_stress.py tests/test_biofield_add_stress.py
git commit -m "feat(biofield-b3a): generalize add_stress; add_voice_stress delegates"
```

---

### Task 2: `mine_profile_stresses`

**Files:**
- Create: `dashboard/biofield_profile.py`
- Test: `tests/test_biofield_profile_mine.py`

**Interfaces:**
- Produces: `mine_profile_stresses(profile, extract) -> list[str]` — `profile` is a dict from `/api/people`; `extract(text) -> list[str]` is injected (the B2 parser bound to a completer). Discrete fields `tags, conditions, terrain_concerns, body_systems` (each a list OR a comma/semicolon string) → labels directly; free-text `challenges, goals, notes` concatenated → `extract(...)`. Combined, deduped case-insensitively (first casing kept). Empty/missing profile → `[]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_profile_mine.py
from dashboard.biofield_profile import mine_profile_stresses


def _extract(text):
    # stub: pretend the LLM pulled two labels from the free text when present
    return ["Chronic fatigue", "Poor sleep"] if text.strip() else []


def test_discrete_fields_list_and_string_forms():
    profile = {"tags": ["Inflammation", "Heavy metals"],
               "conditions": "Hashimoto's; Eczema",
               "terrain_concerns": "Acidic",
               "body_systems": ["Liver"]}
    out = mine_profile_stresses(profile, lambda t: [])
    assert set(out) == {"Inflammation", "Heavy metals", "Hashimoto's", "Eczema", "Acidic", "Liver"}


def test_free_text_goes_through_extract_and_dedupes():
    profile = {"tags": ["Inflammation"], "challenges": "always tired", "goals": "sleep better"}
    out = mine_profile_stresses(profile, _extract)
    assert "Inflammation" in out and "Chronic fatigue" in out and "Poor sleep" in out


def test_dedupe_case_insensitive():
    profile = {"tags": ["Inflammation", "inflammation"], "conditions": "INFLAMMATION"}
    out = mine_profile_stresses(profile, lambda t: [])
    assert out == ["Inflammation"]


def test_empty_profile():
    assert mine_profile_stresses({}, _extract) == []
    assert mine_profile_stresses(None, _extract) == []
```

- [ ] **Step 2: Run** → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# dashboard/biofield_profile.py
"""Mine a client's consolidated /api/people profile into discrete stress labels
for the local Biofield Intake balancing loop (B3a). Pure: the free-text extractor
is injected so this is testable offline."""

_DISCRETE = ("tags", "conditions", "terrain_concerns", "body_systems")
_FREETEXT = ("challenges", "goals", "notes")


def _items(v):
    """A profile field may be a list or a comma/semicolon-separated string."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        parts = v
    else:
        parts = str(v).replace(";", ",").split(",")
    return [p.strip() for p in (str(x).strip() for x in parts) if p.strip()]


def mine_profile_stresses(profile, extract):
    """profile dict + extract(text)->[labels] -> deduped stress labels."""
    profile = profile or {}
    labels = []
    for field in _DISCRETE:
        labels.extend(_items(profile.get(field)))
    free = "\n".join(str(profile.get(f) or "").strip() for f in _FREETEXT if profile.get(f))
    if free.strip():
        labels.extend(extract(free) or [])
    out, seen = [], set()
    for label in labels:
        label = (label or "").strip()
        k = label.lower()
        if label and k not in seen:
            seen.add(k)
            out.append(label)
    return out
```

- [ ] **Step 4: Run** → PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_profile.py tests/test_biofield_profile_mine.py
git commit -m "feat(biofield-b3a): mine_profile_stresses (discrete + free-text)"
```

---

### Task 3: `fetch_profile` + `mine-profile` route + seed-hook wiring

**Files:**
- Modify: `biofield_local_app.py`
- Test: `tests/test_biofield_mine_profile_routes.py`

**Interfaces:**
- Consumes: `mine_profile_stresses` (Task 2), `add_stress` (Task 1), `interpret_stresses` (B2), `_report_for`, `interpret_complete`, `_seed_stresses` (B1).
- Produces: `create_app(..., fetch_profile=None)` (new kwarg, default real HTTP impl); `POST /author/<test_id>/mine-profile` → `{"added": n}` or `{"added": 0, "error": ...}`; `_mine_profile(cx, test_id)` helper (best-effort) also called from `_seed_stresses`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_mine_profile_routes.py
import sqlite3
import pytest
from biofield_local_app import create_app


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}
_PROFILE = {"email": "j@x.com", "tags": ["Inflammation"], "conditions": "Eczema",
            "challenges": "always tired"}


def _app(db, profile, stresses):
    import json as _j
    return create_app(db, scan_lookup=lambda e: _NONE,
                      fetch_profile=lambda e: profile if e == "j@x.com" else {},
                      interpret_complete=lambda s, u: _j.dumps({"stresses": stresses}))


def _new(client, email):
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"name": "J", "email": email, "date": "2026-06-25"})
    return tid


def test_mine_profile_adds_tag_stresses(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, _PROFILE, ["Chronic fatigue"]).test_client()
    tid = _new(client, "j@x.com")
    j = client.post(f"/author/{tid}/mine-profile", json={}).get_json()
    assert j["added"] >= 2
    data = client.get(f"/author/{tid}/stresses").get_json()["data"]
    labels = {x["label"] for x in data["active"] + data["balanced"]}
    sources = {x["source"] for x in data["active"] + data["balanced"]}
    assert {"Inflammation", "Eczema", "Chronic fatigue"} <= labels and "tag" in sources


def test_mine_profile_no_email(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, _PROFILE, []).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    j = client.post(f"/author/{tid}/mine-profile", json={}).get_json()
    assert j["added"] == 0 and "error" in j


def test_mine_profile_empty_profile(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db, {}, []).test_client()
    tid = _new(client, "nobody@x.com")
    assert client.post(f"/author/{tid}/mine-profile", json={}).get_json()["added"] == 0


def test_mine_profile_failure_is_best_effort(tmp_path):
    db = str(tmp_path / "c.db")
    def boom(e):
        raise RuntimeError("people hub down")
    client = create_app(db, scan_lookup=lambda e: _NONE, fetch_profile=boom,
                        interpret_complete=lambda s, u: "{}").test_client()
    tid = _new(client, "j@x.com")
    j = client.post(f"/author/{tid}/mine-profile", json={}).get_json()
    assert j["added"] == 0 and "error" in j
```

- [ ] **Step 2: Run** → FAIL (route 404 / `fetch_profile` kwarg unknown).

- [ ] **Step 3: Implement** — in `biofield_local_app.py`:

(a) Add `fetch_profile=None` to the `create_app(...)` signature, and after the `scan_lookup = scan_lookup or (...)` block add the default impl:

```python
    fetch_profile = fetch_profile or _default_fetch_profile
```

(b) Add the module-level default fetcher near the other helpers (e.g. beside `deepgram_temp_key`):

```python
def _default_fetch_profile(email):
    """Best-effort: pull a client's consolidated profile from the prod People hub
    (same endpoint e4l-reveal-push.py:fetch_history uses). Returns {} on any failure."""
    import urllib.parse, urllib.request
    email = (email or "").strip()
    if not email:
        return {}
    try:
        key = os.environ["CONSOLE_SECRET"]
        base = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
        url = f"{base}/api/people?key=" + urllib.parse.quote(key) + "&q=" + urllib.parse.quote(email)
        req = urllib.request.Request(url, headers={"X-Console-Key": key})
        people = json.load(urllib.request.urlopen(req, timeout=20)).get("people", [])
        return next((p for p in people if (p.get("email") or "").lower() == email.lower()), {})
    except Exception:
        return {}
```

(c) Add the `_mine_profile` helper inside `create_app` (near `_seed_stresses`):

```python
    def _mine_profile(cx, test_id):
        from dashboard.biofield_interpret import interpret_stresses
        from dashboard.biofield_profile import mine_profile_stresses
        from dashboard import biofield_stress as _st
        rep = _report_for(cx, test_id)
        email = ((rep.get("client") or {}).get("email") or "").strip()
        if not email:
            return {"added": 0, "error": "No client selected yet"}
        try:
            profile = fetch_profile(email) or {}
            labels = mine_profile_stresses(profile, lambda t: interpret_stresses(t, interpret_complete))
            added = sum(1 for label in labels if _st.add_stress(cx, test_id, label, source="tag"))
        except Exception as e:
            return {"added": 0, "error": str(e)[:200]}
        return {"added": added}
```

(d) Add the route (near `author_capture_stresses`):

```python
    @app.route("/author/<test_id>/mine-profile", methods=["POST"])
    def author_mine_profile(test_id):
        with sqlite3.connect(db_path) as cx:
            return _mine_profile(cx, test_id)
```

(e) In `_seed_stresses`, after the scan-seeding work, add a best-effort profile mine (so it's always-on and runs even without a scan). At the END of `_seed_stresses` add:

```python
        try:
            _mine_profile(cx, test_id)
        except Exception:
            pass
```
(Place this so it runs regardless of whether a scan was found — i.e. not inside the `if not ctx.get("found"): return` early-out. If the current `_seed_stresses` early-returns on no scan, hoist the `_mine_profile` call to run before that return, or restructure so mining always runs. The implementer should ensure profile mining executes even when there is no fresh scan.)

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_mine_profile_routes.py tests/test_biofield_stress_routes.py tests/test_biofield_capture_stresses_routes.py -v` → PASS (B1/B2 routes green).

- [ ] **Step 5: Commit**

```bash
git add biofield_local_app.py tests/test_biofield_mine_profile_routes.py
git commit -m "feat(biofield-b3a): fetch_profile + mine-profile route + always-on hook"
```

---

### Task 4: Narrative profile weaving

**Files:**
- Modify: `dashboard/biofield_narrative.py`; `biofield_local_app.py` (`narrative_generate` route)
- Test: `tests/test_biofield_narrative_profile.py`

**Interfaces:**
- Produces: `build_narrative_prompt(report, notes, scan=None, profile=None)` and `generate_narrative(report, notes, complete, scan=None, profile=None)`; with `profile=None` the output is byte-identical to before. `narrative_generate` route passes `profile=` (fetched best-effort).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_narrative_profile.py
from dashboard.biofield_narrative import build_narrative_prompt

_REP = {"client": {"name": "Jane"}, "date": "2026-06-25", "layers": []}


def test_profile_none_is_backcompat():
    a = build_narrative_prompt(_REP, "notes")
    b = build_narrative_prompt(_REP, "notes", profile=None)
    assert a == b
    assert "CLIENT-STATED" not in a["user"]


def test_profile_content_appended():
    prof = {"challenges": "always tired", "goals": "sleep better", "conditions": "Eczema"}
    p = build_narrative_prompt(_REP, "notes", profile=prof)
    assert "always tired" in p["user"] and "sleep better" in p["user"]
    assert "CLIENT-STATED" in p["user"]          # the profile block header
    assert p["system"] != build_narrative_prompt(_REP, "notes")["system"]   # guidance appended


def test_empty_profile_no_block():
    p = build_narrative_prompt(_REP, "notes", profile={})
    assert "CLIENT-STATED" not in p["user"]
    assert p["system"] == build_narrative_prompt(_REP, "notes")["system"]
```

- [ ] **Step 2: Run** → FAIL (`profile` kwarg unknown).

- [ ] **Step 3: Implement** — in `dashboard/biofield_narrative.py`:

Add after `_SCAN_GUIDANCE`:

```python
_PROFILE_GUIDANCE = (
    "\n- If a CLIENT-STATED CONCERNS block is present, acknowledge the client's own "
    "stated symptoms, challenges, and goals in plain, validating language and connect "
    "them to the causal chain where honest to do so. Do not invent concerns beyond those listed.")

_PROFILE_FIELDS = ("conditions", "challenges", "goals", "tags", "terrain_concerns", "body_systems")


def _profile_content(profile):
    return bool(profile) and any(str((profile or {}).get(f) or "").strip() for f in _PROFILE_FIELDS)


def _profile_block(profile):
    if not _profile_content(profile):
        return ""
    lines = ["CLIENT-STATED CONCERNS (acknowledge in the client's own terms):"]
    for f in _PROFILE_FIELDS:
        v = profile.get(f)
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x).strip() for x in v if str(x).strip())
        v = str(v or "").strip()
        if v:
            lines.append(f"- {f.replace('_', ' ')}: {v}")
    return "\n".join(lines)
```

Update `_user_block` to take + append the profile block:

```python
def _user_block(report, notes, scan=None, profile=None):
    c = report.get("client") or {}
    lines = [f"PATIENT: {c.get('name') or ''}",
             f"DATE: {report.get('date') or ''}",
             "",
             "CAUSAL CHAIN (top-down, most recent layer first to deepest root):"]
    for l in report.get("layers") or []:
        ln = l.get("layer")
        lines.append(
            f"- Layer {ln if ln is not None else '?'}: {l.get('head') or ''}"
            f" (most affected: {l.get('most_affected') or ''})"
            f" -> remedy: {l.get('remedy') or ''}; dose: {l.get('dosage') or ''}"
            f" {l.get('frequency') or ''} {l.get('timing') or ''}".rstrip())
    sb = _scan_block(scan)
    if sb:
        lines += ["", sb]
    pb = _profile_block(profile)
    if pb:
        lines += ["", pb]
    lines += ["", "CLINICIAN VERBAL NOTES (weave in naturally):", (notes or "(none)")]
    return "\n".join(lines)
```

Update the prompt + generate signatures:

```python
def build_narrative_prompt(report, notes, scan=None, profile=None):
    system = _system_with_scan(_SYSTEM, scan)
    if _profile_content(profile):
        system += _PROFILE_GUIDANCE
    return {"system": system, "user": _user_block(report, notes, scan, profile)}


def generate_narrative(report, notes, complete, scan=None, profile=None):
    """complete(system, user) -> narrative text. scan = E4L context; profile = People-hub context."""
    p = build_narrative_prompt(report, notes, scan, profile)
    return complete(p["system"], p["user"])
```

Then in `biofield_local_app.py` `narrative_generate`, fetch + pass the profile (best-effort) — change the `generate_narrative(...)` call:

```python
            prof = {}
            try:
                prof = fetch_profile(((rep.get("client") or {}).get("email") or "").strip()) or {}
            except Exception:
                prof = {}
            text = generate_narrative(rep, notes, complete, scan=ctx, profile=prof)
```

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_narrative_profile.py tests/test_biofield_narrative.py tests/test_biofield_narrative_scan.py -v` → PASS (existing narrative + scan-narrative tests stay green — `profile=None` keeps prompts byte-identical).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_narrative.py biofield_local_app.py tests/test_biofield_narrative_profile.py
git commit -m "feat(biofield-b3a): weave client profile into the narrative (back-compatible)"
```

---

### Task 5: UI — "Mine profile → stresses" button

**Files:**
- Modify: `dashboard/biofield_report_html.py`
- Test: `tests/test_biofield_mine_profile_button.py`

**Interfaces:**
- Consumes: existing `__TID__`/`post()`/`rstat()`/`loadStress()` JS conventions.
- Produces: a "Mine profile → stresses" button in the author page; `mineProfile()` JS POSTs to `/author/__TID__/mine-profile` then calls `loadStress()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_biofield_mine_profile_button.py
from dashboard.biofield_report_html import render_author_html


def _html():
    rep = {"test_id": "a7", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [], "schedule": []}
    return render_author_html(rep, [], "")


def test_mine_profile_button_and_handler():
    h = _html()
    assert "Mine profile" in h
    assert "function mineProfile" in h
    assert "/author/a7/mine-profile" in h
    assert "loadStress()" in h
```

- [ ] **Step 2: Run** → FAIL (button/handler absent).

- [ ] **Step 3: Implement** — in `dashboard/biofield_report_html.py`:

(a) In `_AUTHOR_JS`, add (next to `captureStresses`):

```javascript
async function mineProfile(){rstat('Mining client profile for stresses\\u2026');
 var j=await post('/author/__TID__/mine-profile',{});
 if(j.error){rstat('Mine profile: '+j.error);return}
 rstat('Added '+j.added+' profile stress(es).');loadStress()}
```

(b) Add the button. Put it in the live-session widget's toggle row (next to the Phase buttons) OR immediately above the `<div id=stresspanel></div>`. Add this button markup adjacent to the stress panel container (in the `render_author_html` body where `<div id=stresspanel></div>` is rendered):

```python
        "<div class=btnrow style='margin:6px 0'>"
        "<button class='btn ghost' onclick=mineProfile()>Mine profile &rarr; stresses</button>"
        "</div>"
        "<div id=stresspanel></div>"
```
(Replace the existing bare `"<div id=stresspanel></div>"` occurrence with the button row + the div, preserving everything else around it.)

- [ ] **Step 4: Run** → `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_mine_profile_button.py tests/test_biofield_author_html.py tests/test_biofield_stress_panel.py -v` → PASS.

- [ ] **Step 5: Run the B3a + adjacent suite + commit**

```bash
~/.venvs/deploy-chat311/bin/python -m pytest \
  tests/test_biofield_add_stress.py tests/test_biofield_profile_mine.py \
  tests/test_biofield_mine_profile_routes.py tests/test_biofield_narrative_profile.py \
  tests/test_biofield_mine_profile_button.py tests/test_biofield_add_voice_stress.py \
  tests/test_biofield_stress_routes.py tests/test_biofield_capture_stresses_routes.py \
  tests/test_biofield_stress_derive.py tests/test_biofield_stress_labelmatch.py \
  tests/test_biofield_narrative.py tests/test_biofield_narrative_scan.py \
  tests/test_biofield_author_html.py tests/test_biofield_stress_panel.py -q
git add dashboard/biofield_report_html.py tests/test_biofield_mine_profile_button.py
git commit -m "feat(biofield-b3a): Mine-profile button + mineProfile()"
```

---

## Self-Review

**Spec coverage:**
- `fetch_profile` (injectable HTTP, /api/people) → Task 3. ✓
- `mine_profile_stresses` discrete + free-text via interpret_stresses → Task 2. ✓
- `add_stress(source='tag')` generalization, B2-compatible → Task 1. ✓
- mine-profile route + always-on seed hook (runs without a scan) → Task 3. ✓
- Narrative weaving, back-compatible → Task 4. ✓
- UI button → Task 5. ✓
- Merge by normalized label / source='tag' / best-effort → Tasks 1 & 3. ✓

**Placeholder scan:** No TBDs; every code step complete. Task 3(e) and Task 5(b) explicitly tell the implementer to preserve surrounding code and ensure mining runs even without a scan.

**Type consistency:** `add_stress(cx,tid,label,*,source,balance)->bool` (T1) consumed by T3; `mine_profile_stresses(profile, extract)->[str]` (T2) consumed by T3; `fetch_profile(email)->dict` (T3) consumed by T4's route; `build_narrative_prompt/generate_narrative(..., profile=None)` (T4). Consistent.

## Verification (manual, after all tasks)

```bash
cd ~/deploy-chat && doppler run -p remedy-match -c prd -- python3 biofield_local_app.py
```
Open a test for a client with a People-hub profile: "Mine profile → stresses" (and header-save) adds their tags/conditions + LLM-extracted challenges/goals to the Active stress list (deduped against scan/voice). Generate the narrative and confirm it acknowledges the client's stated concerns.
