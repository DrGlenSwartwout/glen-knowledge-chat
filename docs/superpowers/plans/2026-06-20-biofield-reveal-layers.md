# Biofield Reveal Stress-Pattern Layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry the matcher's stress-pattern layers into the Biofield reveal end to end - stored on the reveal, editable/namable in the console (pending AND approved), and rendered to the member as titled layers with title+summary always shown and the per-layer remedy gated by the existing free-top / $1-rest model.

**Architecture:** Add a `layers` field to `biofield_reveals`; the matcher push contract gains `layers` (back-compatible with old remedies-only pushes); the ingest guardrail/canonical-override runs per layer remedy; a derived flat `remedies` keeps #4b/#4c working; the reveal payload emits titles+summaries always and gates per-layer remedies; the console editor renders/edits layers and lists+edits approved reveals; the member reveal renders titled layers. Approved reveals become re-editable.

**Tech Stack:** Python 3.11 / Flask (single `app.py`), SQLite (`LOG_DB`), `dashboard/biofield_reveals.py`, `dashboard/biofield_reveal_actions.py` (dispatch spine), pytest. Front-end is vanilla JS in `static/begin-biofield.html` + `static/console-biofield-reveals.html`.

## Global Constraints

- No emoji, no em dashes (code, comments, commit messages).
- A layer is `{n:int, title:str, summary:str, patterns:[str], remedy:{name,slug,meaning}|null}`. `title` + `summary` are member-shown and ALWAYS visible; `patterns` (raw codes) are NOT member-shown; `remedy` is gated.
- Gating: titles + summaries always shown; the per-layer remedy is visible only when paid (all layers) or, for a free member, the TOP layer's remedy after `first_approved` + the member's one-time free unblock. A withheld remedy's name/slug/meaning NEVER leaves the server (anti-bypass).
- The flat `remedies` column/payload stays, DERIVED from the layers (ordered, surviving remedies), so `_biofield_visible_slugs`, the #4b $1 trial, and the #4c cart are unchanged.
- Ingest guardrail per layer: resolve the remedy to a catalog slug (else DROP the remedy from its layer - keep the layer's title+summary - and record the dropped name); apply the canonical-meaning override when a slug resolves. The ingest NEVER rejects a push.
- Back-compat: a push with only `remedies` (old matcher) is wrapped into titleless single-remedy layers; a reveal with no layers (legacy) falls back to the flat-remedy render.
- Re-editability: the console edit methods work on approved reveals too; the matcher re-push (`upsert`) still only updates un-approved reveals.
- Everything wrapped/never-raises; the member render never emits a withheld remedy.
- Test runner: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest <target> -v`. tmp `LOG_DB`; mock Stripe/LLM where needed.

---

## Critical files

- `dashboard/biofield_reveals.py` - `layers_json` column, `_row` parse, `set_layers`, `list_approved`, `upsert(... , layers=None)`, drop the `first_approved=0` guard from `set_interpretation`/`set_remedies`/`set_layers`.
- `app.py` - the `/api/e4l/reveal-draft` ingest (layers + per-layer guardrail + derive remedies + back-compat); `_biofield_layer_payload` + the `layers` block in `begin_biofield_reveal`; the console data endpoint's approved list.
- `dashboard/biofield_reveal_actions.py` - `_exec_edit` accepts `layers`, writes `set_layers`, re-derives `remedies`, promotes "remember".
- `static/begin-biofield.html` - member render of titled layers.
- `static/console-biofield-reveals.html` - console layer editor + approved section.
- Tests: `tests/test_biofield_layers.py` (new); keep `tests/test_biofield_trial.py` + `tests/test_biofield_cart.py` green.

---

## Task 1: Store - layers column, set_layers, list_approved, re-editability (`dashboard/biofield_reveals.py`)

**Files:**
- Modify: `dashboard/biofield_reveals.py`
- Test: `tests/test_biofield_layers.py`

**Interfaces:**
- Produces: a `layers_json` column (idempotent ALTER); `_row` returns `layers` (a list); `upsert(cx, email, scan_date, interpretation, remedies, source, layers=None)` stores layers; `set_layers(cx, rid, layers)`; `list_approved(cx, limit=50)`; `set_interpretation`/`set_remedies`/`set_layers` work regardless of `first_approved` (re-editable).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_biofield_layers.py`:

```python
# tests/test_biofield_layers.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load(mod):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module(mod)
    except Exception as e:
        pytest.skip(f"{mod} not importable: {e}")


def _LAYERS():
    return [
        {"n": 1, "title": "Mineral Mobilization", "summary": "Surface mineral stress.",
         "patterns": ["A1"], "remedy": {"name": "Nous Energy", "slug": "nous-energy", "meaning": "m"}},
        {"n": 2, "title": "Terrain Balance", "summary": "Deeper biofilm terrain.",
         "patterns": ["B2"], "remedy": {"name": "Cistus", "slug": "cistus-syntropy-immunitea", "meaning": "m2"}},
    ]


def test_layers_roundtrip(tmp_path):
    br = _load("dashboard.biofield_reveals")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx); br.init_table(cx)  # idempotent ALTER
        rid, is_new = br.upsert(cx, "a@b.com", "2026-06-20", {"body": "x"}, [], "s", layers=_LAYERS())
        row = br.get(cx, rid)
    assert is_new and len(row["layers"]) == 2 and row["layers"][0]["title"] == "Mineral Mobilization"


def test_set_layers_and_reedit_when_approved(tmp_path):
    br = _load("dashboard.biofield_reveals")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        rid, _ = br.upsert(cx, "a@b.com", "2026-06-20", {"body": "x"}, [], "s", layers=_LAYERS())
        br.approve_first(cx, rid, "glen")  # now first_approved=1
        new = _LAYERS(); new[0]["title"] = "Renamed Layer"
        br.set_layers(cx, rid, new)
        br.set_interpretation(cx, rid, {"body": "edited after approval"})
        row = br.get(cx, rid)
    assert row["first_approved"] is True
    assert row["layers"][0]["title"] == "Renamed Layer"        # set_layers works post-approval
    assert row["interpretation"]["body"] == "edited after approval"  # set_interpretation too


def test_list_approved(tmp_path):
    br = _load("dashboard.biofield_reveals")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        r1, _ = br.upsert(cx, "p@x.com", "2026-06-20", {}, [], "s")
        r2, _ = br.upsert(cx, "a@x.com", "2026-06-20", {}, [], "s")
        br.approve_first(cx, r2, "glen")
        pend = [r["id"] for r in br.list_pending(cx)]
        appr = [r["id"] for r in br.list_approved(cx)]
    assert r1 in pend and r2 not in pend and r2 in appr and r1 not in appr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -k "layers_roundtrip or reedit or list_approved" -v`
Expected: FAIL (`upsert` has no `layers` kw / `set_layers` missing / re-edit blocked / `list_approved` missing).

- [ ] **Step 3: Add the column + methods**

In `dashboard/biofield_reveals.py`, in `init_table`, add the additive column next to the existing `dropped` ALTER:

```python
    try:
        cx.execute("ALTER TABLE biofield_reveals ADD COLUMN layers_json TEXT NOT NULL DEFAULT '[]'")
    except Exception:
        pass
```

In `_row`, parse it (after the `dropped` line):

```python
    d["layers"] = json.loads(d.pop("layers_json", "[]") or "[]")
```

Change `upsert` to accept + store `layers` (both the INSERT and the not-approved UPDATE). Replace the `upsert` body's UPDATE and INSERT with:

```python
def upsert(cx, email, scan_date, interpretation, remedies, source, layers=None):
    """Insert or update a reveal. Content updates only while not yet approved (matcher
    re-push). Returns (id, is_new). layers defaults to [] when not provided."""
    email = (email or "").strip().lower()
    now = _now()
    lj = json.dumps(layers or [])
    existing = cx.execute(
        "SELECT id, first_approved FROM biofield_reveals WHERE email=? AND scan_date=?",
        (email, scan_date)).fetchone()
    if existing is not None:
        rid, approved = existing[0], existing[1]
        if not approved:
            cx.execute(
                "UPDATE biofield_reveals SET interpretation_json=?, remedies_json=?, layers_json=?, updated_at=? WHERE id=?",
                (json.dumps(interpretation or {}), json.dumps(remedies or []), lj, now, rid))
            cx.commit()
        return rid, False
    cur = cx.execute(
        "INSERT INTO biofield_reveals (email, scan_date, interpretation_json, remedies_json, layers_json, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (email, scan_date, json.dumps(interpretation or {}), json.dumps(remedies or []), lj, now, now))
    cx.commit()
    return cur.lastrowid, True
```

Drop the `AND first_approved=0` guard from `set_interpretation` and `set_remedies` (re-editability), and add `set_layers`:

```python
def set_interpretation(cx, rid, interpretation):
    cx.execute("UPDATE biofield_reveals SET interpretation_json=?, updated_at=? WHERE id=?",
               (json.dumps(interpretation or {}), _now(), rid))
    cx.commit()


def set_remedies(cx, rid, remedies):
    cx.execute("UPDATE biofield_reveals SET remedies_json=?, updated_at=? WHERE id=?",
               (json.dumps(remedies or []), _now(), rid))
    cx.commit()


def set_layers(cx, rid, layers):
    cx.execute("UPDATE biofield_reveals SET layers_json=?, updated_at=? WHERE id=?",
               (json.dumps(layers or []), _now(), rid))
    cx.commit()
```

Add `list_approved` (next to `list_pending`):

```python
def list_approved(cx, limit=50):
    rows = _rows_cursor(cx).execute(
        "SELECT * FROM biofield_reveals WHERE first_approved=1 ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -v`
Expected: PASS (3 tests). Then `tests/test_biofield_trial.py tests/test_biofield_cart.py -q` stay green (upsert's new kwarg is optional; the dropped guards do not change pending-reveal behavior).

- [ ] **Step 5: Commit**

```bash
git add dashboard/biofield_reveals.py tests/test_biofield_layers.py
git commit -m "feat: reveal layers column + set_layers + list_approved + re-editable approved"
```

---

## Task 2: Ingest - layers + per-layer guardrail + derived remedies + back-compat (`app.py`)

**Files:**
- Modify: `app.py` (the `api_e4l_reveal_draft` ingest block)
- Test: `tests/test_biofield_layers.py`

**Interfaces:**
- Consumes: `_resolve_remedy_slug`, `dashboard.biofield_meanings.get_map`, `dashboard.biofield_reveals` (`upsert` with `layers=`, `set_dropped`).
- Produces: the ingest stores `layers` (each layer's remedy guardrail-cleaned: catalog slug resolved, non-catalog remedy dropped from the layer + recorded, canonical override applied) and the DERIVED flat `remedies` (ordered surviving layer remedies); a `remedies`-only push is wrapped into titleless layers.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_biofield_layers.py` (these load `app`):

```python
def _app_db(monkeypatch, tmp_path):
    app_module = _load("app")
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals as br, biofield_meanings as bm
    with sqlite3.connect(db) as cx:
        br.init_table(cx); bm.init_table(cx)
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return app_module, db


def _push(app_module, body, key):
    return app_module.app.test_client().post("/api/e4l/reveal-draft", headers={"X-Console-Key": key}, json=body)


def _key(app_module):
    import os
    return os.environ.get("CRON_SECRET") or app_module.CONSOLE_SECRET or ""


def test_ingest_stores_layers_and_derives_remedies(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = _key(app_module)
    if not key: pytest.skip("no secret")
    prods = app_module._PRODUCTS.get("products") or {}
    real = next(iter(prods), None)
    if not real: pytest.skip("no catalog")
    rname = prods[real]["name"]
    layers = [{"n": 1, "title": "Layer One", "summary": "s1", "patterns": ["A"],
               "remedy": {"name": rname, "slug": real, "meaning": "pushed"}},
              {"n": 2, "title": "Layer Two", "summary": "s2", "patterns": ["B"],
               "remedy": {"name": "Totally Made Up", "slug": "nope-xyz", "meaning": "ghost"}}]
    r = _push(app_module, {"email": "a@b.com", "scan_date": "2026-06-20",
                           "interpretation": {"body": "x"}, "layers": layers}, key)
    assert r.get_json().get("ok") is True
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        row = br.list_pending(cx)[0]
    titles = [L["title"] for L in row["layers"]]
    assert titles == ["Layer One", "Layer Two"]                 # both layers kept (titles always)
    assert row["layers"][0]["remedy"]["slug"] == real           # catalog remedy survives
    assert row["layers"][1]["remedy"] is None                   # non-catalog remedy dropped from its layer
    assert "Totally Made Up" in row["dropped"]
    assert [rr["slug"] for rr in row["remedies"]] == [real]      # derived flat remedies = surviving layer remedies


def test_ingest_remedies_only_wraps_into_layers(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = _key(app_module)
    if not key: pytest.skip("no secret")
    prods = app_module._PRODUCTS.get("products") or {}
    real = next(iter(prods), None)
    if not real: pytest.skip("no catalog")
    _push(app_module, {"email": "c@b.com", "scan_date": "2026-06-20", "interpretation": {"body": "x"},
                       "remedies": [{"name": prods[real]["name"], "slug": real, "meaning": "m"}]}, key)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        row = br.list_pending(cx)[0]
    assert len(row["layers"]) == 1 and row["layers"][0]["title"] == ""   # titleless wrap
    assert row["layers"][0]["remedy"]["slug"] == real
    assert [rr["slug"] for rr in row["remedies"]] == [real]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -k ingest -v`
Expected: FAIL (no layers stored; remedies-only not wrapped).

- [ ] **Step 3: Rewrite the ingest guardrail block**

In `app.py`, in `api_e4l_reveal_draft`, replace the existing guardrail block (the `cleaned, dropped = [], []` loop through `_br.upsert(...)` + `_br.set_dropped(...)`) with a layer-aware version. The new block (inside the same `with _db_lock, sqlite3.connect(LOG_DB) as cx:` after `_br.init_table(cx)` / `_bm.init_table(cx)` / `canon = ...`):

```python
            # Build layers: prefer the pushed `layers`; else wrap each pushed remedy
            # into a titleless single-remedy layer (back-compat with the old matcher).
            raw_layers = data.get("layers")
            if not isinstance(raw_layers, list) or not raw_layers:
                raw_layers = [{"n": i + 1, "title": "", "summary": "", "patterns": [],
                               "remedy": rr} for i, rr in enumerate(remedies or []) if isinstance(rr, dict)]
            cleaned_layers, dropped = [], []
            for i, L in enumerate(raw_layers):
                if not isinstance(L, dict):
                    continue
                rem = L.get("remedy") if isinstance(L.get("remedy"), dict) else None
                out_rem = None
                if rem is not None:
                    slug = _resolve_remedy_slug(rem)
                    if not slug:
                        dropped.append((rem.get("name") or "").strip() or "(unnamed)")
                    else:
                        out_rem = dict(rem); out_rem["slug"] = slug
                        cm = canon.get(slug)
                        if cm:
                            out_rem["meaning"] = cm
                cleaned_layers.append({
                    "n": L.get("n", i + 1), "title": (L.get("title") or "").strip(),
                    "summary": (L.get("summary") or "").strip(),
                    "patterns": L.get("patterns") or [], "remedy": out_rem})
            # Derived flat remedies = the surviving layer remedies, in order (for #4b/#4c).
            derived = [L["remedy"] for L in cleaned_layers if L.get("remedy")]
            rid, is_new = _br.upsert(cx, email, scan_date, interp, derived,
                                     (data.get("source") or "").strip(), layers=cleaned_layers)
            try:
                _br.set_dropped(cx, rid, dropped)
            except Exception:
                pass
```

(The token-mint-on-`is_new` + email block below it stays unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -k ingest -v`
Expected: PASS. Then `tests/test_biofield_trial.py tests/test_biofield_cart.py -q` stay green (remedies-only pushes still produce the same derived `remedies`).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_biofield_layers.py
git commit -m "feat: reveal ingest stores layers + per-layer guardrail + derived remedies"
```

---

## Task 3: Reveal payload - emit layers (titles+summaries always, remedy gated) (`app.py`)

**Files:**
- Modify: `app.py` (add `_biofield_layer_payload`; add `layers` to both payloads in `begin_biofield_reveal`)
- Test: `tests/test_biofield_layers.py`

**Interfaces:**
- Consumes: `_biofield_remedy_payload`, `_biofield_unlock_flags`.
- Produces: `_biofield_layer_payload(layer, include_remedy) -> dict` ({n, title, summary, remedy|None, remedy_blurred}); the reveal payload gains `layers` (titles+summaries always; per-layer remedy present only when visible; withheld remedy details never emitted).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_biofield_layers.py`:

```python
def _seed_approved_layers(app_module, db, email="t@x.com"):
    import secrets as _s
    from datetime import datetime, timezone, timedelta
    from dashboard import biofield_reveals as br
    token = "tk_" + _s.token_urlsafe(8)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, email, "2026-06-20", {"greeting": "Hi", "body": "b"},
                           [{"name": "Top", "slug": "rx-aaa", "meaning": "m"},
                            {"name": "Deep", "slug": "rx-bbb", "meaning": "m2"}], "s",
                           layers=[{"n": 1, "title": "Surface", "summary": "s1", "patterns": [],
                                    "remedy": {"name": "Top", "slug": "rx-aaa", "meaning": "m"}},
                                   {"n": 2, "title": "Root", "summary": "s2", "patterns": [],
                                    "remedy": {"name": "Deep", "slug": "rx-bbb", "meaning": "m2"}}])
        br.set_token(cx, rid, th); br.approve_first(cx, rid, "glen")
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, email, "biofield_reveal", datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))
        cx.commit()
    return token


def _reveal_payload(app_module, token):
    import re, json as _j
    html = app_module.app.test_client().get(f"/begin/biofield/{token}").get_data(as_text=True)
    m = re.search(r"window.__REVEAL__ = (\{.*?\});", html)
    return _j.loads(m.group(1).replace("\\u003c", "<").replace("\\u003e", ">").replace("\\u0026", "&")) if m else None


def test_payload_titles_always_remedy_gated_nonpaid(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)  # not paid
    token = _seed_approved_layers(app_module, db)  # approved but member has NOT claimed free unlock
    d = _reveal_payload(app_module, token)
    assert d["paid"] is False
    titles = [L["title"] for L in d["layers"]]
    assert titles == ["Surface", "Root"]                 # titles always shown
    assert all(L["summary"] for L in d["layers"])        # summaries always shown
    assert all(L["remedy"] is None and L["remedy_blurred"] for L in d["layers"])  # no remedy visible yet
    blob = __import__("json").dumps(d)
    assert "rx-aaa" not in blob and "rx-bbb" not in blob   # anti-bypass: withheld remedy slugs never emitted


def test_payload_paid_shows_all_layer_remedies(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})  # paid
    token = _seed_approved_layers(app_module, db)
    d = _reveal_payload(app_module, token)
    assert d["paid"] is True
    assert [L["title"] for L in d["layers"]] == ["Surface", "Root"]
    assert d["layers"][0]["remedy"]["slug"] == "rx-aaa" and d["layers"][1]["remedy"]["slug"] == "rx-bbb"
    assert all(not L["remedy_blurred"] for L in d["layers"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -k payload -v`
Expected: FAIL (`layers` not in the payload).

- [ ] **Step 3: Add the layer-payload helper**

In `app.py`, immediately after `_biofield_top_payload` (~line 1559), add:

```python
def _biofield_layer_payload(layer, include_remedy):
    """Layer payload: title + summary ALWAYS; the remedy only when include_remedy and present.
    A withheld remedy is emitted as remedy=None, remedy_blurred=True (its details never leave)."""
    try:
        rem = layer.get("remedy") if isinstance(layer, dict) else None
        has_rem = isinstance(rem, dict) and ((rem.get("slug") or "").strip() or (rem.get("name") or "").strip())
        out = {"n": layer.get("n"), "title": (layer.get("title") or ""),
               "summary": (layer.get("summary") or "")}
        if include_remedy and has_rem:
            out["remedy"] = _biofield_remedy_payload(rem)
            out["remedy_blurred"] = False
        else:
            out["remedy"] = None
            out["remedy_blurred"] = bool(has_rem)
        return out
    except Exception:
        return {"n": None, "title": "", "summary": "", "remedy": None, "remedy_blurred": False}
```

- [ ] **Step 4: Add `layers` to both payloads**

In `begin_biofield_reveal`, build the layers payload from `row["layers"]` and add it to both the paid and non-paid `payload` dicts. After the `flags = _biofield_unlock_flags(...)` line and before the `if paid:` block, add:

```python
    _layers_raw = row.get("layers") or []
    if paid:
        _layers_payload = [_biofield_layer_payload(L, include_remedy=True) for L in _layers_raw]
    else:
        _layers_payload = [_biofield_layer_payload(L, include_remedy=(top_unlocked and i == 0))
                           for i, L in enumerate(_layers_raw)]
    _blurred_layers = sum(1 for lp in _layers_payload if lp["remedy_blurred"])
```

Then add `"layers": _layers_payload` to BOTH payload dicts, and set `"blurred_count": _blurred_layers` in the non-paid payload (replace the existing `len(...) - (1 if top_unlocked else 0)`); keep `"blurred_count": 0` for paid. The paid payload keeps its existing `"remedies": [...]` (back-compat); the non-paid payload keeps `"top"` (back-compat for legacy reveals with no layers).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -k payload -v`
Expected: PASS. Then `tests/test_biofield_trial.py tests/test_biofield_cart.py -q` stay green (the existing `remedies`/`top`/`blurred_count` fields remain).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_biofield_layers.py
git commit -m "feat: reveal payload emits titled layers, per-layer remedy gated (anti-bypass)"
```

---

## Task 4: Member render - titled layers (`static/begin-biofield.html`)

**Files:**
- Modify: `static/begin-biofield.html`
- Test: `tests/test_biofield_layers.py` (serve assertion)

**Interfaces:**
- Consumes: the reveal payload's `layers` ([{n, title, summary, remedy|null, remedy_blurred}]).
- Produces: titled-layer rendering on the reveal page; falls back to the existing flat-remedy render when `layers` is absent/empty.

- [ ] **Step 1: Write the failing serve test**

Add to `tests/test_biofield_layers.py`:

```python
def test_member_page_ships_layer_render(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    token = _seed_approved_layers(app_module, db)
    html = app_module.app.test_client().get(f"/begin/biofield/{token}").get_data(as_text=True)
    assert "renderLayers" in html and "layer-title" in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py::test_member_page_ships_layer_render -v`
Expected: FAIL (`renderLayers` not in the page).

- [ ] **Step 3: Add the layer render**

In `static/begin-biofield.html`, add CSS in the `<style>` block:

```css
  .bf-layer{ margin-top:22px; padding-top:16px; border-top:1px solid var(--border,#ddd); }
  .bf-layer .layer-title{ font-weight:700; font-size:1.05em; }
  .bf-layer .layer-summary{ color:var(--muted,#666); margin:6px 0 10px; white-space:pre-wrap; }
  .bf-layer .layer-blurred{ filter:blur(5px); user-select:none; pointer-events:none; }
```

In the member-render script, add a `renderLayers(container, data)` function and call it when `data.layers` is present (preferring it over the flat-remedy render). Insert near the existing remedy-render code. The function (all dynamic text via textContent/setAttribute, no innerHTML of server data):

```javascript
      function renderLayers(root, data) {
        var layers = data.layers || [];
        for (var i = 0; i < layers.length; i++) {
          var L = layers[i];
          var sec = document.createElement("div");
          sec.className = "bf-layer";
          var t = document.createElement("div"); t.className = "layer-title"; t.textContent = L.title || ("Layer " + (L.n || (i + 1)));
          sec.appendChild(t);
          if (L.summary) { var s = document.createElement("p"); s.className = "layer-summary"; s.textContent = L.summary; sec.appendChild(s); }
          if (L.remedy && L.remedy.slug) {
            renderRemedyInto(sec, L.remedy);              // existing helper: name link + meaning + Order btn (+ #4c cart row)
          } else if (L.remedy_blurred) {
            var b = document.createElement("p"); b.className = "layer-blurred"; b.textContent = "Your matched remedy for this layer";
            sec.appendChild(b);
          }
          root.appendChild(sec);
        }
      }
```

In the member-render flow (the `State 3: Member` branch, after the interpretation card), call `renderLayers` when `data.layers && data.layers.length` (and SKIP the old paid/`data.remedies` and non-paid top/blurred remedy blocks in that case); otherwise fall through to the existing flat-remedy render. Keep the sticky #4c cart bar logic - it already collects `cartItems` from `renderRemedyInto`, so per-layer visible remedies feed the cart unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py::test_member_page_ships_layer_render -v`
Expected: PASS. Then `tests/test_biofield_cart.py -q` green (the cart still reads visible remedy slugs).

- [ ] **Step 5: Commit**

```bash
git add static/begin-biofield.html tests/test_biofield_layers.py
git commit -m "feat: reveal page renders titled stress-pattern layers (remedy gated per layer)"
```

---

## Task 5: Console edit action - accept layers + re-derive remedies + approved list (`dashboard/biofield_reveal_actions.py`, `app.py`)

**Files:**
- Modify: `dashboard/biofield_reveal_actions.py` (`_exec_edit`)
- Modify: `app.py` (the `/api/console/biofield-reveals` data endpoint to also return approved)
- Test: `tests/test_biofield_layers.py`

**Interfaces:**
- Consumes: `dashboard.biofield_reveals` (`set_layers`, `set_interpretation`, `set_remedies`, `get`, `list_pending`, `list_approved`), `dashboard.biofield_meanings.upsert`.
- Produces: `_exec_edit` accepts a `layers` param (each layer's remedy carrying a `remember` flag, default true), writes `set_layers`, re-derives + stores the flat `remedies`, and promotes remembered meanings; the console data endpoint returns `{drafts: [...pending], approved: [...approved]}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_biofield_layers.py`:

```python
def test_edit_action_writes_layers_and_derives_remedies(tmp_path):
    bra = _load("dashboard.biofield_reveal_actions")
    br = _load("dashboard.biofield_reveals")
    bm = _load("dashboard.biofield_meanings")
    db = str(tmp_path / "r.db")
    with sqlite3.connect(db) as cx:
        br.init_table(cx); bm.init_table(cx)
        rid, _ = br.upsert(cx, "a@b.com", "2026-06-20", {"body": "x"}, [], "s", layers=_LAYERS())
        br.approve_first(cx, rid, "glen")  # approved -> must still be editable
        layers = _LAYERS(); layers[0]["title"] = "Edited Title"
        layers[0]["remedy"]["meaning"] = "NEW MEANING"; layers[0]["remedy"]["remember"] = True
        bra._exec_edit({"id": rid, "layers": layers}, {"cx": cx, "actor": None})
        row = br.get(cx, rid)
        canon = bm.get_map(cx)
    assert row["layers"][0]["title"] == "Edited Title"
    assert "remember" not in row["layers"][0]["remedy"]                # stripped from stored layer
    assert [rr["slug"] for rr in row["remedies"]] == ["nous-energy", "cistus-syntropy-immunitea"]  # re-derived
    assert canon.get("nous-energy") == "NEW MEANING"                   # remember promoted to canonical


def test_console_endpoint_returns_pending_and_approved(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "sek", raising=False)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        p, _ = br.upsert(cx, "p@x.com", "2026-06-20", {}, [], "s")
        a, _ = br.upsert(cx, "a@x.com", "2026-06-20", {}, [], "s"); br.approve_first(cx, a, "glen"); cx.commit()
    j = app_module.app.test_client().get("/api/console/biofield-reveals?key=sek").get_json()
    assert p in [d["id"] for d in j["drafts"]] and a in [d["id"] for d in j.get("approved", [])]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -k "edit_action or console_endpoint" -v`
Expected: FAIL (`_exec_edit` ignores `layers`; endpoint has no `approved`).

- [ ] **Step 3: Extend `_exec_edit` for layers**

In `dashboard/biofield_reveal_actions.py`, in `_exec_edit`, after the `set_interpretation` call, add a `layers` branch (and keep the existing `remedies` branch for back-compat):

```python
    if isinstance(params.get("layers"), list):
        from dashboard import biofield_meanings as _bm
        _bm.init_table(ctx["cx"])
        stored_layers, derived = [], []
        for L in params["layers"]:
            if not isinstance(L, dict):
                continue
            rem = L.get("remedy") if isinstance(L.get("remedy"), dict) else None
            out_rem = None
            if rem is not None:
                remember = rem.get("remember", True)  # default ON
                out_rem = {k: v for k, v in rem.items() if k != "remember"}
                slug = (out_rem.get("slug") or "").strip()
                meaning = (out_rem.get("meaning") or "").strip()
                if remember and slug and meaning:
                    try:
                        _bm.upsert(ctx["cx"], slug, meaning, _actor_name(ctx.get("actor")), "glen")
                    except Exception as e:
                        print(f"[remedy-meaning] promote {e!r}", flush=True)
                if out_rem.get("slug"):
                    derived.append(out_rem)
            stored_layers.append({"n": L.get("n"), "title": (L.get("title") or "").strip(),
                                  "summary": (L.get("summary") or "").strip(),
                                  "patterns": L.get("patterns") or [], "remedy": out_rem})
        _br.set_layers(ctx["cx"], rid, stored_layers)
        _br.set_remedies(ctx["cx"], rid, derived)
```

- [ ] **Step 4: Add `approved` to the console data endpoint**

In `app.py`, in `api_console_biofield_reveals`, return both lists:

```python
    with sqlite3.connect(LOG_DB) as cx:
        _br.init_table(cx)
        drafts = _br.list_pending(cx)
        approved = _br.list_approved(cx)
    return jsonify({"drafts": drafts, "approved": approved})
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -v`
Expected: PASS (the whole file). Then `tests/test_biofield_trial.py tests/test_biofield_cart.py -q` stay green.

- [ ] **Step 6: Commit**

```bash
git add dashboard/biofield_reveal_actions.py app.py tests/test_biofield_layers.py
git commit -m "feat: console edit action writes layers + re-derives remedies; endpoint returns approved"
```

---

## Task 6: Console front-end - layer editor + approved section (`static/console-biofield-reveals.html`)

**Files:**
- Modify: `static/console-biofield-reveals.html`
- Test: `tests/test_biofield_layers.py` (serve assertion)

**Interfaces:**
- Consumes: `/api/console/biofield-reveals` (`{drafts, approved}`) and the `biofield_reveal.edit` action (now accepting `layers`).
- Produces: the console renders each reveal's layers (editable title/summary/remedy + per-remedy "remember" toggle), shows the "dropped" notice, lists pending + an approved section, and posts `layers` on Save.

- [ ] **Step 1: Write the failing serve test**

Add to `tests/test_biofield_layers.py`:

```python
def test_console_page_ships_layer_editor(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "sek", raising=False)
    html = app_module.app.test_client().get("/console/biofield-reveals?key=sek").get_data(as_text=True)
    assert "layer-title-field" in html and "Approved" in html and "collectLayers" in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py::test_console_page_ships_layer_editor -v`
Expected: FAIL.

- [ ] **Step 3: Rework the console page to render/edit layers + approved**

In `static/console-biofield-reveals.html`, modify `buildCard(d)` to render each layer of `d.layers` (when present) instead of the flat remedies: per layer a `.layer-title-field` (text input, value `L.title`), a `.layer-summary-field` (textarea, value `L.summary`), and the remedy fields (`.remedy-name`/`.remedy-slug`/`.remedy-meaning` as today) plus the "remember" checkbox; render the existing `.status` + Save/Approve row. Add a `collectLayers(card)` that returns the layers array (mirroring `collectRemedies`, reading title/summary/remedy per layer, including `remember` from the checkbox). Change `saveEdit` to post `{ id, greeting, body, layers: collectLayers(card) }` to `biofield_reveal.edit`. Show `d.dropped` (if non-empty) as a "Dropped (not in catalog): ..." note. Modify `loadList` to render `r.json.drafts` under a "Pending" heading and `r.json.approved` under an "Approved" heading (reuse `buildCard`; approved cards keep the same editor since editing is now allowed post-approval). Keep the `setTimeout(loadList, 900)` reload-on-approve from the prior fix. All dynamic text via textContent/value/setAttribute (no innerHTML of server data). Keep the literal strings `layer-title-field`, `collectLayers`, and an `Approved` heading so the serve test passes. No emoji, no em dashes.

(Model the new fields on the existing remedy fields in this file; read the file first for its exact buildCard/collectRemedies/saveEdit structure.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -v`
Expected: PASS (the whole file).

- [ ] **Step 5: Commit**

```bash
git add static/console-biofield-reveals.html tests/test_biofield_layers.py
git commit -m "feat: console biofield-reveals layer editor + approved section"
```

---

## Verification

- Per task: the named `pytest` target passes (doppler + venv).
- Full sweep after Task 6: `tests/test_biofield_layers.py` all green; plus `tests/test_biofield_trial.py tests/test_biofield_cart.py -v` (the derived flat `remedies` keeps #4b/#4c behavior-identical).
- Final Opus whole-branch review (focus: the per-layer guardrail drops a non-catalog remedy from its layer while keeping the layer + recording the drop; the derived flat `remedies` exactly equals the surviving layer remedies so `_biofield_visible_slugs`/#4b/#4c are unchanged; the payload emits titles+summaries ALWAYS and never emits a withheld remedy's slug/name/meaning (anti-bypass); re-editability works on approved reveals while the matcher `upsert` still only updates un-approved; remedies-only pushes wrap into titleless layers; XSS-safe front-ends; no emoji/em-dash).
- Manual visual pass (live, console-gated + reveal): the console shows pending + approved with editable titled layers + the dropped notice; an approved reveal is editable; the member reveal shows titled layers with summaries always and remedies gated; the #4c order bar still works.
- Ship via PR + merge to `main` (auto-deploys). The console change is live immediately; the member layer render only appears for reveals that have layered approved drafts (still effectively dark until the matcher pushes layered drafts and Glen approves). Gentle probe per the warm-up rule; update memory.

## Build order
Task 1 (store) -> Task 2 (ingest) -> Task 3 (payload) -> Task 4 (member render) -> Task 5 (console action + endpoint) -> Task 6 (console front-end). Tasks 3-6 depend on Tasks 1-2; Task 4 on Task 3; Task 6 on Task 5. The matcher must start pushing `layers` to populate real layered reveals; until then pushes wrap into titleless layers Glen can name.
