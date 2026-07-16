# Stress-vocab Dropdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Glen add a stress to a biofield test by typing it into a searchable dropdown on the Biofield Intake authoring page — balanced on a chosen layer, or as an active stress — with brand-new terms persisted to a reusable custom vocabulary.

**Architecture:** A new `custom_stress_vocab` table in `chat_log.db` is UNIONed into the existing `stress_vocab()` autocomplete so typed-in terms survive FMP re-imports and show for every client. One new endpoint `POST /author/<test_id>/stress/add` creates/merges the stress onto the test and, when a layer is given, balances it via that layer's remedies (reusing the existing `cover_stress` path). Two plain `<input list=vocab>` typeaheads in the stress panel call it.

**Tech Stack:** Python 3 (Flask app `biofield_local_app.py`), SQLite (`chat_log.db`), server-rendered HTML strings + vanilla JS, pytest.

## Global Constraints

- Authoring app is **local-only** (`:8011`); no prod sync of the custom vocab.
- The vocabulary source `fmp_snap_client_active_main_stress` is a **read-only FMP mirror** — never write custom terms into it; they must live in `custom_stress_vocab`.
- `add_stress` returns a **bool** (inserted vs merged) and is used as a counter elsewhere — do NOT change its signature/return.
- Run **focused** test files only (`pytest tests/<file>.py`), never the bare full suite (it can send live email). These tests construct the app directly and need no Doppler.
- Test IDs use the `"a<N>"` form (e.g. `"a5"`); `_num` strips the leading `a`.
- `test_id` chain rows live in `biofield_auth_chain` (has columns `id, test_id, layer, remedy, …`); stresses in `biofield_auth_stress` (`id, test_id, code, label, source, balance, manual_balanced, …`).

---

### Task 1: Custom vocab table + autocomplete union

**Files:**
- Modify: `dashboard/biofield_stress.py` (add `init_custom_vocab`, `add_custom_vocab`, `vocab_has`)
- Modify: `dashboard/biofield_authoring.py:511-520` (`stress_vocab` → union custom terms)
- Test: `tests/test_stress_custom_vocab.py` (create)

**Interfaces:**
- Produces:
  - `init_custom_vocab(cx) -> None` — creates `custom_stress_vocab(term TEXT PK, created_at TEXT, created_by TEXT)`.
  - `add_custom_vocab(cx, term: str) -> bool` — inserts a term (case-insensitive idempotent); True if newly inserted.
  - `vocab_has(cx, term: str) -> bool` — True if term already known (FMP snapshot OR custom), case-insensitive; True for blank.
  - `stress_vocab(cx, q="", limit=20) -> list[str]` — now FMP ∪ custom, deduped case-insensitively.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stress_custom_vocab.py`:

```python
import sqlite3
from dashboard.biofield_stress import init_custom_vocab, add_custom_vocab, vocab_has
from dashboard.biofield_authoring import stress_vocab


def _cx(tmp_path):
    return sqlite3.connect(str(tmp_path / "c.db"))


def _seed_fmp(cx, terms):
    cx.execute("CREATE TABLE fmp_snap_client_active_main_stress(id_pk INTEGER, main_stress TEXT)")
    cx.executemany("INSERT INTO fmp_snap_client_active_main_stress(main_stress) VALUES(?)",
                   [(t,) for t in terms])
    cx.commit()


def test_add_custom_vocab_idempotent_case_insensitive(tmp_path):
    cx = _cx(tmp_path)
    assert add_custom_vocab(cx, "Geopathic Stress") is True
    assert add_custom_vocab(cx, "  geopathic stress ") is False   # case/space dup
    assert cx.execute("SELECT COUNT(*) FROM custom_stress_vocab").fetchone()[0] == 1


def test_stress_vocab_unions_custom(tmp_path):
    cx = _cx(tmp_path)
    _seed_fmp(cx, ["Liver Congestion", "Adrenal Fatigue"])
    add_custom_vocab(cx, "Geopathic Stress")
    assert "Geopathic Stress" in stress_vocab(cx, "geo")
    assert "Liver Congestion" in stress_vocab(cx, "liver")


def test_stress_vocab_dedupes_across_sources(tmp_path):
    cx = _cx(tmp_path)
    _seed_fmp(cx, ["Liver Congestion"])
    add_custom_vocab(cx, "liver congestion")                      # same term, diff case
    got = stress_vocab(cx, "liver")
    assert sum(1 for t in got if t.lower() == "liver congestion") == 1


def test_vocab_has_across_sources(tmp_path):
    cx = _cx(tmp_path)
    _seed_fmp(cx, ["Liver Congestion"])
    add_custom_vocab(cx, "Geopathic Stress")
    assert vocab_has(cx, "liver congestion") is True             # FMP
    assert vocab_has(cx, "GEOPATHIC STRESS") is True             # custom
    assert vocab_has(cx, "Nonexistent Term") is False


def test_custom_vocab_survives_fmp_reimport(tmp_path):
    cx = _cx(tmp_path)
    _seed_fmp(cx, ["Liver Congestion"])
    add_custom_vocab(cx, "Geopathic Stress")
    # Simulate an FMP snapshot re-import: wipe + rewrite the FMP table.
    cx.execute("DELETE FROM fmp_snap_client_active_main_stress")
    cx.execute("INSERT INTO fmp_snap_client_active_main_stress(main_stress) VALUES('Adrenal Fatigue')")
    cx.commit()
    assert "Geopathic Stress" in stress_vocab(cx, "geo")         # custom intact
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_stress_custom_vocab.py -v`
Expected: FAIL — `ImportError: cannot import name 'init_custom_vocab'` (and `add_custom_vocab`, `vocab_has`).

- [ ] **Step 3: Add the vocab helpers to `dashboard/biofield_stress.py`**

Add these functions (near the other module-level helpers, after `init_stress_tables`):

```python
def init_custom_vocab(cx):
    """Durable, reusable stress terms Glen coins in the picker. Kept separate from the
    FMP snapshot (which is overwritten on every re-import)."""
    cx.execute("""CREATE TABLE IF NOT EXISTS custom_stress_vocab(
        term       TEXT PRIMARY KEY,
        created_at TEXT,
        created_by TEXT DEFAULT 'glen')""")
    cx.commit()


def _table_exists(cx, name):
    return cx.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                      (name,)).fetchone() is not None


def add_custom_vocab(cx, term):
    """Persist a stress term to the reusable custom vocabulary. Idempotent and
    case-insensitive. Returns True if a new row was inserted."""
    init_custom_vocab(cx)
    t = (term or "").strip()
    if not t:
        return False
    if cx.execute("SELECT 1 FROM custom_stress_vocab WHERE LOWER(term)=LOWER(?)", (t,)).fetchone():
        return False
    cx.execute("INSERT INTO custom_stress_vocab(term,created_at,created_by) VALUES(?,?,?)",
               (t, _now(), "glen"))
    cx.commit()
    return True


def vocab_has(cx, term):
    """True if term is already a known stress vocabulary term — in the FMP snapshot
    or the custom table (case-insensitive). Blank counts as known (never persist blank)."""
    t = (term or "").strip()
    if not t:
        return True
    if _table_exists(cx, "fmp_snap_client_active_main_stress") and cx.execute(
            "SELECT 1 FROM fmp_snap_client_active_main_stress "
            "WHERE LOWER(TRIM(main_stress))=LOWER(?) LIMIT 1", (t,)).fetchone():
        return True
    if _table_exists(cx, "custom_stress_vocab") and cx.execute(
            "SELECT 1 FROM custom_stress_vocab WHERE LOWER(term)=LOWER(?) LIMIT 1",
            (t,)).fetchone():
        return True
    return False
```

- [ ] **Step 4: Union the custom terms in `dashboard/biofield_authoring.py`**

Replace `stress_vocab` (currently at lines 511-520):

```python
def stress_vocab(cx, q="", limit=20):
    """Stress-factor terms for autocomplete: FMP snapshot terms UNION custom
    (glen-added) terms, deduped case-insensitively, filtered by q."""
    like = f"%{(q or '').strip()}%"
    have_fmp = _has(cx, "fmp_snap_client_active_main_stress")
    have_custom = _has(cx, "custom_stress_vocab")
    if not have_fmp and not have_custom:
        return []
    parts, params = [], []
    if have_fmp:
        parts.append("SELECT main_stress AS term FROM fmp_snap_client_active_main_stress "
                     "WHERE TRIM(COALESCE(main_stress,''))<>'' AND main_stress LIKE ?")
        params.append(like)
    if have_custom:
        parts.append("SELECT term FROM custom_stress_vocab "
                     "WHERE TRIM(COALESCE(term,''))<>'' AND term LIKE ?")
        params.append(like)
    sql = ("SELECT term FROM (" + " UNION ".join(parts) + ") "
           "GROUP BY LOWER(term) ORDER BY term LIMIT ?")
    params.append(limit)
    return [r[0] for r in cx.execute(sql, params).fetchall()]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_stress_custom_vocab.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Guard against regressions in the existing vocab test**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_biofield_authoring.py -v`
Expected: PASS (the union is a superset; existing FMP-only assertions still hold).

- [ ] **Step 7: Commit**

```bash
cd ~/deploy-chat && git add dashboard/biofield_stress.py dashboard/biofield_authoring.py tests/test_stress_custom_vocab.py
git commit -m "feat(biofield): custom stress vocabulary unioned into autocomplete"
```

---

### Task 2: `stress/add` endpoint + helpers

**Files:**
- Modify: `dashboard/biofield_stress.py` (add `stress_id_for`, `layer_chain_rids`)
- Modify: `biofield_local_app.py` (add route `POST /author/<test_id>/stress/add`, near the other `stress/<sid>/…` routes ~line 1377)
- Test: `tests/test_biofield_stress_add_route.py` (create)

**Interfaces:**
- Consumes (from Task 1): `add_custom_vocab`, `vocab_has`.
- Consumes (existing): `add_stress(cx, tid, label, *, source, balance) -> bool`, `cover_stress(cx, tid, stress_id, rids) -> str|None`, `set_manual_balanced(cx, tid, stress_id, value)`, `resolve_stress_name(cx, spoken, cutoff=0.82) -> str`.
- Produces:
  - `stress_id_for(cx, tid, label) -> int|None` — id of the test's stress matching label's normalized code.
  - `layer_chain_rids(cx, tid, layer) -> list[int]` — remedy-bearing chain-row ids on that layer.
  - Route `POST /author/<test_id>/stress/add` body `{label: str, layer?: int}` → JSON `{ok: bool, sid: int|None, label: str, layer: int|None}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_biofield_stress_add_route.py`:

```python
import sqlite3
import pytest
from biofield_local_app import create_app


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)


_NONE = {"status": "none", "found": False, "findings": [], "days_ago": None, "fresh": False}


def _app(db):
    return create_app(db, scan_lookup=lambda e: _NONE)


def _new(client):
    return client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]


def _stress_state(client, tid):
    return client.get(f"/author/{tid}/stresses").get_json()["data"]


def test_add_active_stress_unassigned(tmp_path):
    client = _app(str(tmp_path / "c.db")).test_client()
    tid = _new(client)
    j = client.post(f"/author/{tid}/stress/add", json={"label": "Geopathic Stress"}).get_json()
    assert j["ok"] and isinstance(j["sid"], int) and j["layer"] is None
    s = _stress_state(client, tid)
    active = {x["label"] for x in s["active"]}
    assert "Geopathic Stress" in active


def test_add_new_term_persisted_to_vocab(tmp_path):
    db = str(tmp_path / "c.db")
    client = _app(db).test_client()
    tid = _new(client)
    client.post(f"/author/{tid}/stress/add", json={"label": "Scalar Interference"})
    cx = sqlite3.connect(db)
    assert cx.execute("SELECT 1 FROM custom_stress_vocab WHERE LOWER(term)='scalar interference'").fetchone()


def test_add_balanced_on_layer(tmp_path):
    client = _app(str(tmp_path / "c.db")).test_client()
    tid = _new(client)
    # Give layer 1 a remedy row so the stress can be balanced by it.
    client.post(f"/author/{tid}/row", json={"layer": 1, "head": "Liver", "remedy": "Liver Support"})
    j = client.post(f"/author/{tid}/stress/add",
                    json={"label": "Liver Congestion", "layer": 1}).get_json()
    assert j["ok"] and j["layer"] == 1
    s = _stress_state(client, tid)
    balanced = {x["label"] for x in s["balanced"]}
    assert "Liver Congestion" in balanced
    assert "Liver Congestion" not in {x["label"] for x in s["active"]}


def test_add_empty_label_rejected(tmp_path):
    client = _app(str(tmp_path / "c.db")).test_client()
    tid = _new(client)
    resp = client.post(f"/author/{tid}/stress/add", json={"label": "   "})
    assert resp.status_code == 400
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_biofield_stress_add_route.py -v`
Expected: FAIL — 404 on `/author/<tid>/stress/add` (route not defined) → assertions error.

- [ ] **Step 3: Add the helpers to `dashboard/biofield_stress.py`**

```python
def stress_id_for(cx, tid, label):
    """id of the test's stress whose normalized code matches label, or None."""
    init_stress_tables(cx)
    n = _norm(label)
    if not n:
        return None
    r = cx.execute("SELECT id FROM biofield_auth_stress WHERE test_id=? AND code=? "
                   "ORDER BY id LIMIT 1", (_num(tid), n)).fetchone()
    return r[0] if r else None


def layer_chain_rids(cx, tid, layer):
    """Remedy-bearing chain-row ids on a given layer of a test (inputs to cover_stress)."""
    try:
        ln = int(layer)
    except (TypeError, ValueError):
        return []
    rows = cx.execute("SELECT id FROM biofield_auth_chain "
                      "WHERE test_id=? AND layer=? AND TRIM(COALESCE(remedy,''))<>''",
                      (_num(tid), ln)).fetchall()
    return [r[0] for r in rows]
```

- [ ] **Step 4: Add the route to `biofield_local_app.py`**

Insert immediately after the `author_stress_cover` route (the `POST /author/<test_id>/stress/<int:sid>/cover` handler, ~line 1377-1382):

```python
    @app.route("/author/<test_id>/stress/add", methods=["POST"])
    def author_stress_add(test_id):
        from dashboard import biofield_stress as _st
        from dashboard.biofield_authoring import resolve_stress_name
        d = request.get_json(silent=True) or {}
        raw = (d.get("label") or "").strip()
        if not raw:
            return {"ok": False, "error": "empty label"}, 400
        layer = d.get("layer")
        with sqlite3.connect(db_path) as cx:
            label = resolve_stress_name(cx, raw)
            if not _st.vocab_has(cx, label):
                _st.add_custom_vocab(cx, label)
            _st.add_stress(cx, test_id, label, source="manual", balance="required")
            sid = _st.stress_id_for(cx, test_id, label)
            balanced_layer = None
            if sid is not None and layer is not None:
                rids = _st.layer_chain_rids(cx, test_id, layer)
                if rids:
                    _st.cover_stress(cx, test_id, sid, rids)
                else:
                    _st.set_manual_balanced(cx, test_id, sid, True)
                balanced_layer = int(layer)
        return {"ok": sid is not None, "sid": sid, "label": label, "layer": balanced_layer}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_biofield_stress_add_route.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
cd ~/deploy-chat && git add dashboard/biofield_stress.py biofield_local_app.py tests/test_biofield_stress_add_route.py
git commit -m "feat(biofield): POST /author/<id>/stress/add — typed stress, balanced-on-layer or active"
```

---

### Task 3: Two typeahead inputs in the stress panel

**Files:**
- Modify: `dashboard/biofield_report_html.py` (`render_stress_panel` by_layer branch: per-layer + active inputs; add `addStress()` JS near `balanceStress` ~line 456)
- Test: `tests/test_stress_add_inputs.py` (create)

**Interfaces:**
- Consumes (from Task 2): route `POST /author/<test_id>/stress/add`.
- Produces: per-layer `add balanced stress…` input calling `addStress(value, <layer>)`; panel-bottom `add active stress…` input calling `addStress(value, null)`; JS `addStress(label, layer)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stress_add_inputs.py`:

```python
from dashboard.biofield_report_html import render_stress_panel


def _by_layer_data():
    return {"by_layer": [{"layer": 2, "head": "Liver", "remedy": "Liver Support",
                          "remedies": ["Liver Support"], "stresses": []}],
            "unassigned": []}


def test_per_layer_add_input_present_with_layer_number():
    html = render_stress_panel(_by_layer_data())
    assert "add balanced stress" in html
    assert "addStress(this.value,2)" in html


def test_active_add_input_present():
    html = render_stress_panel(_by_layer_data())
    assert "add active stress" in html
    assert "addStress(this.value,null)" in html
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_stress_add_inputs.py -v`
Expected: FAIL — `assert 'add balanced stress' in html` fails (inputs not rendered yet).

- [ ] **Step 3: Add the per-layer input in `render_stress_panel`**

In the `by_layer` branch, find the block that appends each layer (currently ends `... + "</div>" + body)`). Change it to also append a per-layer add input. Replace:

```python
            parts.append(f"<div class=food style='font-weight:600;margin-top:6px'>"
                         f"Layer {_e(str(L.get('layer')))}"
                         + (f" <span style='font-weight:400'>&mdash; {sub}</span>" if sub else "")
                         + "</div>" + body)
```

with:

```python
            add_in = (f"<input class=stress-add list=vocab placeholder='add balanced stress…' "
                      f"onkeydown=\"if(event.key==='Enter'){{addStress(this.value,{int(L.get('layer'))});this.value=''}}\" "
                      f"style='width:100%;margin:2px 0 8px;font-size:12px'>")
            parts.append(f"<div class=food style='font-weight:600;margin-top:6px'>"
                         f"Layer {_e(str(L.get('layer')))}"
                         + (f" <span style='font-weight:400'>&mdash; {sub}</span>" if sub else "")
                         + "</div>" + body + add_in)
```

- [ ] **Step 4: Add the active-stress input at the panel bottom**

Still in the `by_layer` branch, find where `inner` is assembled (`inner = "".join(parts) or ...`). Immediately BEFORE that line, append the active-add control to `parts`:

```python
        parts.append("<div class=food style='font-weight:600;margin-top:8px'>Add active stress</div>"
                     "<input class=stress-add list=vocab placeholder='add active stress…' "
                     "onkeydown=\"if(event.key==='Enter'){addStress(this.value,null);this.value=''}\" "
                     "style='width:100%;margin:2px 0 6px;font-size:12px'>")
```

- [ ] **Step 5: Add the `addStress` JS helper**

In `dashboard/biofield_report_html.py`, right after the `balanceStress` definition (line ~456), add:

```javascript
async function addStress(label,layer){label=(label||'').trim();if(!label)return;
 astat('Adding stress…');
 const body=(layer==null?{label:label}:{label:label,layer:layer});
 const j=await post('/author/__TID__/stress/add',body);
 astat(j&&j.ok?'Stress added.':((j&&j.error)||'Add failed.'));loadStress()}
```

(Add it inside the same JS string block that defines `balanceStress`, matching its `async function …` style so `__TID__` is substituted at render.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_stress_add_inputs.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Guard the existing author-html render test**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_biofield_author_html.py -v`
Expected: PASS (inputs are additive).

- [ ] **Step 8: Commit**

```bash
cd ~/deploy-chat && git add dashboard/biofield_report_html.py tests/test_stress_add_inputs.py
git commit -m "feat(biofield): per-layer 'add balanced stress' + panel 'add active stress' typeaheads"
```

---

## Manual verification (after all tasks)

1. `bash /tmp/ri.sh` won't apply (that was the prior fix); instead pull this branch's merge and restart `:8011` the same way, or run the app locally from the worktree.
2. Open a test's authoring page, confirm: typing in a layer's "add balanced stress" box shows the vocab dropdown; picking or typing-new a term makes it appear **balanced under that layer** with its remedies; a brand-new term reappears in the dropdown on a *different* test (proves persistence); the panel-bottom "add active stress" box adds an unassigned active stress.

## Self-review notes (checked against the spec)

- Spec "custom_stress_vocab table + union" → Task 1. ✓
- Spec "POST /author/<id>/stress/add, three behaviors" → Task 2 (active / balanced-on-layer / novel-persist) + tests. ✓
- Spec "two typeahead inputs" → Task 3. ✓
- Spec "FMP re-import doesn't wipe custom terms" → Task 1 `test_custom_vocab_survives_fmp_reimport`. ✓
- Spec "balancing always via the cover call, never add_stress" → Task 2 route: `add_stress(..., balance="required")` then `cover_stress`/`set_manual_balanced`. ✓
- Edge case beyond spec: layer with no remedy rows → `set_manual_balanced(True)` fallback so "add as balanced" still holds. Documented in the route.
