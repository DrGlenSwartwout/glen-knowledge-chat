# Household + Caregiver Overlay (Address-Suggested Connections) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Suggest household connections between same-street-address clients on the CRM person card, and let an operator connect them as a household member and/or a directional caregiver, with consent handling correct for both dependents and capacitated adults.

**Architecture:** Household membership is the base link (CRM `households` grouping); "caregiver" is a directional overlay in `dashboard/household.py` carrying a relationship word. The word's membership in `DEPENDENT_RELATIONSHIPS` decides consent behavior. Two guards make operational (non-dependent) links safe. Detection gains a street-address signal; the person card gains a "Possible connections" block backed by two new endpoints.

**Tech Stack:** Python/Flask (`app.py`), SQLite (`LOG_DB`), `dashboard/household.py`, vanilla-JS console (`static/console.html`), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-17-household-caregiver-overlay-design.md`.
- `DEPENDENT_RELATIONSHIPS = {"child","pet","dependent","charge","caregiving-client"}` is the consent gate. New operational words: `partner`, `spouse`, `manages-account`.
- Dependent TOS coverage stays dark behind `DEPENDENT_TOS_ENABLED`; this work must be correct when it flips.
- No auto-connect: address match only suggests; an operator confirms.
- deploy-chat has no CI — merge = deploy. Work stays on branch `sess/10b1fd25`; Glen makes the merge call.
- App-importing tests **silently skip under bare pytest** — run them with `doppler run -- python -m pytest ...`. Pure-module tests (no `import app`) run under bare `python -m pytest`.
- User-facing copy follows Glen's rules: no em dashes, no ALL CAPS.

---

### Task 1: Guard 1 — dependent-only TOS coverage

**Files:**
- Modify: `dashboard/household.py:139-143` (`caregivers_for`)
- Modify: `app.py:16879-16886` (`_portal_tos_agreed` coverage loop)
- Test: `tests/test_household.py`, `tests/test_dependent_tos_overlay.py` (new)

**Interfaces:**
- Produces: `household.caregivers_for(cx, member_email) -> list[{"primary_email","share_consent","relationship"}]` (adds `relationship`).

- [ ] **Step 1: Write the failing module test**

In `tests/test_household.py` add:

```python
def test_caregivers_for_includes_relationship():
    cx = _cx()
    h.add_member(cx, "cg@x.com", "dep@x.com", "Dep", "dependent")
    rows = h.caregivers_for(cx, "dep@x.com")
    assert rows[0]["primary_email"] == "cg@x.com"
    assert rows[0]["relationship"] == "dependent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-10b1fd25 && python -m pytest tests/test_household.py::test_caregivers_for_includes_relationship -v`
Expected: FAIL with `KeyError: 'relationship'`.

- [ ] **Step 3: Add relationship to `caregivers_for`**

Replace `dashboard/household.py` `caregivers_for` body:

```python
def caregivers_for(cx, member_email):
    rows = cx.execute(
        "SELECT primary_email, share_consent, relationship FROM household_members "
        "WHERE member_email=? ORDER BY created_at, id", (_norm(member_email),)).fetchall()
    return [{"primary_email": r[0], "share_consent": int(r[1] if r[1] is not None else 1),
             "relationship": r[2] or ""} for r in rows]
```

- [ ] **Step 4: Run module test to verify it passes**

Run: `python -m pytest tests/test_household.py::test_caregivers_for_includes_relationship -v`
Expected: PASS.

- [ ] **Step 5: Write the failing coverage test (app-level)**

Create `tests/test_dependent_tos_overlay.py`:

```python
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _make_member(appmod, email):
    # Mark `email` as a paid member so is_member() is True for the caregiver.
    import dashboard.memberships as m  # membership check source
    # Fallback: monkeypatch is_member at the app module if no direct seed exists.


def test_operational_caregiver_gets_no_tos_coverage(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx)
        # caregiver is a member; cared-for adult is NOT and did not agree
        h.add_member(cx, "cg@x.com", "adult@x.com", "Partner", "partner")
        h.set_share_consent(cx, "cg@x.com", "adult@x.com", 1)
    monkeypatch.setattr(appmod, "is_member",
                        lambda email=None, **k: (email or "").lower() == "cg@x.com")
    # operational relationship must NOT grant coverage
    assert appmod._portal_tos_agreed("adult@x.com") is False


def test_dependent_caregiver_grants_tos_coverage(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx)
        h.add_member(cx, "cg@x.com", "kid@x.com", "Kid", "dependent")
    monkeypatch.setattr(appmod, "is_member",
                        lambda email=None, **k: (email or "").lower() == "cg@x.com")
    assert appmod._portal_tos_agreed("kid@x.com") is True
```

- [ ] **Step 6: Run to verify the operational test fails**

Run: `doppler run -- python -m pytest tests/test_dependent_tos_overlay.py -v`
Expected: `test_operational_caregiver_gets_no_tos_coverage` FAILS (coverage granted today), `test_dependent...` PASSES.

- [ ] **Step 7: Add the dependent filter to `_portal_tos_agreed`**

In `app.py`, inside the `for cg in caregivers:` loop, change the condition:

```python
        for cg in caregivers:
            # is_member opens its own _db_lock connection, so do not nest it inside cx.
            if (_hh.is_dependent(cg["relationship"])
                    and cg["share_consent"] and is_member(email=cg["primary_email"])):
                return True
```

- [ ] **Step 8: Run both tests to verify they pass**

Run: `doppler run -- python -m pytest tests/test_dependent_tos_overlay.py tests/test_household.py -v`
Expected: PASS.

- [ ] **Step 9: Mutation-check the guard**

Temporarily delete `_hh.is_dependent(cg["relationship"]) and ` from the condition, re-run Step 8, confirm `test_operational...` goes RED, then restore.

- [ ] **Step 10: Commit**

```bash
git add dashboard/household.py app.py tests/test_household.py tests/test_dependent_tos_overlay.py
git commit -m "guard: TOS coverage counts only dependent-relationship caregivers"
```

---

### Task 2: Guard 2 + consent columns — consent-class-aware `add_member`

**Files:**
- Modify: `dashboard/household.py:29-83` (`init_household_tables`, `add_member`) + new helpers
- Test: `tests/test_household_consent.py` (new)

**Interfaces:**
- Produces:
  - `household.add_member(cx, primary_email, member_email, label="", relationship="", consent_basis=None, consent_by="") -> bool` — dependent word → `share_consent=1, consent_basis="caregiver-authority"`; operational word → `share_consent=0, consent_basis=""` unless `consent_basis in ("verbal","written")` given, then `share_consent=1`.
  - `household.confirm_consent(cx, primary_email, member_email) -> bool` — sets `share_consent=1, consent_basis="portal-confirmed", consent_confirmed_at=now`; idempotent; never downgrades.
  - `household.consent_state(cx, primary_email, member_email) -> {"share_consent","consent_basis","consent_confirmed_at"}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_household_consent.py`:

```python
import sqlite3
from dashboard import household as h


def _cx():
    cx = sqlite3.connect(":memory:")
    h.init_household_tables(cx)
    return cx


def test_dependent_defaults_shared_and_authority():
    cx = _cx()
    h.add_member(cx, "cg@x.com", "kid@x.com", "Kid", "dependent")
    st = h.consent_state(cx, "cg@x.com", "kid@x.com")
    assert st["share_consent"] == 1 and st["consent_basis"] == "caregiver-authority"


def test_operational_defaults_dark_and_pending():
    cx = _cx()
    h.add_member(cx, "cg@x.com", "adult@x.com", "Partner", "partner")
    st = h.consent_state(cx, "cg@x.com", "adult@x.com")
    assert st["share_consent"] == 0 and st["consent_basis"] == ""


def test_operational_with_verbal_basis_is_active_but_unconfirmed():
    cx = _cx()
    h.add_member(cx, "cg@x.com", "adult@x.com", "Partner", "partner",
                 consent_basis="verbal", consent_by="rae")
    st = h.consent_state(cx, "cg@x.com", "adult@x.com")
    assert st["share_consent"] == 1 and st["consent_basis"] == "verbal"
    assert st["consent_confirmed_at"] in (None, "")


def test_confirm_consent_hard_and_idempotent():
    cx = _cx()
    h.add_member(cx, "cg@x.com", "adult@x.com", "Partner", "partner")
    assert h.confirm_consent(cx, "cg@x.com", "adult@x.com") is True
    st = h.consent_state(cx, "cg@x.com", "adult@x.com")
    assert st["share_consent"] == 1 and st["consent_basis"] == "portal-confirmed"
    assert st["consent_confirmed_at"]
    first = st["consent_confirmed_at"]
    h.confirm_consent(cx, "cg@x.com", "adult@x.com")  # idempotent, no downgrade
    st2 = h.consent_state(cx, "cg@x.com", "adult@x.com")
    assert st2["consent_confirmed_at"] == first and st2["share_consent"] == 1
```

- [ ] **Step 2: Run to verify failures**

Run: `python -m pytest tests/test_household_consent.py -v`
Expected: FAIL (`consent_state`/`confirm_consent` undefined; operational default is 1 today).

- [ ] **Step 3: Add the consent columns**

In `dashboard/household.py` `init_household_tables`, after the `cc_enabled` ALTER block, add:

```python
    for _col, _default in (("consent_basis", "''"), ("consent_recorded_by", "''"),
                           ("consent_confirmed_at", "''")):
        try:
            cx.execute(f"ALTER TABLE household_members ADD COLUMN {_col} TEXT DEFAULT {_default}")
        except Exception:
            pass
    cx.commit()
```

- [ ] **Step 4: Make `add_member` consent-class-aware**

Replace `add_member`:

```python
def add_member(cx, primary_email, member_email, label="", relationship="",
               consent_basis=None, consent_by=""):
    p, m = _norm(primary_email), _norm(member_email)
    if not p or not m or p == m:
        return False
    if is_dependent(relationship):
        share, basis = 1, "caregiver-authority"
    elif consent_basis in ("verbal", "written"):
        share, basis = 1, consent_basis
    else:
        share, basis = 0, ""
    cx.execute(
        "INSERT OR IGNORE INTO household_members "
        "(primary_email, member_email, label, relationship, created_at, share_consent, "
        " cc_enabled, consent_basis, consent_recorded_by) VALUES (?,?,?,?,?,?,?,?,?)",
        (p, m, label or "", relationship or "", _now(), share,
         default_cc_for(relationship), basis, consent_by or ""))
    cx.commit()
    return True
```

- [ ] **Step 5: Add `confirm_consent` and `consent_state`**

Append to `dashboard/household.py`:

```python
def consent_state(cx, primary_email, member_email):
    r = cx.execute(
        "SELECT share_consent, consent_basis, consent_confirmed_at FROM household_members "
        "WHERE primary_email=? AND member_email=?",
        (_norm(primary_email), _norm(member_email))).fetchone()
    if not r:
        return {"share_consent": 0, "consent_basis": "", "consent_confirmed_at": ""}
    return {"share_consent": int(r[0] if r[0] is not None else 0),
            "consent_basis": r[1] or "", "consent_confirmed_at": r[2] or ""}


def confirm_consent(cx, primary_email, member_email):
    p, m = _norm(primary_email), _norm(member_email)
    row = cx.execute("SELECT consent_confirmed_at FROM household_members "
                     "WHERE primary_email=? AND member_email=?", (p, m)).fetchone()
    if not row:
        return False
    if row[0]:  # already hard-confirmed — idempotent, never downgrade
        return True
    cx.execute("UPDATE household_members SET share_consent=1, consent_basis='portal-confirmed', "
               "consent_confirmed_at=? WHERE primary_email=? AND member_email=?", (_now(), p, m))
    cx.commit()
    return True
```

- [ ] **Step 6: Run to verify passes**

Run: `python -m pytest tests/test_household_consent.py -v`
Expected: PASS.

- [ ] **Step 7: Regression — existing household tests still green**

Run: `python -m pytest tests/test_household.py -v`
Expected: PASS (dependent adds unchanged; `add_member` signature is backward-compatible).

- [ ] **Step 8: Commit**

```bash
git add dashboard/household.py tests/test_household_consent.py
git commit -m "feat: consent-class-aware caregiver links + hybrid consent (option C)"
```

---

### Task 3: Street-address detection signal

**Files:**
- Modify: `app.py:30176-30234` (`detect_household_candidates` — people SELECT + new signal)
- Test: `tests/test_household_street_signal.py` (new)

**Interfaces:**
- Consumes: `_candidate_dedup_key`, `_emit_signal` (internal to `detect_household_candidates`).
- Produces: candidate rows with `signal="shared-street-address"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_household_street_signal.py`:

```python
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _person(cx, email, first, last, address1="", zip=""):
    cx.execute("INSERT INTO people (email, first_name, last_name, address1, zip) "
               "VALUES (?,?,?,?,?)", (email, first, last, address1, zip))


def test_diff_surname_same_street_makes_candidate(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _person(cx, "a@x.com", "Ann", "Lee", "12 Palm St", "96720")
        _person(cx, "b@x.com", "Bo", "Reyes", " 12  Palm St ", "96720")  # dirty, diff surname
        cx.commit()
    appmod.detect_household_candidates()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sigs = [r[0] for r in cx.execute(
            "SELECT signal FROM household_candidates").fetchall()]
    assert "shared-street-address" in sigs


def test_empty_street_makes_no_street_candidate(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _person(cx, "a@x.com", "Ann", "Lee", "", "")
        _person(cx, "b@x.com", "Bo", "Reyes", "", "")
        cx.commit()
    appmod.detect_household_candidates()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sigs = [r[0] for r in cx.execute("SELECT signal FROM household_candidates").fetchall()]
    assert "shared-street-address" not in sigs
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -- python -m pytest tests/test_household_street_signal.py -v`
Expected: `test_diff_surname...` FAILS (no such signal).

- [ ] **Step 3: Add address1/zip to the people SELECT**

In `detect_household_candidates`, extend the SELECT to include street fields:

```python
        people = cx.execute("""
            SELECT id, LOWER(TRIM(email)) AS email_lc, LOWER(TRIM(last_name)) AS last_lc,
                   phone, LOWER(TRIM(city)) AS city_lc, LOWER(TRIM(state)) AS state_lc, tags,
                   LOWER(TRIM(address1)) AS addr1_lc, LOWER(TRIM(zip)) AS zip_lc
            FROM people
        """).fetchall()
```

- [ ] **Step 4: Add the street signal clustering + emit**

After the `by_addr` block (Signal 3), add:

```python
        # ── Signal 4: shared-street-address (no last-name requirement) ────────
        import re as _re
        def _norm_street(s):
            return _re.sub(r"\s+", " ", (s or "").strip().rstrip(".,")).strip()
        by_street = {}
        for p in people:
            street = _norm_street(p["addr1_lc"])
            if len(street) < 4 or not p["zip_lc"]:
                continue
            by_street.setdefault((street, p["zip_lc"]), []).append(p["id"])
```

Then add the emit call alongside the others:

```python
        _emit_signal("shared-address-lastname", by_addr)
        _emit_signal("shared-street-address",   by_street)
```

- [ ] **Step 5: Run to verify both tests pass**

Run: `doppler run -- python -m pytest tests/test_household_street_signal.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_household_street_signal.py
git commit -m "feat: shared-street-address household detection signal"
```

---

### Task 4: `GET /api/people/<id>/household-suggestions`

**Files:**
- Modify: `app.py` (add route near `get_person_household`, ~line 29743)
- Test: `tests/test_household_suggestions_api.py` (new)

**Interfaces:**
- Produces: `GET /api/people/<id>/household-suggestions -> {"suggestions": [{"person_id","email","name","address1","already_in_household_together":bool,"existing_caregiver_link":{"direction","relationship"}|None,"dismissed":bool}]}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_household_suggestions_api.py`:

```python
import importlib, sqlite3, sys, json
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _person(cx, email, first, last, address1="", zip=""):
    cx.execute("INSERT INTO people (email, first_name, last_name, address1, zip) "
               "VALUES (?,?,?,?,?)", (email, first, last, address1, zip))
    return cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()[0]


def test_suggestions_returns_same_street_other(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        a = _person(cx, "a@x.com", "Ann", "Lee", "12 Palm St", "96720")
        b = _person(cx, "b@x.com", "Bo", "Reyes", "12 Palm St", "96720")
        cx.commit()
    c = appmod.app.test_client()
    r = c.get(f"/api/people/{a}/household-suggestions", headers={"X-Console-Key": ""})
    body = r.get_json()
    ids = [s["person_id"] for s in body["suggestions"]]
    assert b in ids and a not in ids
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -- python -m pytest tests/test_household_suggestions_api.py -v`
Expected: FAIL (404 — route missing).

- [ ] **Step 3: Add the route**

In `app.py`, after `get_person_household`, add:

```python
@app.route("/api/people/<int:person_id>/household-suggestions", methods=["GET"])
def person_household_suggestions(person_id):
    auth_err = _check_console_or_scoped_auth()
    if auth_err: return auth_err
    import re as _re
    from dashboard import household as _hh
    def _norm_street(s): return _re.sub(r"\s+", " ", (s or "").strip().rstrip(".,")).strip().lower()
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _hh.init_household_tables(cx)
        me = cx.execute("SELECT id, email, address1, zip FROM people WHERE id=?",
                        (person_id,)).fetchone()
        if not me:
            return jsonify({"error": "person not found"}), 404
        street, zipc = _norm_street(me["address1"]), (me["zip"] or "").strip().lower()
        if len(street) < 4 or not zipc:
            return jsonify({"suggestions": []})
        rows = cx.execute("SELECT id, email, first_name, last_name, address1, zip FROM people "
                          "WHERE id != ?", (person_id,)).fetchall()
        out = []
        for r in rows:
            if _norm_street(r["address1"]) != street or (r["zip"] or "").strip().lower() != zipc:
                continue
            link = None
            if _hh.can_view(cx, me["email"], r["email"]):
                link = {"direction": "cares-for-other", "relationship": ""}
            elif _hh.can_view(cx, r["email"], me["email"]):
                link = {"direction": "other-cares-for-this", "relationship": ""}
            out.append({
                "person_id": r["id"], "email": r["email"],
                "name": f'{r["first_name"] or ""} {r["last_name"] or ""}'.strip(),
                "address1": r["address1"] or "",
                "already_in_household_together": _hh.same_household(cx, me["email"], r["email"]),
                "existing_caregiver_link": link,
                "dismissed": False,
            })
    return jsonify({"suggestions": out})
```

- [ ] **Step 4: Run to verify it passes**

Run: `doppler run -- python -m pytest tests/test_household_suggestions_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_household_suggestions_api.py
git commit -m "feat: person household-suggestions endpoint (same-street matches)"
```

---

### Task 5: `POST /api/people/<id>/connect` — routing + naming

**Files:**
- Modify: `app.py` (add route after Task 4's route; small naming helper)
- Test: `tests/test_household_connect_api.py` (new)

**Interfaces:**
- Consumes: `create_household()` (internal test-request-context call, as `confirm_household_candidate` does), `household.add_member(..., consent_basis, consent_by)`.
- Produces: `POST /api/people/<id>/connect` with body `{"other_person_id", "mode": "member"|"caregiver"|"dismiss", "caregiver_person_id"?, "cared_for_person_id"?, "relationship"?, "consent"?: {"method":"portal"|"verbal"|"written"}}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_household_connect_api.py`:

```python
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
        # GHL push is best-effort; stub it so tests don't hit the network.
        monkeypatch.setattr(appmod, "_push_household_tags_to_ghl",
                            lambda *a, **k: (True, None), raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _person(cx, email, first, last):
    cx.execute("INSERT INTO people (email, first_name, last_name) VALUES (?,?,?)",
               (email, first, last))
    return cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()[0]


def _seed(appmod):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        a = _person(cx, "a@x.com", "Ann", "Lee")
        b = _person(cx, "b@x.com", "Bo", "Reyes")
        cx.commit()
    return a, b


def test_connect_member_creates_grouping_only(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    a, b = _seed(appmod)
    c = appmod.app.test_client()
    r = c.post(f"/api/people/{a}/connect",
               json={"other_person_id": b, "mode": "member"},
               headers={"X-Console-Key": ""})
    assert r.get_json()["ok"] is True
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert h.members_for(cx, "a@x.com") == []  # no caregiver link
        assert appmod._person_household_slug(cx, a)  # grouped


def test_connect_operational_caregiver_defaults_dark(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    a, b = _seed(appmod)
    c = appmod.app.test_client()
    r = c.post(f"/api/people/{a}/connect",
               json={"other_person_id": b, "mode": "caregiver",
                     "caregiver_person_id": a, "cared_for_person_id": b,
                     "relationship": "partner", "consent": {"method": "portal"}},
               headers={"X-Console-Key": ""})
    assert r.get_json()["ok"] is True
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        st = h.consent_state(cx, "a@x.com", "b@x.com")
        assert st["share_consent"] == 0  # dark until confirmed
        assert appmod._person_household_slug(cx, a)  # also grouped


def test_connect_operational_caregiver_verbal_is_active(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    a, b = _seed(appmod)
    c = appmod.app.test_client()
    c.post(f"/api/people/{a}/connect",
           json={"other_person_id": b, "mode": "caregiver",
                 "caregiver_person_id": a, "cared_for_person_id": b,
                 "relationship": "partner", "consent": {"method": "verbal"}},
           headers={"X-Console-Key": ""})
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        st = h.consent_state(cx, "a@x.com", "b@x.com")
        assert st["share_consent"] == 1 and st["consent_basis"] == "verbal"
```

- [ ] **Step 2: Run to verify failures**

Run: `doppler run -- python -m pytest tests/test_household_connect_api.py -v`
Expected: FAIL (route missing).

- [ ] **Step 3: Add a 2-person naming helper**

In `app.py`, near `_household_slug` (~line 29243), add:

```python
def _pair_household_name(cx, head_id, other_id):
    """Household display name for a 2-person connect. '{Last} Household' when
    surnames match; '{HeadLast} / {OtherLast} Household' when they differ."""
    def _last(pid):
        r = cx.execute("SELECT last_name FROM people WHERE id=?", (pid,)).fetchone()
        return ((r[0] if r else "") or "").strip()
    hl, ol = _last(head_id), _last(other_id)
    if hl and ol and hl.lower() != ol.lower():
        return f"{hl} / {ol} Household"
    return f"{hl or ol or 'New'} Household"
```

- [ ] **Step 4: Add the connect route**

In `app.py`, after `person_household_suggestions`, add:

```python
@app.route("/api/people/<int:person_id>/connect", methods=["POST"])
def person_connect(person_id):
    auth_err = _check_console_or_scoped_auth()
    if auth_err: return auth_err
    from dashboard import household as _hh
    body = request.get_json(force=True) or {}
    mode = (body.get("mode") or "").strip()
    other_id = body.get("other_person_id")
    if mode not in ("member", "caregiver", "dismiss") or not other_id:
        return jsonify({"error": "mode + other_person_id required"}), 400

    if mode == "dismiss":
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute("INSERT INTO household_candidates (detected_at, signal, person_ids, "
                       "status, resolved_at, resolved_by) VALUES (?,?,?,?,?,?)",
                       (datetime.now(timezone.utc).isoformat(), "manual-dismiss",
                        json.dumps(sorted([int(person_id), int(other_id)])),
                        "dismissed", datetime.now(timezone.utc).isoformat(), "glen"))
            cx.commit()
        return jsonify({"ok": True, "dismissed": True})

    # Ensure a CRM household grouping exists for the pair (idempotent).
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        existing = _person_household_slug(cx, person_id) or _person_household_slug(cx, other_id)
        head_id = body.get("caregiver_person_id") or person_id  # caregiver heads the household
        name = _pair_household_name(cx, head_id, other_id if head_id == person_id else person_id)

    if existing:
        with app.test_request_context(f"/api/households/{existing}/members", method="POST",
                                      json={"person_id": other_id},
                                      headers={"X-Console-Key": CONSOLE_SECRET or ""}):
            add_household_member(existing)
    else:
        member_ids = sorted({int(person_id), int(other_id)})
        with app.test_request_context("/api/households", method="POST",
                                      json={"name": name, "head_person_id": head_id,
                                            "member_person_ids": member_ids},
                                      headers={"X-Console-Key": CONSOLE_SECRET or ""}):
            resp = create_household()
        if isinstance(resp, tuple) and resp[1] != 200:
            return resp

    if mode == "member":
        return jsonify({"ok": True, "mode": "member"})

    # mode == caregiver: layer the directional link on top of the grouping.
    cg_id = body.get("caregiver_person_id"); cf_id = body.get("cared_for_person_id")
    relationship = (body.get("relationship") or "dependent").strip().lower()
    if not cg_id or not cf_id:
        return jsonify({"error": "caregiver_person_id + cared_for_person_id required"}), 400
    method = ((body.get("consent") or {}).get("method") or "portal").strip()
    consent_basis = method if method in ("verbal", "written") else None
    with sqlite3.connect(LOG_DB) as cx:
        emails = {r[0]: r[1] for r in cx.execute(
            "SELECT id, email FROM people WHERE id IN (?,?)", (cg_id, cf_id)).fetchall()}
    cg_email, cf_email = emails.get(cg_id), emails.get(cf_id)
    if not cg_email or not cf_email:
        return jsonify({"error": "caregiver/cared-for email missing"}), 400
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _hh.init_household_tables(cx)
        _hh.add_member(cx, cg_email, cf_email, label="", relationship=relationship,
                       consent_basis=consent_basis, consent_by="console")
    return jsonify({"ok": True, "mode": "caregiver", "relationship": relationship,
                    "consent_basis": consent_basis or "pending"})
```

- [ ] **Step 5: Run to verify passes**

Run: `doppler run -- python -m pytest tests/test_household_connect_api.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_household_connect_api.py
git commit -m "feat: person connect endpoint (member/caregiver overlay/dismiss)"
```

---

### Task 6: CRM person card — "Possible connections" block

**Files:**
- Modify: `static/console.html` — `loadPersonDetail` (~2143), `renderPersonDetail` (~2155), add render + handlers
- Verify: headless Chrome render (per render-verify discipline)

**Interfaces:**
- Consumes: `GET /api/people/<id>/household-suggestions`, `POST /api/people/<id>/connect`.

- [ ] **Step 1: Fetch suggestions in `loadPersonDetail`**

Change the `Promise.all` in `loadPersonDetail` to add a third fetch:

```javascript
  const [p, h, s] = await Promise.all([
    fetch(`${BASE}/api/people/${id}`, { headers:{'X-Console-Key':consoleKey} }),
    fetch(`${BASE}/api/people/${id}/household`, { headers:{'X-Console-Key':consoleKey} }),
    fetch(`${BASE}/api/people/${id}/household-suggestions`, { headers:{'X-Console-Key':consoleKey} }),
  ].map(pr => pr.then(r => r.json())));
  renderPersonDetail(p, h.household, s.suggestions || []);
```

- [ ] **Step 2: Thread suggestions into `renderPersonDetail`**

Change the signature and add the block after `_renderHouseholdSection(p, household)`:

```javascript
function renderPersonDetail(p, household, suggestions) {
```

and in the returned template, immediately after `${_renderHouseholdSection(p, household)}`:

```javascript
      ${_renderConnectionSuggestions(p, suggestions || [])}
```

- [ ] **Step 3: Add the render + handlers**

Append near `_renderHouseholdSection`:

```javascript
function _renderConnectionSuggestions(person, suggestions) {
  const fresh = suggestions.filter(s => !s.already_in_household_together
                                      && !s.existing_caregiver_link && !s.dismissed);
  if (!fresh.length) return '';
  const rows = fresh.map(s => `
    <div class="conn-row" data-other="${s.person_id}" style="border-top:1px solid #eee;padding:8px 0">
      <div style="font-size:13px">Same address: <b>${_esc(s.name)}</b> (${_esc(s.email)})</div>
      <label><input type="radio" name="conn-${s.person_id}" value="member" checked> Household member</label>
      <label><input type="radio" name="conn-${s.person_id}" value="caregiver"> Caregiver</label>
      <label><input type="radio" name="conn-${s.person_id}" value="dismiss"> Do not connect</label>
      <div class="conn-cg" data-other="${s.person_id}" style="display:none;margin:6px 0 0 18px">
        <select class="conn-dir">
          <option value="this-cares">This person cares for ${_esc(s.name)}</option>
          <option value="other-cares">${_esc(s.name)} cares for this person</option>
        </select>
        <select class="conn-rel">
          <optgroup label="Dependent (Terms covered)">
            <option value="dependent">dependent</option><option value="child">child</option>
            <option value="charge">charge</option><option value="caregiving-client">caregiving-client</option>
            <option value="pet">pet</option>
          </optgroup>
          <optgroup label="Operational (own consent)">
            <option value="partner">partner</option><option value="spouse">spouse</option>
            <option value="manages-account">manages-account</option>
          </optgroup>
        </select>
        <select class="conn-consent">
          <option value="portal">Send portal confirmation</option>
          <option value="verbal">Record verbal consent</option>
          <option value="written">Record written consent</option>
        </select>
        <div style="font-size:11px;color:var(--hh-meta)">Dependents are covered by the caregiver's Terms. Operational links stay off until the cared-for adult confirms.</div>
      </div>
      <button class="conn-go" data-other="${s.person_id}" style="margin-top:6px;padding:5px 10px;background:#f5b87a;color:#0a0c12;border:none;border-radius:4px;font-size:12px;cursor:pointer">Confirm</button>
    </div>`).join('');
  setTimeout(_wireConnectionSuggestions, 0);
  return `<div class="detail-section"><h4>Possible connections</h4>${rows}</div>`;
}

function _wireConnectionSuggestions() {
  document.querySelectorAll('.conn-row').forEach(row => {
    const other = row.dataset.other;
    row.querySelectorAll(`input[name="conn-${other}"]`).forEach(radio => {
      radio.addEventListener('change', () => {
        const cg = row.querySelector('.conn-cg');
        cg.style.display = (radio.value === 'caregiver' && radio.checked) ? 'block' : cg.style.display;
        const picked = row.querySelector(`input[name="conn-${other}"]:checked`).value;
        cg.style.display = picked === 'caregiver' ? 'block' : 'none';
      });
    });
  });
  document.querySelectorAll('.conn-go').forEach(btn => {
    btn.addEventListener('click', () => _submitConnection(btn.dataset.other));
  });
}

async function _submitConnection(otherId) {
  const row = document.querySelector(`.conn-row[data-other="${otherId}"]`);
  const mode = row.querySelector(`input[name="conn-${otherId}"]:checked`).value;
  const selfId = _currentPersonId;  // set in loadPersonDetail
  const payload = { other_person_id: Number(otherId), mode };
  if (mode === 'caregiver') {
    const dir = row.querySelector('.conn-dir').value;
    const caregiver = dir === 'this-cares' ? selfId : Number(otherId);
    const caredFor  = dir === 'this-cares' ? Number(otherId) : selfId;
    payload.caregiver_person_id = caregiver;
    payload.cared_for_person_id = caredFor;
    payload.relationship = row.querySelector('.conn-rel').value;
    payload.consent = { method: row.querySelector('.conn-consent').value };
  }
  await fetch(`${BASE}/api/people/${selfId}/connect`, {
    method: 'POST', headers: {'Content-Type':'application/json','X-Console-Key':consoleKey},
    body: JSON.stringify(payload),
  });
  loadPersonDetail(selfId);
}
```

- [ ] **Step 4: Track the current person id**

In `loadPersonDetail(id)`, add near the top: `_currentPersonId = id;` and declare `let _currentPersonId = null;` near the other module-level `let` declarations.

- [ ] **Step 5: Render-verify in headless Chrome**

Start the app locally (`doppler run -- python app.py` or the project run skill), open `/console.html?key=...`, open a person who shares a street address with another, and confirm the "Possible connections" block renders, the Caregiver radio expands the direction/relationship/consent selects, and Confirm connects then refreshes the card (household section now shows the link). Capture a screenshot.

- [ ] **Step 6: Commit**

```bash
git add static/console.html
git commit -m "feat: possible-connections block on CRM person card"
```

---

## Self-Review notes

- **Spec coverage:** Core model (T5/T6), consent classes (T2), Guard 1 (T1), Guard 2 (T2), option C columns + confirm (T2), street signal (T3), suggestions endpoint (T4), connect routing + differing-surname naming + caregiver-as-head (T5), CRM card UI (T6). Portal-side confirmation *UI* and intake/order surfaces are out of scope per spec; `confirm_consent` (the endpoint's upgrade primitive) ships in T2 for the later portal screen to call.
- **Type consistency:** `add_member(..., consent_basis, consent_by)`, `consent_state`, `confirm_consent`, `caregivers_for[...]["relationship"]`, `person_connect` body keys, and `household-suggestions` shape are used identically across tasks.
- **Deferred:** the portal route that calls `confirm_consent` is a follow-up slice (spec "out of scope").
