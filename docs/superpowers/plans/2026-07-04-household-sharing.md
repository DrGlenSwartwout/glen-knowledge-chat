# Household Communication Sharing & Routing (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two-sided household consent + cc routing: a member permits/denies sharing their info+comms with a caregiver (gates the view and cc), a caregiver opts into cc per member, and report/invoice notifications cc the caregiver only when both switches are on.

**Architecture:** Two columns on the shipped `household_members` link (`share_consent`, `cc_enabled`) + model functions in `dashboard/household.py`; `can_view` gains a consent requirement; token-scoped portal endpoints let the member set consent and the caregiver set cc; the report/invoice send sites deliver a private caregiver copy when routing resolves one. Behind `HOUSEHOLD_SHARING_ENABLED`.

**Tech Stack:** Python 3, Flask, SQLite (`LOG_DB`), pytest, vanilla JS.

## Global Constraints

- **Two-switch rule:** a caregiver is cc'd about a member only when `share_consent=1 AND cc_enabled=1`. The member permission ALSO gates the view (`can_view` requires `share_consent=1`).
- **Defaults:** `share_consent` default `1` (member shared by default, revocable). `cc_enabled` default derived from relationship — dependents ON, adults OFF. Dependent relationships = `{"child","pet","dependent","charge","caregiving-client"}`; everything else = adult.
- **No regression:** `share_consent` defaults to 1, so `can_view` is unchanged for every existing/consented link. `same_household` (reassignment) is NEVER consent-gated.
- **CC delivery = a private, separate caregiver copy** (its own email via the same send helper, pointing the caregiver to their portal), NOT a shared `Cc:` header — this refines the spec's "Cc header" wording to avoid cross-exposing the member's and caregiver's email addresses. Best-effort: a cc failure never blocks the member's own send.
- **Scope:** cc only for report (`_send_reveal_link`) + invoice (`orders._send_invoice_exec`). No other notification types.
- **Behind `HOUSEHOLD_SHARING_ENABLED` (default OFF):** toggles hidden, endpoints inert, cc routing off. `share_consent` column still applies (default 1 = no change).
- Emails lowercased/stripped; portal endpoints token-scoped (member sets only own consent, caregiver only own cc); console gated by `_portal_console_ok()`.
- Tests via `doppler run -p remedy-match -c dev -- python3 -m pytest ...` (use `python3`). Do NOT `git stash`.

---

## File Structure

- **Modify** `dashboard/household.py` — two columns + backfill in `init_household_tables`; `is_dependent`/`default_cc_for`; `add_member` sets `cc_enabled`; `set_share_consent`/`set_cc_enabled`; `can_view` requires `share_consent`; `viewable_members_for`; `cc_recipients_for`; `caregivers_for`; `members_for` returns the two flags.
- **Modify** `app.py` — `_household_sharing_enabled()`; `POST /api/portal/<token>/share-consent`; `POST /api/portal/<token>/cc-pref`; switcher payload uses `viewable_members_for` + carries the member's caregiver(s) + per-member cc state; cc hook in `_send_reveal_link`; console-household GET/POST expose+set both flags.
- **Modify** `dashboard/orders.py` — cc hook in `_send_invoice_exec`.
- **Modify** `static/client-portal.html` — member "Sharing" toggle + caregiver "Family notifications" cc list.
- **Modify** `static/console-household.html` — show/toggle both flags per link.
- **Test** `tests/test_household_sharing.py`, `tests/test_household_sharing_routes.py`.

---

### Task 1: `household.py` — consent + cc columns, classification, gates

**Files:** Modify `dashboard/household.py`; Test `tests/test_household_sharing.py`

**Interfaces:**
- Produces: `is_dependent(rel)->bool`; `default_cc_for(rel)->0|1`; `set_share_consent(cx, primary, member, consent)`; `set_cc_enabled(cx, primary, member, enabled)`; `viewable_members_for(cx, primary)->[{email,label,relationship}]`; `cc_recipients_for(cx, member)->[primary_email]`; `caregivers_for(cx, member)->[{primary_email,share_consent}]`; `members_for` now returns `share_consent`/`cc_enabled` per row; `can_view` requires `share_consent=1`; `add_member` sets `cc_enabled` from relationship. `init_household_tables` migrates + backfills.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_household_sharing.py
import sqlite3
from dashboard import household as h


def _cx():
    cx = sqlite3.connect(":memory:"); h.init_household_tables(cx); return cx


def test_classification():
    assert h.default_cc_for("pet") == 1 and h.default_cc_for("child") == 1
    assert h.default_cc_for("caregiving-client") == 1 and h.default_cc_for("dependent") == 1
    assert h.default_cc_for("spouse") == 0 and h.default_cc_for("adult-child") == 0
    assert h.default_cc_for("") == 0 and h.default_cc_for("PET") == 1  # case-insensitive


def test_add_member_sets_cc_from_relationship_and_consent_default():
    cx = _cx()
    h.add_member(cx, "p@x.com", "sasha@x.com", "Sasha", "pet")
    h.add_member(cx, "p@x.com", "rob@x.com", "Rob", "spouse")
    ms = {m["email"]: m for m in h.members_for(cx, "p@x.com")}
    assert ms["sasha@x.com"]["cc_enabled"] == 1 and ms["sasha@x.com"]["share_consent"] == 1
    assert ms["rob@x.com"]["cc_enabled"] == 0 and ms["rob@x.com"]["share_consent"] == 1


def test_can_view_requires_consent():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m@x.com", "M", "spouse")
    assert h.can_view(cx, "p@x.com", "m@x.com") is True       # default consented
    h.set_share_consent(cx, "p@x.com", "m@x.com", 0)
    assert h.can_view(cx, "p@x.com", "m@x.com") is False      # revoked → not viewable
    assert h.can_view(cx, "m@x.com", "m@x.com") is True       # self always


def test_viewable_members_excludes_revoked():
    cx = _cx()
    h.add_member(cx, "p@x.com", "a@x.com", "A", "child")
    h.add_member(cx, "p@x.com", "b@x.com", "B", "spouse")
    h.set_share_consent(cx, "p@x.com", "b@x.com", 0)
    assert [m["email"] for m in h.viewable_members_for(cx, "p@x.com")] == ["a@x.com"]
    # members_for (console) still lists both
    assert len(h.members_for(cx, "p@x.com")) == 2


def test_cc_recipients_two_switch_rule():
    cx = _cx()
    h.add_member(cx, "p@x.com", "m@x.com", "M", "pet")        # cc default 1, consent 1
    assert h.cc_recipients_for(cx, "m@x.com") == ["p@x.com"]
    h.set_cc_enabled(cx, "p@x.com", "m@x.com", 0)
    assert h.cc_recipients_for(cx, "m@x.com") == []            # cc off
    h.set_cc_enabled(cx, "p@x.com", "m@x.com", 1); h.set_share_consent(cx, "p@x.com", "m@x.com", 0)
    assert h.cc_recipients_for(cx, "m@x.com") == []            # consent off
    h.set_share_consent(cx, "p@x.com", "m@x.com", 1)
    assert h.cc_recipients_for(cx, "m@x.com") == ["p@x.com"]   # both on


def test_same_household_ignores_consent():
    cx = _cx()
    h.add_member(cx, "p@x.com", "a@x.com", "A", "child")
    h.add_member(cx, "p@x.com", "b@x.com", "B", "child")
    h.set_share_consent(cx, "p@x.com", "a@x.com", 0)          # revoked
    assert h.same_household(cx, "a@x.com", "b@x.com") is True  # reassignment still works


def test_migration_backfills_cc_for_existing_dependent_rows():
    # simulate a pre-feature table (no share_consent/cc_enabled columns)
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE household_members (id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "primary_email TEXT, member_email TEXT, label TEXT, relationship TEXT, created_at TEXT, "
               "UNIQUE(primary_email, member_email))")
    cx.execute("INSERT INTO household_members (primary_email, member_email, label, relationship, created_at) "
               "VALUES ('p@x.com','pet@x.com','Sasha','pet','t'), ('p@x.com','sp@x.com','Sp','spouse','t')")
    cx.commit()
    h.init_household_tables(cx)   # ALTER + backfill
    ms = {m["email"]: m for m in h.members_for(cx, "p@x.com")}
    assert ms["pet@x.com"]["cc_enabled"] == 1 and ms["pet@x.com"]["share_consent"] == 1
    assert ms["sp@x.com"]["cc_enabled"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python3 -m pytest tests/test_household_sharing.py -q`
Expected: FAIL (`AttributeError: default_cc_for` / column errors).

- [ ] **Step 3: Write minimal implementation** (edit `dashboard/household.py`)

Add classification near the top (after `_norm`):

```python
DEPENDENT_RELATIONSHIPS = {"child", "pet", "dependent", "charge", "caregiving-client"}


def is_dependent(relationship):
    return (relationship or "").strip().lower() in DEPENDENT_RELATIONSHIPS


def default_cc_for(relationship):
    return 1 if is_dependent(relationship) else 0
```

In `init_household_tables`, after the `CREATE TABLE household_members` + indexes and before the `scan_reassignments` block, add the additive migration + one-time backfill:

```python
    # v1 sharing/cc columns (additive). share_consent defaults 1 (member shared,
    # revocable). cc_enabled default 0 at the column level, but a brand-new column
    # is backfilled from relationship (dependents → 1) exactly once.
    try:
        cx.execute("ALTER TABLE household_members ADD COLUMN share_consent INTEGER DEFAULT 1")
    except Exception:
        pass
    try:
        cx.execute("ALTER TABLE household_members ADD COLUMN cc_enabled INTEGER DEFAULT 0")
        # column is brand new (ALTER succeeded) → backfill dependents once
        cx.execute(
            "UPDATE household_members SET cc_enabled=1 WHERE lower(coalesce(relationship,'')) IN (%s)"
            % ",".join("?" * len(DEPENDENT_RELATIONSHIPS)), tuple(sorted(DEPENDENT_RELATIONSHIPS)))
    except Exception:
        pass
```

Update `add_member` to set `cc_enabled` (and rely on the column default for `share_consent`):

```python
def add_member(cx, primary_email, member_email, label="", relationship=""):
    p, m = _norm(primary_email), _norm(member_email)
    if not p or not m or p == m:
        return False
    cx.execute(
        "INSERT OR IGNORE INTO household_members "
        "(primary_email, member_email, label, relationship, created_at, share_consent, cc_enabled) "
        "VALUES (?,?,?,?,?,1,?)",
        (p, m, label or "", relationship or "", _now(), default_cc_for(relationship)))
    cx.commit()
    return True
```

Update `members_for` to return the flags:

```python
def members_for(cx, primary_email):
    rows = cx.execute(
        "SELECT member_email, label, relationship, share_consent, cc_enabled FROM household_members "
        "WHERE primary_email=? ORDER BY created_at, id", (_norm(primary_email),)).fetchall()
    return [{"email": r[0], "label": r[1] or "", "relationship": r[2] or "",
             "share_consent": int(r[3] if r[3] is not None else 1),
             "cc_enabled": int(r[4] if r[4] is not None else 0)} for r in rows]
```

Update `can_view` to require consent:

```python
def can_view(cx, viewer_email, target_email):
    v, t = _norm(viewer_email), _norm(target_email)
    if not v or not t:
        return False
    if v == t:
        return True
    return cx.execute(
        "SELECT 1 FROM household_members WHERE primary_email=? AND member_email=? "
        "AND share_consent=1 LIMIT 1", (v, t)).fetchone() is not None
```

Add the new functions:

```python
def set_share_consent(cx, primary_email, member_email, consent):
    cx.execute("UPDATE household_members SET share_consent=? WHERE primary_email=? AND member_email=?",
               (1 if consent else 0, _norm(primary_email), _norm(member_email)))
    cx.commit()


def set_cc_enabled(cx, primary_email, member_email, enabled):
    cx.execute("UPDATE household_members SET cc_enabled=? WHERE primary_email=? AND member_email=?",
               (1 if enabled else 0, _norm(primary_email), _norm(member_email)))
    cx.commit()


def viewable_members_for(cx, primary_email):
    rows = cx.execute(
        "SELECT member_email, label, relationship FROM household_members "
        "WHERE primary_email=? AND share_consent=1 ORDER BY created_at, id",
        (_norm(primary_email),)).fetchall()
    return [{"email": r[0], "label": r[1] or "", "relationship": r[2] or ""} for r in rows]


def cc_recipients_for(cx, member_email):
    rows = cx.execute(
        "SELECT primary_email FROM household_members "
        "WHERE member_email=? AND share_consent=1 AND cc_enabled=1", (_norm(member_email),)).fetchall()
    return [r[0] for r in rows]


def caregivers_for(cx, member_email):
    rows = cx.execute(
        "SELECT primary_email, share_consent FROM household_members WHERE member_email=? "
        "ORDER BY created_at, id", (_norm(member_email),)).fetchall()
    return [{"primary_email": r[0], "share_consent": int(r[1] if r[1] is not None else 1)} for r in rows]
```

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add dashboard/household.py tests/test_household_sharing.py
git commit -m "feat(household): share_consent + cc_enabled columns, classification, consent-gated can_view"
```

---

### Task 2: portal endpoints (share-consent, cc-pref) + switcher payload

**Files:** Modify `app.py`; Test `tests/test_household_sharing_routes.py`

**Interfaces:**
- Consumes: `household.{set_share_consent,set_cc_enabled,viewable_members_for,members_for,caregivers_for,can_view}`; `_portal_record_for`; `_household_view_enabled`.
- Produces: `_household_sharing_enabled()`; `POST /api/portal/<token>/share-consent`; `POST /api/portal/<token>/cc-pref`; `api_client_portal` payload: the switcher list uses `viewable_members_for`, and (when sharing enabled) carries `household_caregivers` (the token email's inbound caregivers + consent) and `household_cc` (per-member cc state for the primary).

**Context:** In `api_client_portal`, the household block builds `household = _hh.members_for(_cxh, primary_email)` (~app.py:14401) and the `?member=` gate uses `_hh.can_view` (~14403); payload set at `payload["household"]` (~14532). Switch the switcher list to `viewable_members_for`. The console CRUD is `api_console_household` (~15564).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_household_sharing_routes.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch, *, share="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HOUSEHOLD_VIEW_ENABLED", "1")
    monkeypatch.setenv("HOUSEHOLD_SHARING_ENABLED", share)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _mint(appmod, email):
    from dashboard import client_portal as cp, household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); h.init_household_tables(cx)
        tok = cp.upsert_portal(cx, email, "N", {}); cx.commit()
    return tok[0] if isinstance(tok, (tuple, list)) else tok


def test_member_sets_own_consent_only(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx); h.add_member(cx, "care@x.com", "mem@x.com", "M", "spouse"); cx.commit()
    mem_tok = _mint(appmod, "mem@x.com")
    if not mem_tok: pytest.skip("no mint helper")
    c = appmod.app.test_client()
    # member revokes sharing with their caregiver
    r = c.post(f"/api/portal/{mem_tok}/share-consent", json={"caregiver_email": "care@x.com", "consent": 0})
    assert r.status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert h.can_view(cx, "care@x.com", "mem@x.com") is False
    # member cannot set consent for a link where they are NOT the member (token=mem, tries as if caregiver)
    r2 = c.post(f"/api/portal/{mem_tok}/share-consent", json={"caregiver_email": "stranger@x.com", "consent": 1})
    # no row (mem is not a member of stranger) → no-op, still 200 but nothing changed
    assert r2.status_code == 200


def test_caregiver_sets_cc_only_for_own_members(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx); h.add_member(cx, "care@x.com", "mem@x.com", "M", "spouse"); cx.commit()
    care_tok = _mint(appmod, "care@x.com")
    if not care_tok: pytest.skip("no mint helper")
    c = appmod.app.test_client()
    r = c.post(f"/api/portal/{care_tok}/cc-pref", json={"member_email": "mem@x.com", "cc_enabled": 1})
    assert r.status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert h.cc_recipients_for(cx, "mem@x.com") == ["care@x.com"]


def test_flag_off_endpoints_inert(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, share="0")
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx); h.add_member(cx, "care@x.com", "mem@x.com", "M", "pet"); cx.commit()
    mem_tok = _mint(appmod, "mem@x.com")
    if not mem_tok: pytest.skip("no mint helper")
    c = appmod.app.test_client()
    r = c.post(f"/api/portal/{mem_tok}/share-consent", json={"caregiver_email": "care@x.com", "consent": 0})
    assert r.get_json().get("recorded") is False or r.get_json().get("reason") == "disabled"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert h.can_view(cx, "care@x.com", "mem@x.com") is True   # unchanged (flag off)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (404 on the new routes).

- [ ] **Step 3: Write minimal implementation**

Flag helper near `_household_view_enabled` (app.py):

```python
def _household_sharing_enabled():
    return (os.environ.get("HOUSEHOLD_SHARING_ENABLED", "") or "").strip().lower() in ("1", "true", "yes")
```

Two endpoints (near the other `/api/portal/<token>/*` POST routes):

```python
@app.route("/api/portal/<token>/share-consent", methods=["POST"])
def api_portal_share_consent(token):
    """The MEMBER sets whether they share their info+comms with a caregiver. Token-scoped:
    only affects a link where the token's email is the MEMBER."""
    if not _household_sharing_enabled():
        return jsonify({"ok": True, "recorded": False, "reason": "disabled"})
    from dashboard import client_portal as _cp
    from dashboard import household as _hh
    data = request.get_json(silent=True) or {}
    caregiver = (data.get("caregiver_email") or "").strip().lower()
    consent = 1 if data.get("consent") else 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx); _hh.init_household_tables(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"ok": False, "error": "not found"}), 404
        member = (portal.get("email") or "").strip().lower()
        # token owner is the MEMBER of (caregiver -> member)
        _hh.set_share_consent(cx, caregiver, member, consent)
    return jsonify({"ok": True, "recorded": True, "consent": consent})


@app.route("/api/portal/<token>/cc-pref", methods=["POST"])
def api_portal_cc_pref(token):
    """The CAREGIVER (primary) sets cc for one of their members. Token-scoped: only affects
    a link where the token's email is the PRIMARY."""
    if not _household_sharing_enabled():
        return jsonify({"ok": True, "recorded": False, "reason": "disabled"})
    from dashboard import client_portal as _cp
    from dashboard import household as _hh
    data = request.get_json(silent=True) or {}
    member = (data.get("member_email") or "").strip().lower()
    enabled = 1 if data.get("cc_enabled") else 0
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx); _hh.init_household_tables(cx)
        portal = _portal_record_for(cx, token)
        if not portal:
            return jsonify({"ok": False, "error": "not found"}), 404
        primary = (portal.get("email") or "").strip().lower()
        _hh.set_cc_enabled(cx, primary, member, enabled)
    return jsonify({"ok": True, "recorded": True, "cc_enabled": enabled})
```

Switcher payload — change the household list to the consent-filtered set, and (when sharing on) add the caregiver's cc map + the member's inbound caregivers. In `api_client_portal`'s household block, replace `household = _hh.members_for(_cxh, primary_email)` with:

```python
                household = _hh.viewable_members_for(_cxh, primary_email)
                if _household_sharing_enabled():
                    _full = _hh.members_for(_cxh, primary_email)
                    household_cc = {m["email"]: m["cc_enabled"] for m in _full}
                    household_caregivers = _hh.caregivers_for(_cxh, primary_email)
```

Then, immediately before `payload["household"] = household` (~14532), add (guarded):

```python
    if _household_view_enabled() and _household_sharing_enabled():
        try:
            payload["household_cc"] = household_cc          # {member_email: 0|1} for the caregiver UI
            payload["household_caregivers"] = household_caregivers  # this email's inbound caregivers
        except NameError:
            pass
```

> Implementer note: `household_cc`/`household_caregivers` are only defined inside the `_household_view_enabled()` try-block; guard with the `NameError` catch (or initialize them to `{}`/`[]` at the top of the function alongside `household = []`). Prefer initializing to empty defaults up top for clarity.

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_household_sharing_routes.py
git commit -m "feat(household): share-consent + cc-pref endpoints + consent-filtered switcher payload"
```

---

### Task 3: CC routing at the report + invoice sends

**Files:** Modify `app.py` (`_send_reveal_link`), `dashboard/orders.py` (`_send_invoice_exec`); Test `tests/test_household_sharing_routes.py`

**Interfaces:**
- Consumes: `household.cc_recipients_for(cx, member_email)`; the existing send helpers `_send_inquiry_email(to_email, subject, body, reply_to=None)` (app.py) and `dashboard.inbox.send_email(to_email, subject, body, ...)` (used by invoices).
- Produces: a private caregiver-copy send at each site when `_household_sharing_enabled()` and `cc_recipients_for` is non-empty.

**CC delivery = a SEPARATE private copy** to each caregiver via the same helper (subject/body point them to their own portal), NOT a shared `Cc` header — avoids cross-exposing addresses. Best-effort: wrap in try/except so it never blocks the member's own send.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_household_sharing_routes.py
def test_cc_copy_sent_for_report(monkeypatch, tmp_path):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx); h.add_member(cx, "care@x.com", "mem@x.com", "M", "pet"); cx.commit()
    sent = []
    monkeypatch.setattr(appmod, "_send_inquiry_email",
                        lambda to, subj, body, **k: sent.append(to) or (True, ""))
    # call the cc-fanout helper directly (the site calls it after the member send)
    appmod._household_cc_report("mem@x.com", "New scan for M")
    assert "care@x.com" in sent          # caregiver got a private copy


def test_no_cc_copy_when_switch_off(monkeypatch, tmp_path):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx); h.add_member(cx, "care@x.com", "mem@x.com", "M", "pet")
        h.set_cc_enabled(cx, "care@x.com", "mem@x.com", 0); cx.commit()
    sent = []
    monkeypatch.setattr(appmod, "_send_inquiry_email",
                        lambda to, subj, body, **k: sent.append(to) or (True, ""))
    appmod._household_cc_report("mem@x.com", "New scan for M")
    assert sent == []                     # cc off → no copy
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`_household_cc_report` undefined).

- [ ] **Step 3: Write minimal implementation**

Add the fanout helper in `app.py` (near `_send_reveal_link`):

```python
def _household_cc_report(member_email, member_label_or_subject):
    """Best-effort: send a private 'a new report is available for <member>' copy to each
    caregiver who is consented + cc-subscribed for this member. Points them to their own
    portal (no member token forwarded, no shared Cc header — addresses aren't cross-exposed)."""
    if not _household_sharing_enabled():
        return
    try:
        from dashboard import household as _hh
        with sqlite3.connect(LOG_DB) as cx:
            _hh.init_household_tables(cx)
            recips = _hh.cc_recipients_for(cx, (member_email or "").strip().lower())
        for care in recips:
            subj = "A new biofield report is available for someone in your care"
            body = ("A new biofield report was just published for a member of your household. "
                    "Open your portal to view it.\n\nhttps://illtowell.com/portal/login")
            try:
                _send_inquiry_email(care, subj, body)
            except Exception as _e:
                print(f"[household-cc] report copy to {care}: {_e!r}", flush=True)
    except Exception as _e:
        print(f"[household-cc] report fanout: {_e!r}", flush=True)
```

Call it in `_send_reveal_link` right AFTER the member's own send succeeds (find the `email` var + the successful-send path; add `_household_cc_report(email, None)` inside a try/except so it never affects the return).

Add an invoice fanout helper in `app.py` (symmetry) and call it from `dashboard/orders.py::_send_invoice_exec` after the member invoice email sends — OR, to keep orders.py decoupled, add the fanout inline in `_send_invoice_exec` using `dashboard.household` + `dashboard.inbox.send_email`:

```python
    # household cc: private copy to consented+subscribed caregivers (best-effort)
    try:
        import os as _os
        if (_os.environ.get("HOUSEHOLD_SHARING_ENABLED", "") or "").strip().lower() in ("1","true","yes"):
            from dashboard import household as _hh
            from dashboard import inbox as _inbox2
            _hh.init_household_tables(ctx_db)   # use this function's db handle (read it: the cx/conn in scope)
            for _care in _hh.cc_recipients_for(ctx_db, email):
                try:
                    _inbox2.send_email(_care, "An invoice is available for someone in your care",
                                       "A new invoice was sent to a member of your household. "
                                       "Open your portal to view it: https://illtowell.com/portal/login")
                except Exception as _e:
                    print(f"[household-cc] invoice copy to {_care}: {_e!r}", flush=True)
    except Exception as _e:
        print(f"[household-cc] invoice fanout: {_e!r}", flush=True)
```

> Implementer note: `_send_invoice_exec` has its own db handle/params — read the function and use the connection it already opens (do not open a nested locked connection). Place the fanout AFTER the member's invoice `send_email` succeeds. Confirm `inbox.send_email`'s exact required args (from_name is optional). Keep everything in try/except so an invoice send never fails because of the cc.

- [ ] **Step 4: Run to verify it passes** — GREEN.

- [ ] **Step 5: Commit**

```bash
git add app.py dashboard/orders.py tests/test_household_sharing_routes.py
git commit -m "feat(household): cc private caregiver copy on report + invoice sends (flag-gated, best-effort)"
```

---

### Task 4: member + caregiver toggles in `static/client-portal.html`

**Files:** Modify `static/client-portal.html`

**Interfaces:**
- Consumes: `d.household` (viewable members, for the caregiver list), `d.household_cc` (`{member_email: 0|1}`), `d.household_caregivers` (`[{primary_email, share_consent}]`), `TOKEN`.

UI-only (no pytest). Behind the payload: render nothing new when `d.household_caregivers === undefined` (flag off).

- [ ] **Step 1: Add the two controls**

In `render(d, v)`:
- **Member "Sharing" toggle:** when `d.household_caregivers && d.household_caregivers.length`, for each caregiver render a checkbox "Share my scans & messages with this practitioner/caregiver" bound to `share_consent`. On change → `POST /api/portal/${TOKEN}/share-consent {caregiver_email, consent}`.
- **Caregiver "Family notifications" list:** when `d.household && d.household.length` (the caregiver has viewable members) AND `d.household_cc`, render one row per member with a checkbox "Email me about **[label]**" bound to `d.household_cc[member.email]`. On change → `POST /api/portal/${TOKEN}/cc-pref {member_email, cc_enabled}`.
- Use `data-*` attributes + `addEventListener` (NOT inline onclick with interpolated values — the codebase XSS convention). Escape all labels/emails with `esc()`.

```javascript
  // Household sharing controls (member consent + caregiver cc).
  if (d.household_caregivers && d.household_caregivers.length) {
    let rows = "";
    for (const c of d.household_caregivers) {
      const ck = c.share_consent ? "checked" : "";
      rows += `<label style="display:flex;align-items:center;gap:8px;margin:.3rem 0">
        <input type="checkbox" class="hh-share" data-care="${esc(c.primary_email)}" ${ck}>
        Share my scans &amp; messages with ${esc(c.primary_email)}</label>`;
    }
    html += `<div class="card"><h2 style="font-size:1rem">Sharing</h2>${rows}</div>`;
  }
  if (d.household && d.household.length && d.household_cc) {
    let rows = "";
    for (const m of d.household) {
      const ck = d.household_cc[m.email] ? "checked" : "";
      rows += `<label style="display:flex;align-items:center;gap:8px;margin:.3rem 0">
        <input type="checkbox" class="hh-cc" data-member="${esc(m.email)}" ${ck}>
        Email me about ${esc(m.label || m.email)}</label>`;
    }
    html += `<div class="card"><h2 style="font-size:1rem">Family notifications</h2>${rows}</div>`;
  }
```

Wire the listeners after render (where other portal listeners attach):

```javascript
  app.querySelectorAll(".hh-share").forEach(el => el.addEventListener("change", () =>
    fetch(`/api/portal/${encodeURIComponent(TOKEN)}/share-consent`, {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({caregiver_email: el.dataset.care, consent: el.checked ? 1 : 0})})));
  app.querySelectorAll(".hh-cc").forEach(el => el.addEventListener("change", () =>
    fetch(`/api/portal/${encodeURIComponent(TOKEN)}/cc-pref`, {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({member_email: el.dataset.member, cc_enabled: el.checked ? 1 : 0})})));
```

- [ ] **Step 2: Verify (static)** — extract inline `<script>`, `node --check`; grep-confirm both controls read the payload keys, POST the two endpoints, use `data-*`+`addEventListener`, escape strings; nothing renders when the payload keys are absent (flag off). Report live render-verify pending.

- [ ] **Step 3: Commit**

```bash
git add static/client-portal.html
git commit -m "feat(household): member sharing toggle + caregiver cc list on the portal"
```

---

### Task 5: console-household — show/toggle both flags

**Files:** Modify `static/console-household.html`; Modify `app.py` (`api_console_household`)

**Interfaces:**
- Consumes: `household.{members_for (now returns share_consent/cc_enabled), set_share_consent, set_cc_enabled}`.
- Produces: the console household GET already returns members (now carrying the two flags via `members_for`); POST accepts an optional `{primary_email, member_email, share_consent?, cc_enabled?}` update to toggle a flag; the page shows both flags per member with toggle controls.

- [ ] **Step 1: Write the failing test** (append to `tests/test_household_sharing_routes.py`)

```python
def test_console_toggles_flags(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx); h.add_member(cx, "p@x.com", "m@x.com", "M", "spouse"); cx.commit()
    c = appmod.app.test_client()
    # GET shows the flags
    g = c.get("/api/console/household?primary_email=p@x.com").get_json()
    row = next(m for m in g["members"] if m["email"] == "m@x.com")
    assert "share_consent" in row and "cc_enabled" in row
    # POST toggles cc_enabled on
    assert c.post("/api/console/household",
                  json={"primary_email": "p@x.com", "member_email": "m@x.com", "cc_enabled": 1}).status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert h.cc_recipients_for(cx, "m@x.com") == ["p@x.com"]
```

- [ ] **Step 2: Run to verify it fails** — FAIL (POST ignores the flag fields).

- [ ] **Step 3: Write minimal implementation**

In `api_console_household` POST branch, after the existing add/remove handling, apply optional flag toggles when present:

```python
        if request.method == "POST":
            _hh.add_member(cx, data.get("primary_email"), data.get("member_email"),
                           data.get("label", ""), data.get("relationship", ""))
            if "share_consent" in data:
                _hh.set_share_consent(cx, data.get("primary_email"), data.get("member_email"),
                                      1 if data.get("share_consent") else 0)
            if "cc_enabled" in data:
                _hh.set_cc_enabled(cx, data.get("primary_email"), data.get("member_email"),
                                   1 if data.get("cc_enabled") else 0)
```

> Implementer note: `add_member` is `INSERT OR IGNORE`, so calling it for an existing link is a harmless no-op — a flag-only POST (existing member) still reaches the set_* calls. Confirm the GET already returns `share_consent`/`cc_enabled` (it will, via `members_for` from Task 1); if the GET builds its own dict, add the two keys.

In `static/console-household.html`, in the members table, add two columns "Share" and "CC" with checkboxes bound to `m.share_consent`/`m.cc_enabled`; on change POST `/api/console/household` with `{primary_email, member_email, share_consent}` or `{...cc_enabled}`. Use `data-*` + `addEventListener` (no inline onclick with interpolated email), escape values.

```html
<!-- in the row template, add: -->
<td><input type="checkbox" class="hh-share" data-member="${esc(m.email)}" ${m.share_consent ? "checked":""}></td>
<td><input type="checkbox" class="hh-cc" data-member="${esc(m.email)}" ${m.cc_enabled ? "checked":""}></td>
```
```javascript
// after rendering the table:
document.querySelectorAll(".hh-share").forEach(el => el.addEventListener("change", () =>
  fetch("/api/console/household",{method:"POST",headers:H,body:JSON.stringify(
    {primary_email:document.getElementById("primary").value.trim().toLowerCase(),
     member_email:el.dataset.member, share_consent: el.checked?1:0})})));
document.querySelectorAll(".hh-cc").forEach(el => el.addEventListener("change", () =>
  fetch("/api/console/household",{method:"POST",headers:H,body:JSON.stringify(
    {primary_email:document.getElementById("primary").value.trim().toLowerCase(),
     member_email:el.dataset.member, cc_enabled: el.checked?1:0})})));
```

- [ ] **Step 4: Run to verify it passes** — GREEN. Also `node --check` the console page's inline script.

- [ ] **Step 5: Commit**

```bash
git add app.py static/console-household.html tests/test_household_sharing_routes.py
git commit -m "feat(household): console shows + toggles share_consent / cc_enabled per member"
```

---

## Self-Review

**Spec coverage:**
- `share_consent` + `cc_enabled` columns + backfill + relationship classification → Task 1. ✓
- `can_view` requires consent; `viewable_members_for`; `same_household` unchanged → Task 1. ✓
- `cc_recipients_for` two-switch rule → Task 1; hooked at report + invoice sends → Task 3. ✓
- Member `/share-consent` + caregiver `/cc-pref`, token-scoped, flag-gated → Task 2. ✓
- Switcher uses consent-filtered list; caregiver cc map + member caregivers in payload → Task 2. ✓
- Member + caregiver portal toggles → Task 4; owner console both flags → Task 5. ✓
- Behind `HOUSEHOLD_SHARING_ENABLED`, default OFF, no-regression via default-1 → Tasks 1 (default), 2/3 (flag gates). ✓
- CC delivery = private separate copy (refines spec "Cc header" for privacy) → Task 3 Global Constraint + notes. ✓ (flag to Glen at handoff.)

**Placeholder scan:** Tasks 3 & 5 carry implementer notes to read the real `_send_invoice_exec` db handle + `inbox.send_email` args and the console GET dict — "confirm the real name," not TBDs. No hand-waves.

**Type consistency:** `cc_recipients_for -> [primary_email]` (Task 1) consumed by Task 3. `members_for` now returns `share_consent`/`cc_enabled` (Task 1) consumed by Tasks 2 (payload cc map) & 5 (console). `viewable_members_for` (Task 1) = the switcher list (Task 2). `caregivers_for -> [{primary_email,share_consent}]` (Task 1) = `d.household_caregivers` (Tasks 2,4). Payload keys `household_cc`/`household_caregivers` defined in Task 2, consumed in Task 4. Endpoints `/share-consent` (member) + `/cc-pref` (caregiver) named identically across Tasks 2 & 4. `HOUSEHOLD_SHARING_ENABLED` gates Tasks 2,3,4,5; `share_consent` default-1 keeps Task 1's `can_view` change regression-free.
