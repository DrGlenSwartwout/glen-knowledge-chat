# Studio-Credit Free Month Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Glen/Rae log a studio-coaching-app purchaser as a claim in the console and approve it to grant one free 30-day membership comp (one per email per year), reusing the existing access-grant + magic-link plumbing.

**Architecture:** A new `studio_credit_claims` table is the durable record. A standalone, Flask-free store module (`dashboard/studio_credit.py`) holds CRUD + the approve/reject logic with the per-year guard; the grant+notify side effect is injected so the store stays unit-testable. A thin dispatch-actions module (`dashboard/studio_credit_actions.py`) exposes `studio_credit.add/approve/reject` on the existing Business-OS dispatch spine, and app.py wires the real grant function (`_grant_membership` + journey event + magic-link email), a read-only list API, and a static console page. Console-only; no new public flag.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest. Existing dispatch/actions/rbac framework in `dashboard/`.

## Global Constraints

- **Python:** use `python3` (not `python`); never `import app` in tests — the app imports Pinecone at module load and fails in the sandbox. Test `dashboard/*` helpers standalone. (verbatim from project test-isolation rule)
- **Comp is pure access window:** no card, no Stripe, no auto-charge. 30 days. `source="studio_credit"`.
- **Idempotency:** one studio credit per email per **365 days**; blocked grants are overridable only with explicit `force=True`. A prior grant older than 365 days, or none, is allowed.
- **RBAC:** all actions `permission=(OWNER, OPS)`, `risk_tier=LOW_WRITE`.
- **No new public flag.** Console-only, gated by the existing `CONSOLE_SECRET` like `/console/biofield-reveals`.
- **DB:** single sqlite file `LOG_DB`; the `memberships` table already exists with columns `(id, email, granted_at, expires_at, granted_by, source, truly_vip_ref, notes)`; `granted_at`/`expires_at` are `datetime.utcnow().isoformat() + "Z"`.
- **Worktree:** all edits/commits happen in `/tmp/wt-deploy-chat-0cbb0565` (branch `sess/0cbb0565`).

---

### Task 1: Claim store — table + CRUD (`dashboard/studio_credit.py`)

**Files:**
- Create: `dashboard/studio_credit.py`
- Test: `tests/test_studio_credit.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `migrate(cx) -> None` — `CREATE TABLE IF NOT EXISTS studio_credit_claims`.
  - `add_claim(cx, *, email, invoice_ref="", proof_note="", source="console", created_by="") -> dict` — inserts a `pending` row, returns the claim dict.
  - `get(cx, claim_id) -> dict | None`.
  - `list_claims(cx, status=None) -> list[dict]` — newest first; `status` filters when given.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_studio_credit.py
import sqlite3, sys
from pathlib import Path
import pytest


def _mod():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import studio_credit
        return studio_credit
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    _mod().migrate(cx)
    return cx


def test_add_claim_creates_pending(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    c = m.add_claim(cx, email="Buyer@X.com", invoice_ref="INV-9",
                    proof_note="emailed 6/20", source="console", created_by="glen")
    assert c["status"] == "pending"
    assert c["email"] == "buyer@x.com"          # lowercased
    assert c["invoice_ref"] == "INV-9"
    assert c["source"] == "console"
    assert c["id"]
    got = m.get(cx, c["id"])
    assert got["email"] == "buyer@x.com" and got["status"] == "pending"


def test_list_claims_filters_by_status(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    m.add_claim(cx, email="a@x.com", source="console")
    m.add_claim(cx, email="b@x.com", source="console")
    assert len(m.list_claims(cx)) == 2
    assert len(m.list_claims(cx, status="pending")) == 2
    assert m.list_claims(cx, status="approved") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-0cbb0565 && python3 -m pytest tests/test_studio_credit.py -v`
Expected: FAIL / skip — `cannot import name 'studio_credit'`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/studio_credit.py
"""Studio-credit free month: claim store + approve/reject with a one-per-year
guard. Phase 1 = console side only (no public claim form). The grant+notify side
effect is injected at approve time so this module stays Flask-free and unit-testable."""
import sqlite3
import uuid
from datetime import datetime, timedelta


def _now():
    return datetime.utcnow().isoformat() + "Z"


def migrate(cx) -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS studio_credit_claims (
            id            TEXT PRIMARY KEY,
            email         TEXT NOT NULL,
            invoice_ref   TEXT NOT NULL DEFAULT '',
            proof_note    TEXT NOT NULL DEFAULT '',
            status        TEXT NOT NULL DEFAULT 'pending',
            created_at    TEXT NOT NULL,
            created_by    TEXT NOT NULL DEFAULT '',
            decided_at    TEXT,
            decided_by    TEXT,
            decision_note TEXT NOT NULL DEFAULT '',
            membership_id TEXT,
            source        TEXT NOT NULL DEFAULT 'console'
        )
    """)
    cx.commit()


def _row(r):
    return dict(r) if r is not None else None


def add_claim(cx, *, email, invoice_ref="", proof_note="", source="console", created_by=""):
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("valid email required")
    cid = str(uuid.uuid4())
    cx.execute(
        "INSERT INTO studio_credit_claims "
        "(id, email, invoice_ref, proof_note, status, created_at, created_by, source) "
        "VALUES (?,?,?,?, 'pending', ?, ?, ?)",
        (cid, email, invoice_ref or "", proof_note or "", _now(), created_by or "", source or "console"))
    cx.commit()
    return get(cx, cid)


def get(cx, claim_id):
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    return _row(cur.execute(
        "SELECT * FROM studio_credit_claims WHERE id=?", (claim_id,)).fetchone())


def list_claims(cx, status=None):
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    if status:
        rows = cur.execute(
            "SELECT * FROM studio_credit_claims WHERE status=? ORDER BY created_at DESC",
            (status,)).fetchall()
    else:
        rows = cur.execute(
            "SELECT * FROM studio_credit_claims ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-0cbb0565 && python3 -m pytest tests/test_studio_credit.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-0cbb0565
git add dashboard/studio_credit.py tests/test_studio_credit.py
git commit -m "feat(studio-credit): claim store table + CRUD"
```

---

### Task 2: Per-year guard + approve/reject logic (`dashboard/studio_credit.py`)

**Files:**
- Modify: `dashboard/studio_credit.py` (add functions)
- Test: `tests/test_studio_credit.py` (add cases)

**Interfaces:**
- Consumes: `migrate`, `add_claim`, `get` from Task 1.
- Produces:
  - `studio_credit_granted_within_year(cx, email) -> dict | None` — returns `{"granted_at": ..., "until": ...}` for the most-recent `studio_credit` membership granted in the last 365 days, else `None`. Reads the existing `memberships` table.
  - `approve_claim(cx, claim_id, *, decided_by, grant_fn, force=False) -> dict` — `grant_fn(cx, email, days) -> {"membership_id": str, "magic_link_url": str}`. Returns `{"ok": True, "membership_id", "magic_link_url"}` on grant; `{"ok": False, "warning": "granted_within_year", "granted_at", "until"}` when blocked; idempotent re-approve returns the stored membership_id with `"already": True`.
  - `reject_claim(cx, claim_id, *, decided_by, reason="") -> dict` — flips to `rejected`, no grant.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_studio_credit.py

def _seed_memberships_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS memberships "
        "(id TEXT PRIMARY KEY, email TEXT, granted_at TEXT, expires_at TEXT, "
        " granted_by TEXT, source TEXT, truly_vip_ref TEXT, notes TEXT)")
    cx.commit()


def _insert_membership(cx, email, source, granted_days_ago):
    from datetime import datetime, timedelta
    import uuid
    g = (datetime.utcnow() - timedelta(days=granted_days_ago)).isoformat() + "Z"
    e = (datetime.utcnow() - timedelta(days=granted_days_ago) + timedelta(days=30)).isoformat() + "Z"
    cx.execute("INSERT INTO memberships VALUES (?,?,?,?,?,?,?,?)",
               (str(uuid.uuid4()), email.lower(), g, e, source, source, "", ""))
    cx.commit()


class _GrantSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, cx, email, days):
        self.calls.append((email, days))
        mid = "mem-" + str(len(self.calls))
        _insert_membership(cx, email, "studio_credit", 0)  # simulate the real grant row
        return {"membership_id": mid, "magic_link_url": "https://x/coaching/auth/tok"}


def test_approve_grants_30_day_studio_credit(tmp_path):
    m = _mod(); cx = _cx(tmp_path); _seed_memberships_table(cx)
    c = m.add_claim(cx, email="a@x.com", source="console")
    spy = _GrantSpy()
    res = m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy)
    assert res["ok"] is True and res["membership_id"] == "mem-1"
    assert spy.calls == [("a@x.com", 30)]
    assert m.get(cx, c["id"])["status"] == "approved"


def test_double_approve_is_idempotent(tmp_path):
    m = _mod(); cx = _cx(tmp_path); _seed_memberships_table(cx)
    c = m.add_claim(cx, email="a@x.com", source="console")
    spy = _GrantSpy()
    m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy)
    res2 = m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy)
    assert res2.get("already") is True and res2["membership_id"] == "mem-1"
    assert len(spy.calls) == 1   # not granted again


def test_within_year_blocks_without_force(tmp_path):
    m = _mod(); cx = _cx(tmp_path); _seed_memberships_table(cx)
    _insert_membership(cx, "a@x.com", "studio_credit", 100)  # 100 days ago
    c = m.add_claim(cx, email="a@x.com", source="console")
    spy = _GrantSpy()
    res = m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy)
    assert res["ok"] is False and res["warning"] == "granted_within_year"
    assert spy.calls == []
    assert m.get(cx, c["id"])["status"] == "pending"   # unchanged
    res2 = m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy, force=True)
    assert res2["ok"] is True and spy.calls == [("a@x.com", 30)]


def test_grant_older_than_year_is_allowed(tmp_path):
    m = _mod(); cx = _cx(tmp_path); _seed_memberships_table(cx)
    _insert_membership(cx, "a@x.com", "studio_credit", 400)  # >365 days ago
    c = m.add_claim(cx, email="a@x.com", source="console")
    spy = _GrantSpy()
    res = m.approve_claim(cx, c["id"], decided_by="glen", grant_fn=spy)
    assert res["ok"] is True and spy.calls == [("a@x.com", 30)]


def test_reject_grants_nothing(tmp_path):
    m = _mod(); cx = _cx(tmp_path); _seed_memberships_table(cx)
    c = m.add_claim(cx, email="a@x.com", source="console")
    spy = _GrantSpy()
    res = m.reject_claim(cx, c["id"], decided_by="glen", reason="no invoice")
    assert res["ok"] is True
    assert spy.calls == []
    row = m.get(cx, c["id"])
    assert row["status"] == "rejected" and row["decision_note"] == "no invoice"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-0cbb0565 && python3 -m pytest tests/test_studio_credit.py -v`
Expected: the 5 new tests FAIL — `module has no attribute 'approve_claim'`.

- [ ] **Step 3: Write minimal implementation (append to `dashboard/studio_credit.py`)**

```python
def studio_credit_granted_within_year(cx, email):
    """Most-recent studio_credit membership for email granted in the last 365 days,
    or None. Reads the existing memberships table (same DB)."""
    email = (email or "").strip().lower()
    cutoff = (datetime.utcnow() - timedelta(days=365)).isoformat() + "Z"
    try:
        row = cx.execute(
            "SELECT granted_at, expires_at FROM memberships "
            "WHERE email=? AND source='studio_credit' AND granted_at > ? "
            "ORDER BY granted_at DESC LIMIT 1",
            (email, cutoff)).fetchone()
    except sqlite3.OperationalError:
        return None   # memberships table absent (shouldn't happen in prod)
    if not row:
        return None
    return {"granted_at": row[0], "until": row[1]}


def approve_claim(cx, claim_id, *, decided_by, grant_fn, force=False):
    claim = get(cx, claim_id)
    if claim is None:
        raise ValueError("claim not found")
    if claim["status"] == "approved":
        return {"ok": True, "already": True, "membership_id": claim["membership_id"]}
    if claim["status"] == "rejected":
        raise ValueError("claim already rejected")
    email = claim["email"]
    if not force:
        prior = studio_credit_granted_within_year(cx, email)
        if prior is not None:
            return {"ok": False, "warning": "granted_within_year",
                    "granted_at": prior["granted_at"], "until": prior["until"]}
    granted = grant_fn(cx, email, 30)
    cx.execute(
        "UPDATE studio_credit_claims SET status='approved', decided_at=?, decided_by=?, "
        "membership_id=? WHERE id=?",
        (_now(), decided_by or "", granted["membership_id"], claim_id))
    cx.commit()
    return {"ok": True, "membership_id": granted["membership_id"],
            "magic_link_url": granted.get("magic_link_url", "")}


def reject_claim(cx, claim_id, *, decided_by, reason=""):
    claim = get(cx, claim_id)
    if claim is None:
        raise ValueError("claim not found")
    if claim["status"] == "approved":
        raise ValueError("cannot reject an approved claim")
    cx.execute(
        "UPDATE studio_credit_claims SET status='rejected', decided_at=?, decided_by=?, "
        "decision_note=? WHERE id=?",
        (_now(), decided_by or "", reason or "", claim_id))
    cx.commit()
    return {"ok": True}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-0cbb0565 && python3 -m pytest tests/test_studio_credit.py -v`
Expected: PASS (7 passed total).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-0cbb0565
git add dashboard/studio_credit.py tests/test_studio_credit.py
git commit -m "feat(studio-credit): per-year guard + approve/reject logic"
```

---

### Task 3: Dispatch actions (`dashboard/studio_credit_actions.py`)

**Files:**
- Create: `dashboard/studio_credit_actions.py`
- Test: `tests/test_studio_credit_actions.py`

**Interfaces:**
- Consumes: `dashboard.studio_credit` (Task 1+2); `dashboard.actions` (`register_action`, `Action`, `LOW_WRITE`, `get_action`); `dashboard.rbac` (`OWNER`, `OPS`).
- Produces:
  - `configure(grant_fn=None) -> None` — injects the real grant+notify function (app.py supplies it; tests inject a spy).
  - `register() -> None` — idempotently registers `studio_credit.add`, `studio_credit.approve`, `studio_credit.reject`.
  - Executors read `ctx["cx"]` and `ctx["actor"]`; params: `add` → email/invoice_ref/proof_note; `approve` → id/force; `reject` → id/reason.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_studio_credit_actions.py
import sqlite3, sys
from pathlib import Path
import pytest


def _mods():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import studio_credit, studio_credit_actions, actions
        return studio_credit, studio_credit_actions, actions
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


class _Actor:
    role = "owner"
    name = "glen"


def _grant_spy(cx, email, days):
    import uuid
    g = "2026-06-20T00:00:00Z"
    cx.execute("INSERT INTO memberships VALUES (?,?,?,?,?,?,?,?)",
               (str(uuid.uuid4()), email, g, "2026-07-20T00:00:00Z",
                "studio_credit", "studio_credit", "", ""))
    return {"membership_id": "mem-x", "magic_link_url": "https://x/tok"}


def _setup(tmp_path):
    sc, sca, _ = _mods()
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    cx.row_factory = sqlite3.Row
    sc.migrate(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS memberships "
               "(id TEXT PRIMARY KEY, email TEXT, granted_at TEXT, expires_at TEXT, "
               " granted_by TEXT, source TEXT, truly_vip_ref TEXT, notes TEXT)")
    cx.commit()
    sca.configure(grant_fn=_grant_spy)
    sca.register()
    return sc, sca, cx


def test_add_then_approve_via_executor(tmp_path):
    sc, sca, cx = _setup(tmp_path)
    from dashboard.actions import get_action
    add = get_action("studio_credit.add")
    assert add.risk_tier == "low_write" and set(add.permission) == {"owner", "ops"}
    r = add.executor({"email": "a@x.com", "invoice_ref": "INV1"},
                     {"cx": cx, "actor": _Actor()})
    cid = r["id"]
    appr = get_action("studio_credit.approve")
    r2 = appr.executor({"id": cid}, {"cx": cx, "actor": _Actor()})
    assert r2["ok"] is True and r2["membership_id"] == "mem-x"
    assert sc.get(cx, cid)["status"] == "approved"


def test_approve_blocked_within_year_executor(tmp_path):
    sc, sca, cx = _setup(tmp_path)
    from dashboard.actions import get_action
    cx.execute("INSERT INTO memberships VALUES "
               "('m0','a@x.com','2026-06-01T00:00:00Z','2026-07-01T00:00:00Z',"
               "'studio_credit','studio_credit','','')")
    cx.commit()
    cid = get_action("studio_credit.add").executor(
        {"email": "a@x.com"}, {"cx": cx, "actor": _Actor()})["id"]
    res = get_action("studio_credit.approve").executor(
        {"id": cid}, {"cx": cx, "actor": _Actor()})
    assert res["ok"] is False and res["warning"] == "granted_within_year"
```

> Note: the second test seeds a `granted_at` inside the last 365 days relative to 2026-06-20. If running far in the future, adjust the seeded date; the guard is 365 days from `utcnow()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-0cbb0565 && python3 -m pytest tests/test_studio_credit_actions.py -v`
Expected: FAIL / skip — cannot import `studio_credit_actions`.

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/studio_credit_actions.py
"""Studio-credit console actions on the Business-OS dispatch spine: log a claim,
approve (grants the 30-day comp + magic-link email via the injected grant_fn),
reject. OWNER/OPS, LOW_WRITE."""
from dashboard.actions import register_action, Action, LOW_WRITE, get_action
from dashboard.rbac import OWNER, OPS
from dashboard import studio_credit as _sc

_DEPS = {"grant_fn": None}


def configure(grant_fn=None):
    if grant_fn is not None:
        _DEPS["grant_fn"] = grant_fn


def _actor_name(ctx):
    a = ctx.get("actor")
    return (getattr(a, "name", "") or getattr(a, "role", "") or "console")


def _exec_add(params, ctx):
    email = (params.get("email") or "").strip()
    if not email:
        raise ValueError("email required")
    claim = _sc.add_claim(
        ctx["cx"], email=email,
        invoice_ref=(params.get("invoice_ref") or "").strip(),
        proof_note=(params.get("proof_note") or "").strip(),
        source="console", created_by=_actor_name(ctx))
    return {"ok": True, "id": claim["id"], "status": claim["status"]}


def _exec_approve(params, ctx):
    cid = (params.get("id") or "").strip()
    if not cid:
        raise ValueError("id required")
    grant_fn = _DEPS["grant_fn"]
    if grant_fn is None:
        raise RuntimeError("studio_credit_actions not configured with grant_fn")
    return _sc.approve_claim(
        ctx["cx"], cid, decided_by=_actor_name(ctx),
        grant_fn=grant_fn, force=bool(params.get("force")))


def _exec_reject(params, ctx):
    cid = (params.get("id") or "").strip()
    if not cid:
        raise ValueError("id required")
    return _sc.reject_claim(
        ctx["cx"], cid, decided_by=_actor_name(ctx),
        reason=(params.get("reason") or "").strip())


def register():
    if get_action("studio_credit.add"):
        return
    register_action(Action(
        key="studio_credit.add", module="studio_credit", title="Log studio-credit claim",
        description="Record a studio-app purchaser's claim for a free month (stays pending).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_add))
    register_action(Action(
        key="studio_credit.approve", module="studio_credit", title="Approve studio credit",
        description="Grant the 30-day membership comp + send the magic link (one per email per year).",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_approve))
    register_action(Action(
        key="studio_credit.reject", module="studio_credit", title="Reject studio-credit claim",
        description="Reject the claim with a reason; grants nothing.",
        risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_reject))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-0cbb0565 && python3 -m pytest tests/test_studio_credit_actions.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-0cbb0565
git add dashboard/studio_credit_actions.py tests/test_studio_credit_actions.py
git commit -m "feat(studio-credit): dispatch actions add/approve/reject"
```

---

### Task 4: App wiring — startup init, real grant_fn, list API, console route

**Files:**
- Modify: `app.py` — (a) table init at startup; (b) module-level grant function; (c) register+configure block near the other `*_actions` registrations (~line 20299); (d) two routes.

**Interfaces:**
- Consumes: `_grant_membership(cx, email, days, source)` (app.py ~6201), `_mint_membership_magic_link(email)` (~6158), `_send_inquiry_email(...)` (~6587), `RM_INBOUND_INQUIRY_EMAIL`, `LOG_DB`, `_db_lock`, `CONSOLE_SECRET`, `STATIC`, `send_from_directory`.
- Produces: registered `studio_credit.*` actions configured with a real grant_fn; `GET /api/console/studio-credits`; `GET /console/studio-credits`.

- [ ] **Step 1: Add the module-level grant function (place it just below `_grant_membership`, after line ~6211)**

```python
def _studio_credit_grant_and_notify(cx, email, days):
    """Grant a studio-credit comp membership, log the journey event, and email the
    magic link. Returns {membership_id, magic_link_url}. Shared by the console
    approve action."""
    import json as _json
    email = (email or "").strip().lower()
    mid = _grant_membership(cx, email, days, "studio_credit")
    plain = _mint_membership_magic_link(email)
    try:
        base = request.host_url.rstrip("/")
    except Exception:
        base = (PUBLIC_BASE_URL or "").rstrip("/")
    magic_link_url = f"{base}/coaching/auth/{plain}"
    subject = "Your free month of Remedy Match coaching is open"
    body = (
        f"Hi,\n\n"
        f"Thanks for getting the studio coaching app. As a thank-you, your Remedy Match "
        f"coaching membership is open free for the next {days} days.\n\n"
        f"Click here to sign in:\n{magic_link_url}\n\n"
        f"You'll land in your member dashboard with the AI agent loaded for your context.\n\n"
        f"---\n"
        f"Remedy Match LLC, 351 Wailuku Drive, Hilo, Hawai'i 96720 USA\n"
    )
    try:
        _send_inquiry_email(to_email=email, subject=subject, body=body,
                            reply_to=RM_INBOUND_INQUIRY_EMAIL)
    except Exception as e:
        print(f"[studio-credit] email send failed: {e!r}", flush=True)
    try:
        cx.execute(
            "INSERT INTO journey_events "
            "(ts, session_id, email, trigger, detail, rung_before, rung_after) "
            "VALUES (?, ?, ?, 'membership_granted', ?, '', '')",
            (datetime.utcnow().isoformat() + "Z", "", email,
             _json.dumps({"source": "studio_credit", "days": days, "membership_id": mid})))
        cx.commit()
    except Exception as e:
        print(f"[studio-credit] journey_events insert failed: {e!r}", flush=True)
    return {"membership_id": mid, "magic_link_url": magic_link_url}
```

- [ ] **Step 2: Register + configure the actions and init the table (add to the `*_actions` block near line ~20299, after the reviews_actions registration)**

```python
# ── Studio-credit free month: console actions (log claim / approve+grant / reject) ──
from dashboard import studio_credit as _scstore
from dashboard import studio_credit_actions as _sca
with sqlite3.connect(LOG_DB) as _sc_cx:
    _scstore.migrate(_sc_cx)
_sca.configure(grant_fn=_studio_credit_grant_and_notify)
_sca.register()
```

- [ ] **Step 3: Add the list API + console page routes (place near the biofield-reveals routes, ~line 8100)**

```python
@app.route("/api/console/studio-credits", methods=["GET"])
def api_console_studio_credits():
    """List studio-credit claims for console review (default: pending first)."""
    if CONSOLE_SECRET:
        _key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if _key != CONSOLE_SECRET:
            return jsonify({"error": "unauthorized"}), 401
    from dashboard import studio_credit as _sc
    status = request.args.get("status") or None
    with sqlite3.connect(LOG_DB) as cx:
        _sc.migrate(cx)
        claims = _sc.list_claims(cx, status=status)
    return jsonify({"claims": claims})


@app.route("/console/studio-credits", methods=["GET"])
def console_studio_credits_page():
    """Serve the studio-credit review console page."""
    if CONSOLE_SECRET:
        _key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if _key != CONSOLE_SECRET:
            return jsonify({"error": "unauthorized"}), 401
    resp = send_from_directory(STATIC, "console-studio-credits.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp
```

- [ ] **Step 4: Verify app.py still compiles (cannot `import app` — use py_compile)**

Run: `cd /tmp/wt-deploy-chat-0cbb0565 && python3 -m py_compile app.py && echo OK`
Expected: `OK` (no syntax errors). Also re-run the unit suites:
`python3 -m pytest tests/test_studio_credit.py tests/test_studio_credit_actions.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-0cbb0565
git add app.py
git commit -m "feat(studio-credit): wire grant_fn, table init, list API + console route"
```

---

### Task 5: Console page (`static/console-studio-credits.html`)

**Files:**
- Create: `static/console-studio-credits.html`

**Interfaces:**
- Consumes: `GET /api/console/studio-credits` (Task 4); `POST /api/action/studio_credit.add|approve|reject` (existing generic dispatch route at app.py:20264). All calls send the console key.

- [ ] **Step 1: Create the page**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Studio Credits — Console</title>
<style>
  body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1c2430}
  header{background:#0f5132;color:#fff;padding:14px 20px;font-weight:600}
  main{max-width:860px;margin:0 auto;padding:20px}
  .card{background:#fff;border:1px solid #e2e6eb;border-radius:10px;padding:16px;margin-bottom:14px}
  label{display:block;font-size:13px;color:#52606d;margin:8px 0 3px}
  input{width:100%;padding:8px;border:1px solid #cbd2d9;border-radius:6px;box-sizing:border-box}
  button{cursor:pointer;border:0;border-radius:6px;padding:8px 14px;font-weight:600}
  .primary{background:#0f5132;color:#fff}.ok{background:#198754;color:#fff}.no{background:#b42318;color:#fff}
  .muted{color:#697586;font-size:13px}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .pill{font-size:12px;padding:2px 8px;border-radius:999px;background:#eef2f6}
  .warn{background:#fff4e5;border:1px solid #f0c987;padding:8px;border-radius:6px;margin-top:8px;font-size:13px}
</style>
</head>
<body>
<header>Studio Credits — free month of membership</header>
<main>
  <div class="card">
    <strong>Log a claim</strong>
    <div class="muted">A studio-app purchaser emailed their invoice. Record it here, then approve below.</div>
    <label>Email</label><input id="email" type="email" placeholder="buyer@example.com">
    <label>Invoice ref</label><input id="invoice" placeholder="invoice # / Studio.com order id">
    <label>Proof note</label><input id="proof" placeholder="e.g. emailed invoice 6/20">
    <div style="margin-top:12px"><button class="primary" onclick="addClaim()">Add claim</button></div>
    <div id="addmsg" class="muted"></div>
  </div>
  <div id="list"></div>
</main>
<script>
const KEY = new URLSearchParams(location.search).get("key") || "";
const H = {"Content-Type":"application/json","X-Console-Key":KEY};
function esc(s){return (s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}

async function load(){
  const r = await fetch("/api/console/studio-credits?key="+encodeURIComponent(KEY), {headers:H});
  const d = await r.json();
  const el = document.getElementById("list");
  if(!d.claims || !d.claims.length){ el.innerHTML='<div class="card muted">No claims yet.</div>'; return; }
  el.innerHTML = d.claims.map(c=>`
    <div class="card">
      <div class="row">
        <strong>${esc(c.email)}</strong>
        <span class="pill">${esc(c.status)}</span>
        ${c.invoice_ref?`<span class="muted">inv: ${esc(c.invoice_ref)}</span>`:""}
      </div>
      ${c.proof_note?`<div class="muted">${esc(c.proof_note)}</div>`:""}
      ${c.status==="pending"?`<div class="row" style="margin-top:10px">
        <button class="ok" onclick="approve('${c.id}',false)">Approve (30-day comp)</button>
        <button class="no" onclick="reject('${c.id}')">Reject</button>
      </div><div id="w-${c.id}"></div>`:
        c.membership_id?`<div class="muted">membership: ${esc(c.membership_id)}</div>`:""}
    </div>`).join("");
}

async function act(key, params){
  const r = await fetch("/api/action/"+key, {method:"POST", headers:H, body:JSON.stringify(params)});
  return r.json();
}
async function addClaim(){
  const email=document.getElementById("email").value.trim();
  if(!email){document.getElementById("addmsg").textContent="Email required.";return;}
  const res = await act("studio_credit.add", {email,
    invoice_ref:document.getElementById("invoice").value.trim(),
    proof_note:document.getElementById("proof").value.trim()});
  document.getElementById("addmsg").textContent = res.status==="done" ? "Claim logged." : ("Error: "+JSON.stringify(res));
  document.getElementById("email").value=document.getElementById("invoice").value=document.getElementById("proof").value="";
  load();
}
async function approve(id, force){
  const res = await act("studio_credit.approve", {id, force});
  const r = res.result || res;
  if(r && r.ok===false && r.warning==="granted_within_year"){
    document.getElementById("w-"+id).innerHTML =
      `<div class="warn">Already received a studio credit on ${esc((r.granted_at||"").slice(0,10))} (active until ${esc((r.until||"").slice(0,10))}). Override and grant anyway?
       <button class="ok" onclick="approve('${id}',true)">Override &amp; grant</button></div>`;
    return;
  }
  load();
}
async function reject(id){
  const reason = prompt("Reason for rejecting?") || "";
  await act("studio_credit.reject", {id, reason});
  load();
}
load();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify the page is served (manual smoke — only where the app runs)**

Run (in an environment where the app boots): `curl -s "http://localhost:PORT/console/studio-credits?key=$CONSOLE_SECRET" | head -1`
Expected: HTML `<!doctype html>` (200), not a 401 JSON body.
In the sandbox (app won't boot — Pinecone), just confirm the file exists: `test -f static/console-studio-credits.html && echo OK`.

- [ ] **Step 3: Commit**

```bash
cd /tmp/wt-deploy-chat-0cbb0565
git add static/console-studio-credits.html
git commit -m "feat(studio-credit): console review page"
```

---

## Self-Review

**Spec coverage:**
- Pure 30-day comp via `_grant_membership(..., "studio_credit")` → Task 4 grant_fn + Task 2 approve. ✓
- `studio_credit_claims` table with all spec columns → Task 1 `migrate`. ✓
- Console add / pending list / approve / reject → Task 3 actions + Task 4 API + Task 5 page. ✓
- One-per-year guard, force-overridable, expired allowed → Task 2 `studio_credit_granted_within_year` + tests. ✓
- Idempotent re-approve → Task 2 `test_double_approve_is_idempotent`. ✓
- Magic-link email + journey event reused → Task 4 `_studio_credit_grant_and_notify`. ✓
- `source` seam for later automation → Task 1 column, defaulted `console`. ✓
- No new public flag; CONSOLE_SECRET gate → Task 4 routes. ✓
- Not in scope (public form, upload, webhook, convert-to-paid) → not present. ✓

**Placeholder scan:** none — every code/test step is concrete.

**Type consistency:** `grant_fn(cx, email, days) -> {"membership_id","magic_link_url"}` is identical in Task 2 (consumer), Task 3 spy, and Task 4 (`_studio_credit_grant_and_notify`). `approve_claim`/`reject_claim`/`add_claim`/`list_claims` signatures match across store, actions, tests, and app routes. Action keys `studio_credit.add|approve|reject` consistent across Task 3 and Task 5.
