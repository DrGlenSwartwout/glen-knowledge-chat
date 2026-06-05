# BOS Phase 4: Sales & CRM (Home signal over the local work queue)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Light up the Sales & CRM cell on the Home board from the real, local CRM work queue (pending household-dedup candidates, queued merges, unreplied new leads), so the operator sees what CRM needs at a glance. The household/merge actions already dispatch through the audited path (Phase 1c).

**Architecture:** A new `dashboard/crm.py` registers `@signal("crm")` computing the work-queue counts with fast local SQLite reads (all indexed), defensive (gray on any error). No GHL calls in the signal path. `app.py` imports it at startup.

**Builds on:** the merged Business OS (spine + Home + Justus + Orders + Money). New branch `sess/ec0e1f15` off main, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

**Constraint / deferred (decision for Glen):** GHL **writes** (move-deal, add-tag, enroll-workflow, create-opportunity) are blocked from Render by GHL's Cloudflare WAF (the AWS IP is blocked; curl only defeats the JA3/TLS fingerprint). The reliable pattern is record-locally + push from the local Mac (`sync-ghl-leads.py`). So GHL-write CRM actions are NOT built here; they need a local-sync-queue follow-on. The household/merge actions (Phase 1c) already work because their DB writes are local. This phase delivers the CRM signal; the GHL-write pipeline/tag actions are a separate, flagged decision.

---

## File Structure

- `dashboard/crm.py` (new): `crm_signal` registered via `@signal("crm")`, computing the work-queue counts.
- `tests/test_bos_crm.py` (new): unit tests for the signal levels.
- `app.py` (modify): import `dashboard.crm` in the BOS startup block.

---

## Task 1: CRM signal (`dashboard/crm.py`)

**Files:**
- Create: `dashboard/crm.py`
- Test: `tests/test_bos_crm.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bos_crm.py`:

```python
import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _db():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE household_candidates (id INTEGER PRIMARY KEY, status TEXT)")
    cx.execute("CREATE TABLE pending_merges (id INTEGER PRIMARY KEY, status TEXT)")
    cx.execute("CREATE TABLE inbound_leads (id INTEGER PRIMARY KEY, status TEXT, "
               "last_outbound_at TEXT, email TEXT)")
    cx.commit()
    return cx


def test_crm_signal_green_when_empty():
    from dashboard import crm as C, signals as S
    assert C.crm_signal(_db(), None)["level"] == S.GREEN


def test_crm_signal_amber_on_candidates_only():
    from dashboard import crm as C, signals as S
    cx = _db()
    cx.execute("INSERT INTO household_candidates (status) VALUES ('pending')")
    cx.execute("INSERT INTO household_candidates (status) VALUES ('confirmed')")  # not counted
    cx.commit()
    sig = C.crm_signal(cx, None)
    assert sig["level"] == S.AMBER
    assert sig["count"] == 1
    assert "household" in sig["summary"].lower()


def test_crm_signal_red_on_leads_or_merges():
    from dashboard import crm as C, signals as S
    cx = _db()
    cx.execute("INSERT INTO inbound_leads (status, last_outbound_at, email) "
               "VALUES ('pending', '', 'a@b.com')")
    cx.commit()
    assert C.crm_signal(cx, None)["level"] == S.RED  # unreplied new lead is time-sensitive
    cx2 = _db()
    cx2.execute("INSERT INTO pending_merges (status) VALUES ('pending')")
    cx2.commit()
    assert C.crm_signal(cx2, None)["level"] == S.RED


def test_crm_signal_gray_when_tables_missing():
    from dashboard import crm as C, signals as S
    cx = sqlite3.connect(":memory:")  # no CRM tables
    assert C.crm_signal(cx, None)["level"] == S.GRAY


def test_crm_signal_registered():
    from dashboard import crm as C, signals as S
    assert S.SIGNAL_REGISTRY.get("crm") is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_crm.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'dashboard.crm'`).

- [ ] **Step 3: Write the implementation**

Create `dashboard/crm.py`:

```python
"""Business-OS Sales & CRM. Lights up the CRM Home cell from the local work
queue: pending household-dedup candidates, queued merges, and unreplied new
leads. All counts are fast local SQLite reads. The household/merge ACTIONS are
already on the registry (Phase 1c); GHL-write actions are deferred (the WAF
blocks GHL writes from the server)."""
from dashboard.signals import signal as _signal, RED, AMBER, GREEN, GRAY


def crm_signal(cx, actor=None):
    try:
        cand = cx.execute(
            "SELECT COUNT(*) FROM household_candidates WHERE status='pending'").fetchone()[0]
        merges = cx.execute(
            "SELECT COUNT(*) FROM pending_merges WHERE status='pending'").fetchone()[0]
        leads = cx.execute(
            "SELECT COUNT(*) FROM inbound_leads "
            "WHERE (status IS NULL OR status='pending') "
            "  AND (last_outbound_at IS NULL OR last_outbound_at='') "
            "  AND email IS NOT NULL AND email!=''").fetchone()[0]
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}

    total = cand + merges + leads
    if total == 0:
        return {"level": GREEN, "summary": "CRM clear", "top_actions": [], "count": 0}

    bits = []
    if leads:
        bits.append(f"{leads} new lead{'s' if leads != 1 else ''}")
    if cand:
        bits.append(f"{cand} household candidate{'s' if cand != 1 else ''}")
    if merges:
        bits.append(f"{merges} merge{'s' if merges != 1 else ''} to apply")
    # Unreplied leads and queued merges are time-sensitive -> red; dedup-only -> amber.
    level = RED if (leads or merges) else AMBER
    return {"level": level, "summary": ", ".join(bits),
            "top_actions": [{"label": "Open people", "href": "/console"}],
            "count": total}


# Register the signal on import.
crm_signal = _signal("crm")(crm_signal)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_crm.py -q`
Expected: 5 passed.

Run: `python3 -m pytest tests/test_bos_signals.py -q` (the crm signal is now registered; the 1b "money/crm defaults gray" test uses an in-memory DB with no CRM tables, so `crm_signal` returns gray defensively -> still passes).
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/crm.py tests/test_bos_crm.py
git commit -m "feat(bos): Sales & CRM home signal (dedup + lead work queue)"
```

---

## Task 2: Register in `app.py` (verified under doppler)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Import the crm module in the BOS startup block** (near `import dashboard.finance as _bos_finance`):

```python
import dashboard.crm as _bos_crm  # noqa: F401 (registers the CRM home signal)
```

- [ ] **Step 2: Compile + verify under doppler**

Run: `python3 -m py_compile app.py` (OK).
Run:
```bash
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app, sqlite3
from dashboard import signals as S
assert S.SIGNAL_REGISTRY.get("crm") is not None, "crm signal not registered"
cx = sqlite3.connect(app.LOG_DB); cx.row_factory=sqlite3.Row
cells = {c["module"]: c for c in S.aggregate_signals(cx, None)}
print("crm cell:", cells["crm"]["level"], "-", cells["crm"]["summary"])
print("CRM_4_OK")
PY'
rm -rf /tmp/bostest
```
Expected: prints the crm cell level (green/amber/red against the real local CRM queue, or gray if the tables are unexpectedly absent) + `CRM_4_OK`.

Run: `python3 -m pytest tests/test_bos_crm.py tests/test_bos_signals.py tests/test_bos_spine.py -q` (green).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(bos): wire CRM home signal at startup"
```

---

## Self-Review

**Spec coverage** (blueprint 5.3, the buildable-from-server slice):
- CRM Home cell lit from the real work queue -> `crm_signal` (dedup candidates + pending merges + unreplied leads).
- Household/merge actions on the audited path -> already shipped in Phase 1c.

**Deferred (flagged decision):** GHL-write actions (move-deal, add-tag, enroll-workflow, create-opportunity) are blocked from Render by the GHL Cloudflare WAF; they need a local-sync-queue follow-on (record locally -> the local Mac pushes). Not built here.

**Production-only:** the live counts read the real `household_candidates`/`pending_merges`/`inbound_leads` tables on Render. Pure logic is fully unit-tested; the signal is defensive (gray) if a table is missing.

**Placeholder scan:** none.

**Type consistency:** `crm_signal(cx, actor)` matches the `@signal` contract; the cell shape (level/summary/top_actions/count) matches Phase 1b.
