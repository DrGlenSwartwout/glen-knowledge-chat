# BOS Phase 3a: Money & Finance (AR + Money signal + safe actions)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Light up the Money cell on the Home board and give finance an AR-aging view plus the audited, governed finance actions that are safe to ship now: void an invoice and send a payment reminder. Real money-OUT (refunds) is deferred to 3c because QBO has no refund capability wired yet.

**Architecture:** A new `dashboard/finance.py` separates PURE logic (aging computation, summary, signal-from-summary) which is unit-tested, from the QBO-backed reads (`open_invoices`, `finance_summary`, cached with a TTL so the Home board never hammers QBO) which are production-verified. Finance write actions register on the dispatch spine and are governed by the autonomy matrix (finance writes are owner/ops only; Shaira/va is excluded). `void_invoice` lives in `finance.py` (qbo_billing is importable); `send_payment_reminder` registers in `app.py` (it reuses the `_send_inquiry_email` helper).

**Builds on:** the merged Business OS (spine + Home + Orders). New branch `sess/ec0e1f15` off main, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

**Deferred to later sub-phases:** the Finance board UI (3b), `finance.refund_order` (3c, needs a QBO RefundReceipt integration, money_send tier), `finance.issue_invoice` from the board, and pulling the reconciler agent into the console.

---

## File Structure

- `dashboard/finance.py` (new): pure aging/summary/signal logic + cached QBO reads + the Money signal + `finance.void_invoice`.
- `tests/test_bos_finance.py` (new): unit tests for the pure logic + signal-from.
- `app.py` (modify): register the finance module, add `GET /api/finance/ar`, register `finance.send_payment_reminder`, verify under doppler.

---

## Task 1: Pure finance logic (`dashboard/finance.py`)

**Files:**
- Create: `dashboard/finance.py`
- Test: `tests/test_bos_finance.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bos_finance.py`:

```python
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

NOW = datetime(2026, 6, 5, tzinfo=timezone.utc)


def test_aging_filters_zero_balance_and_computes_overdue():
    from dashboard import finance as F
    invs = [
        {"Id": "1", "DocNumber": "1001", "Balance": "70.00", "TotalAmt": "70.00",
         "DueDate": "2026-06-01", "CustomerRef": {"name": "Ann"}},   # 4 days overdue
        {"Id": "2", "DocNumber": "1002", "Balance": "0", "TotalAmt": "50.00",
         "DueDate": "2026-05-01", "CustomerRef": {"name": "Paid"}},  # zero balance -> excluded
        {"Id": "3", "DocNumber": "1003", "Balance": "40.00", "TotalAmt": "40.00",
         "DueDate": "2026-06-20", "CustomerRef": {"name": "Future"}},  # not due yet
    ]
    aged = F.aging(invs, now=NOW)
    assert [a["id"] for a in aged] == ["1", "3"]  # zero-balance dropped, sorted most-overdue first
    assert aged[0]["days_overdue"] == 4
    assert aged[0]["customer"] == "Ann"
    assert aged[1]["days_overdue"] < 0  # future


def test_summarize_totals():
    from dashboard import finance as F
    aged = [{"balance": 70.0, "days_overdue": 4}, {"balance": 40.0, "days_overdue": -5}]
    s = F.summarize(aged, cash_total=1234.5)
    assert s["open_count"] == 2
    assert s["open_total"] == 110.0
    assert s["overdue_count"] == 1
    assert s["overdue_total"] == 70.0
    assert s["cash_total"] == 1234.5


def test_money_signal_from_levels():
    from dashboard import finance as F
    from dashboard import signals as S
    assert F.money_signal_from({"open_count": 0, "overdue_count": 0})["level"] == S.GREEN
    assert F.money_signal_from({"open_count": 3, "overdue_count": 0, "open_total": 200})["level"] == S.AMBER
    red = F.money_signal_from({"open_count": 3, "overdue_count": 2, "overdue_total": 150})
    assert red["level"] == S.RED and red["count"] == 2
    # cash floor breach also goes red
    low = F.money_signal_from({"open_count": 0, "overdue_count": 0, "cash_total": 50}, cash_floor=500)
    assert low["level"] == S.RED
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_finance.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'dashboard.finance'`).

- [ ] **Step 3: Write the pure layer**

Create `dashboard/finance.py`:

```python
"""Business-OS Money & Finance. Pure aging/summary/signal logic (unit-tested) +
cached QBO-backed reads (production-verified) + the Money home signal + the safe
finance actions. Finance writes are owner/ops only (Shaira/va excluded)."""
import os
import time
from datetime import datetime, timezone

from dashboard.signals import signal as _signal, RED, AMBER, GREEN, GRAY


def _parse_date(s):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(str(s)[:10])
        return d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _days_overdue(due_date, now):
    d = _parse_date(due_date)
    if d is None:
        return -9999
    return int((now - d).total_seconds() // 86400)


def aging(invoices, now=None):
    """Pure: QBO Invoice dicts -> AR rows with days_overdue, zero-balance dropped,
    most-overdue first."""
    now = now or datetime.now(timezone.utc)
    out = []
    for inv in invoices or []:
        try:
            bal = float(inv.get("Balance") or 0)
        except (TypeError, ValueError):
            bal = 0.0
        if bal <= 0:
            continue
        due = inv.get("DueDate") or ""
        out.append({
            "id": inv.get("Id"), "doc": inv.get("DocNumber"),
            "customer": (inv.get("CustomerRef") or {}).get("name", ""),
            "email": (inv.get("BillEmail") or {}).get("Address", ""),
            "total": float(inv.get("TotalAmt") or 0), "balance": bal,
            "due_date": due, "days_overdue": _days_overdue(due, now),
        })
    out.sort(key=lambda x: -x["days_overdue"])
    return out


def summarize(aged, cash_total=0.0):
    overdue = [a for a in aged if a["days_overdue"] > 0]
    return {
        "open_count": len(aged),
        "open_total": round(sum(a["balance"] for a in aged), 2),
        "overdue_count": len(overdue),
        "overdue_total": round(sum(a["balance"] for a in overdue), 2),
        "cash_total": round(cash_total or 0.0, 2),
    }


def _cash_floor():
    try:
        return float(os.environ.get("FINANCE_CASH_FLOOR", "0"))
    except (TypeError, ValueError):
        return 0.0


def money_signal_from(summary, cash_floor=0.0):
    oc = summary.get("overdue_count", 0)
    opc = summary.get("open_count", 0)
    cash = summary.get("cash_total", 0)
    low_cash = cash_floor > 0 and cash < cash_floor
    if oc > 0 or low_cash:
        bits = []
        if oc:
            bits.append(f"{oc} overdue (${summary.get('overdue_total', 0):.0f})")
        if low_cash:
            bits.append(f"cash ${cash:.0f} low")
        return {"level": RED, "summary": ", ".join(bits),
                "top_actions": [{"label": "Open finance", "href": "/console/finance"}],
                "count": oc}
    if opc > 0:
        return {"level": AMBER, "summary": f"{opc} open (${summary.get('open_total', 0):.0f})",
                "top_actions": [{"label": "Open finance", "href": "/console/finance"}],
                "count": opc}
    return {"level": GREEN, "summary": "AR clear", "top_actions": [], "count": 0}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_finance.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/finance.py tests/test_bos_finance.py
git commit -m "feat(bos): finance pure logic (AR aging, summary, money signal-from)"
```

---

## Task 2: QBO-backed reads + Money signal + void action (`dashboard/finance.py`)

**Files:**
- Modify: `dashboard/finance.py` (append)
- Test: `tests/test_bos_finance.py` (append)

- [ ] **Step 1: Write the failing test** (append)

```python
def test_money_signal_registered_and_defensive(monkeypatch):
    import sqlite3
    from dashboard import finance as F, signals as S
    # force the QBO-backed summary to blow up -> signal must return GRAY, not raise
    monkeypatch.setattr(F, "finance_summary", lambda: (_ for _ in ()).throw(RuntimeError("qbo down")))
    cx = sqlite3.connect(":memory:")
    cell = F.money_signal(cx, None)
    assert cell["level"] == S.GRAY
    assert S.SIGNAL_REGISTRY.get("money") is not None


def test_void_invoice_action_registered():
    from dashboard import finance as F, actions as A
    a = A.get_action("finance.void_invoice")
    assert a is not None
    assert a.module == "money"
    assert a.permission == ("owner", "ops")  # not va
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bos_finance.py -k "money_signal_registered or void_invoice" -q`
Expected: FAIL (`finance_summary`/`money_signal`/the action not defined).

- [ ] **Step 3: Append the QBO-backed layer + signal + action**

```python
# --- QBO-backed reads (cached) + Money signal + void action ---
from dashboard.actions import action, LOW_WRITE
from dashboard.rbac import OWNER, OPS

_cache = {}


def _cached(key, ttl, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def open_invoices():
    """Cached (10 min): QBO open invoices as AR rows. Production-only (QBO)."""
    def _f():
        from dashboard import qbo_billing as qb
        rs = qb._query("SELECT * FROM Invoice WHERE Balance > '0' ORDER BY DueDate ASC")
        invs = (rs.get("QueryResponse") or {}).get("Invoice") or []
        return aging(invs)
    return _cached("open_invoices", 600, _f)


def finance_summary():
    """Cached: AR summary + cash position (sum of QBO bank balances)."""
    def _f():
        from dashboard import money as M
        aged = open_invoices()
        try:
            cash = sum(a.get("balance", 0) for a in (M.qb_banks().get("accounts") or []))
        except Exception:
            cash = 0.0
        return summarize(aged, cash)
    return _cached("finance_summary", 600, _f)


@_signal("money")
def money_signal(cx, actor=None):
    try:
        s = finance_summary()
    except Exception:
        return {"level": GRAY, "summary": "Not yet wired", "top_actions": [], "count": 0}
    return money_signal_from(s, cash_floor=_cash_floor())


def _void_invoice_exec(params, ctx):
    from dashboard import qbo_billing as qb
    iid = str(params["invoice_id"])
    inv = qb.get_invoice(iid)
    if not inv:
        raise ValueError(f"invoice {iid} not found")
    qb.void_invoice(iid, inv.get("SyncToken"))
    _cache.clear()  # AR changed
    return {"invoice_id": iid, "doc": inv.get("DocNumber"),
            "message": f"Invoice {inv.get('DocNumber', iid)} voided."}


action(key="finance.void_invoice", module="money", title="Void invoice",
       description="Void an unpaid QBO invoice (zeroes it).", risk_tier=LOW_WRITE,
       permission=(OWNER, OPS))(_void_invoice_exec)
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bos_finance.py -q`
Expected: 5 passed.

Run: `python3 -m pytest tests/test_bos_signals.py -q` (the money signal is now registered, but `finance_summary` calls QBO which fails in the bare test env -> defensive GRAY; the 1b "money defaults gray" test still passes).
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/finance.py tests/test_bos_finance.py
git commit -m "feat(bos): QBO AR reads + Money signal + finance.void_invoice"
```

---

## Task 3: app.py wiring (verified under doppler)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Register the finance module in the BOS startup block** (near `import dashboard.orders as _bos_orders`):

```python
import dashboard.finance as _bos_finance  # noqa: F401 (registers money signal + finance actions)
```

- [ ] **Step 2: Add the AR endpoint** (near the other `bos_*` routes):

```python
@app.route("/api/finance/ar", methods=["GET"])
def bos_finance_ar():
    actor = _bos_actor()
    if actor is None:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    try:
        rows = _bos_finance.open_invoices()
        summary = _bos_finance.finance_summary()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502
    return jsonify({"ok": True, "data": rows, "summary": summary})
```

- [ ] **Step 3: Register `finance.send_payment_reminder`** (after `_send_inquiry_email` is defined; search `def _send_inquiry_email`). Add below it:

```python
def _register_finance_email_actions():
    from dashboard.actions import action as _act, get_action as _get
    from dashboard import rbac as _r
    if _get("finance.send_payment_reminder"):
        return

    def _reminder(params, ctx):
        email = (params.get("email") or "").strip()
        if not email:
            raise ValueError("email required")
        doc = params.get("doc") or params.get("invoice_id") or ""
        amount = params.get("amount")
        amt = f" of ${float(amount):.2f}" if amount not in (None, "") else ""
        subject = "A quick note about your invoice"
        body = (f"Aloha,\n\nThis is a friendly reminder that invoice {doc} "
                f"{('for a balance' + amt) if amt else ''} is still open. "
                f"You can reply here with any questions.\n\nIn wellness,\nDr. Glen")
        ok = _send_inquiry_email(to_email=email, subject=subject, body=body,
                                 reply_to=RM_INBOUND_INQUIRY_EMAIL)
        return {"email": email, "doc": doc, "sent": bool(ok),
                "message": f"Payment reminder {'sent to' if ok else 'failed for'} {email}."}

    _act(key="finance.send_payment_reminder", module="money",
         title="Send payment reminder", description="Email a customer about an open invoice.",
         risk_tier=__import__("dashboard.actions", fromlist=["LOW_WRITE"]).LOW_WRITE,
         permission=(_r.OWNER, _r.OPS, _r.VA))(_reminder)


_register_finance_email_actions()
```

- [ ] **Step 4: Compile + verify under doppler**

Run: `python3 -m py_compile app.py` (OK).
Run:
```bash
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app, sqlite3
from dashboard import actions as A, signals as S, finance as F
for k in ("finance.void_invoice","finance.send_payment_reminder"):
    assert A.get_action(k) is not None, "missing "+k
# money signal is live (QBO query runs against real creds): money cell no longer gray on success
cx = sqlite3.connect(app.LOG_DB); cx.row_factory=sqlite3.Row
cells = {c["module"]: c for c in S.aggregate_signals(cx, None)}
print("money cell:", cells["money"]["level"], "-", cells["money"]["summary"])
print("AR summary:", F.finance_summary())
print("FINANCE_3A_OK")
PY'
rm -rf /tmp/bostest
```
Expected: prints `FINANCE_3A_OK`, the money cell level (green/amber/red depending on real AR), and the AR summary. No assertion error. (If QBO is briefly unreachable the money cell may read gray; that is acceptable defensiveness, not a failure.)

Run: `python3 -m pytest tests/test_bos_finance.py tests/test_bos_signals.py tests/test_bos_spine.py -q` (green).

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(bos): finance AR endpoint + payment-reminder action + money signal wiring"
```

---

## Self-Review

**Spec coverage** (blueprint 5.2, the now-safe slice):
- AR aging / unpaid invoices -> `aging` + `open_invoices` + `/api/finance/ar`.
- Cash position -> `finance_summary` (QBO bank balances).
- Money signal lights up the Home cell -> `@signal("money")` (defensive gray on QBO error).
- Safe finance actions through the audited path -> `finance.void_invoice` (owner/ops) + `finance.send_payment_reminder` (owner/ops/va).

**Out of scope (later):** `finance.refund_order` (needs a QBO RefundReceipt integration, money_send tier, 3c); `finance.issue_invoice`/`record_payment` from the board; the Finance board UI (3b); reconciler-in-console.

**Production-only:** the QBO AR query + bank read run only with real QBO creds; the money signal is defensive (gray) on any failure so it never breaks the Home board. Pure logic (aging/summary/signal-from) is fully unit-tested.

**Placeholder scan:** none.

**Type consistency:** `aging`/`summarize`/`money_signal_from`, the AR row keys (`id`, `doc`, `customer`, `balance`, `days_overdue`), the summary keys, the signal cell shape, and the action keys (`finance.void_invoice`, `finance.send_payment_reminder`) are consistent across Tasks 1-3 and the Phase 1a/1b contracts.
```
