# Recommendation Source Tracking — Phase 2a (order-line source capture) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture, at order time, which source produced each order line — a `source` key on each `items_json` line (manual per-line picker + portal buttons), carried through the invoice render whitelist, and emit an idempotent "acted-on" `recommendation_events` event per sourced line at the single `upsert_order` choke point.

**Architecture:** `source` rides inside `items_json` (no schema change), modeled exactly on the existing free-text per-line `note`. It is set by the manual order-entry picker (`static/order-new.html` → `_price_inhouse_invoice`) and by the portal add-to-invoice buttons (FF→`scan`, support→`intake`), carried to the customer invoice by `_invoice_line_view`, and turned into a recommendation event in `dashboard/orders.py::upsert_order` (every creation path lands there). The event emission is failure-isolated so it can never break order creation.

**Tech Stack:** Python 3 / Flask (`app.py`), `dashboard/*.py`, SQLite (`LOG_DB`), pytest (app tests run under `doppler run -- python3`), vanilla-JS static page.

## Global Constraints

- **MONEY PATH.** `upsert_order` and the order builders are the checkout/invoice path. Every change here must be additive and failure-isolated: `source` is OPTIONAL (absent → byte-identical prior behavior), and the event emission is wrapped in `try/except` so a `recommendation_events` failure can NEVER raise out of `upsert_order`.
- **`source` rides in `items_json`** — no `orders` schema change. Model it on the existing per-line `note` (set in `_price_inhouse_invoice`, carried in `_invoice_line_view`).
- **The render whitelist drops unknown keys.** `_invoice_line_view` (app.py:41369) rebuilds each line from a fixed key set; `source` MUST be copied into `out` there or it vanishes from the customer invoice. (It still persists in raw `items_json` regardless — the whitelist is render-only.)
- **`source` values** come from the registry (`self`, `biofield`, `scan`, `intake`, `chat`); the manual picker defaults to `self`.
- **Acted-on ≠ purchased.** The event emitted here fires at order PLACEMENT (`status` proposed/new), keyed `origin_ref = order id`. It is a different event from the paid `purchased` provenance (Phase 1 `ingest_purchased`, keyed on `pay_status='paid'`). Do not conflate.
- **Idempotent.** `record_event` is `INSERT OR IGNORE` on `(client_email, product_key, source_key, origin_ref)`; `upsert_order` re-runs on edit, so re-emission with `origin_ref=order id` is a no-op.
- **CI known_failures ratchet; never run the bare full suite (sends live email).** Run named feature tests via `doppler run -- python3 -m pytest ...`.

---

### Task 1: Persist `source` on inhouse lines + carry it through the render whitelist

**Files:**
- Modify: `app.py::_price_inhouse_invoice` (line dict at ~40123-40140), `app.py::_invoice_line_view` (~41374-41380)
- Test: `tests/test_line_source_whitelist.py`

**Interfaces:**
- Produces: an `items_json` line may carry `"source": "<key>"`; `_invoice_line_view` preserves it into its output dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_line_source_whitelist.py
import app as app_module


def test_invoice_line_view_carries_source():
    out = app_module._invoice_line_view(
        {"slug": "neuro-magnesium", "name": "Neuro Magnesium", "qty": 1,
         "unit_cents": 7000, "line_cents": 7000, "source": "biofield"})
    assert out["source"] == "biofield"


def test_invoice_line_view_omits_source_when_absent():
    out = app_module._invoice_line_view(
        {"slug": "neuro-magnesium", "name": "Neuro Magnesium", "qty": 1,
         "unit_cents": 7000, "line_cents": 7000})
    assert "source" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_line_source_whitelist.py -q`
Expected: FAIL — `KeyError: 'source'` (whitelist drops it).

- [ ] **Step 3: Implement**

In `app.py::_invoice_line_view`, right after the `note` block (~line 41380, before the `membership` early-return), add:

```python
    # Per-line recommendation source (biofield/scan/self/…). Whitelist-carried like note.
    if l.get("source"):
        out["source"] = l.get("source")
```

In `app.py::_price_inhouse_invoice`, right after the `note` block (~line 40129), add:

```python
        _src = (ln.get("source") or "").strip()
        if _src:
            rec["source"] = _src
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_line_source_whitelist.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py tests/test_line_source_whitelist.py
git commit -m "feat(rec): carry per-line source through inhouse builder + invoice whitelist"
```

---

### Task 2: Manual order-entry per-line source picker

**Files:**
- Modify: `static/order-new.html`
- Test: `tests/test_order_new_source_picker.py`

**Interfaces:**
- Consumes: nothing new. Produces: each manual line submits an optional `source` (default `"self"`) in the `/api/orders/manual` body.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_order_new_source_picker.py
import app as app_module


def test_order_new_page_has_source_picker():
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    body = c.get("/orders/new").get_data(as_text=True)
    # a per-line Source select + it is threaded into the POST payload
    assert "setSource" in body
    assert "b.source" in body            # linesPayload includes source
    assert ">self<" in body or "'self'" in body or '"self"' in body   # self default option
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_order_new_source_picker.py -q`
Expected: FAIL (markers absent).

- [ ] **Step 3: Implement** (in `static/order-new.html`)

- Add a **Source** column header near the existing "Note (shown to customer)" header (~line 105).
- In the `LINES` model (~line 184), each line gets `source` (default `"self"` on `addLine()` ~line 249).
- In `renderLines()` (~line 324), render a `<select>` per row next to the note cell (~line 336-337) with options `self, biofield, scan, intake, chat` (a small `SOURCE_OPTS` array), current value `l.source`, `onchange="setSource(<i>, this.value)"`.
- Add `setSource(i, v){ LINES[i].source = v; }` mirroring `setNote()` (~line 351).
- In `linesPayload()` (~line 348), add: `if (l.source && l.source !== 'self') b.source = l.source; else b.source = 'self';` — always send `source` (default `self`) so manual lines are explicitly coded.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_order_new_source_picker.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add static/order-new.html tests/test_order_new_source_picker.py
git commit -m "feat(rec): per-line source picker in manual order entry (default self)"
```

---

### Task 3: Portal add-to-invoice buttons stamp their source

**Files:**
- Modify: `app.py` (FF add-to-invoice line ~20363 → `source="scan"`; support add-to-invoice line ~21045 → `source="intake"`)
- Test: `tests/test_portal_add_to_invoice_source.py`

**Interfaces:**
- Produces: the order line created by the FF portal button carries `source="scan"`; the support-program button carries `source="intake"`.

- [ ] **Step 1: Write the failing test**

Because these endpoints require portal-token setup, test at the line-building level: assert the built line dict includes the right `source` before `upsert_order`. If the line is built inline (not a helper), extract the one-line change and assert via a focused endpoint call with a seeded portal token. Minimal form — assert the source string is present at the call site by exercising the endpoint with a stubbed `upsert_order` capturing `items`:

```python
# tests/test_portal_add_to_invoice_source.py
import app as app_module


def test_ff_add_to_invoice_line_tagged_scan(monkeypatch):
    captured = {}
    def fake_upsert(cx, **kw):
        captured.update(kw)
        return 1
    monkeypatch.setattr(app_module._bos_orders, "upsert_order", fake_upsert)
    # ... seed a portal token for a client email + a scan_date with FF items,
    #     then POST the FF add-to-invoice endpoint; assert every built line has source="scan".
    # (The implementer wires the token/fixture per the existing portal test helpers.)
    # assert all(li.get("source") == "scan" for li in captured["items"])
```

Note to implementer: mirror the existing portal-endpoint tests (grep `tests/` for `add_to_invoice` / portal-token fixtures) for the seeding boilerplate; the assertion is that each built line carries the source. If no such fixture harness exists, assert the source constant is applied by unit-testing the small line-building helper you factor out.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_portal_add_to_invoice_source.py -q`
Expected: FAIL (source not set).

- [ ] **Step 3: Implement**

- FF add-to-invoice (`app.py` ~20363), the line dict `{"slug","name","qty":1,"unit_cents","line_cents"}` → add `"source": "scan"`.
- Support-program add-to-invoice (`app.py` ~21045) → add `"source": "intake"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_portal_add_to_invoice_source.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add app.py tests/test_portal_add_to_invoice_source.py
git commit -m "feat(rec): portal FF/support add-to-invoice buttons stamp line source"
```

---

### Task 4: Emit the acted-on event at `upsert_order` (money-path safe)

**Files:**
- Modify: `dashboard/orders.py::upsert_order`
- Test: `tests/test_upsert_order_emits_source_event.py`

**Interfaces:**
- Consumes: `dashboard.recommendation_events.record_event`.
- Produces: after an order is upserted, each `items` line with a non-empty `source` + `slug` yields one `recommendation_events` row `(email, slug, source, origin_ref=order id)`. Failure-isolated: a `record_event` error never propagates out of `upsert_order`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_upsert_order_emits_source_event.py
import sqlite3
from dashboard import orders, recommendation_events as re


def _cx():
    cx = sqlite3.connect(":memory:")
    orders.init_orders_table(cx)
    re.init_recommendation_events(cx)
    return cx


def test_sourced_lines_emit_events_unsourced_do_not():
    cx = _cx()
    oid = orders.upsert_order(
        cx, source="in-house", external_ref="INH-1", email="A@B.com",
        items=[{"slug": "neuro-magnesium", "qty": 1, "source": "biofield"},
               {"slug": "immune-modulation", "qty": 1}],   # no source -> no event
        status="proposed")
    ev = re.list_events(cx, "a@b.com")
    assert len(ev) == 1
    assert ev[0]["source_key"] == "biofield" and ev[0]["product_key"] == "neuro-magnesium"
    assert ev[0]["origin_ref"] == str(oid)
    # idempotent: re-upsert (edit) emits nothing new
    orders.upsert_order(cx, source="in-house", external_ref="INH-1", email="a@b.com",
                        items=[{"slug": "neuro-magnesium", "qty": 1, "source": "biofield"}],
                        status="proposed")
    assert len(re.list_events(cx, "a@b.com")) == 1


def test_record_event_failure_never_breaks_order(monkeypatch):
    cx = _cx()
    def boom(*a, **k):
        raise RuntimeError("events down")
    monkeypatch.setattr(re, "record_event", boom)
    # order creation must still succeed
    oid = orders.upsert_order(cx, source="in-house", external_ref="INH-2", email="a@b.com",
                              items=[{"slug": "x", "qty": 1, "source": "self"}], status="proposed")
    assert oid is not None
    got = orders.get_order(cx, oid)
    assert got and got["email"] == "a@b.com"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_upsert_order_emits_source_event.py -q`
Expected: FAIL — `test_sourced_lines_emit_events...` (no events emitted).

- [ ] **Step 3: Implement**

In `dashboard/orders.py::upsert_order`, factor the emission into a local helper called on BOTH the update-return path and the insert-return path, once the order id is known. Add before the two `return` points:

```python
def _emit_source_events(cx, order_id, email, items):
    """Best-effort: one acted-on recommendation event per sourced line. NEVER raises."""
    try:
        from dashboard import recommendation_events
        for line in (items or []):
            src = (line.get("source") or "").strip()
            slug = (line.get("slug") or "").strip()
            if not src or not slug:
                continue
            recommendation_events.record_event(
                cx, email, slug, src, occurred_at=_now(), origin_ref=str(order_id),
                commit=False)
        cx.commit()
    except Exception:
        pass
```

Call `_emit_source_events(cx, row[0], email, items)` just before `return row[0]` (update path), and `_emit_source_events(cx, <new_id>, email, items)` just before the insert path's `return <new_id>`. Only emit when `items is not None` (an update that doesn't touch items shouldn't re-emit off a stale list — guard: `if items is not None:`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp/wt-deploy-chat-e42ec522 && doppler run -- python3 -m pytest tests/test_upsert_order_emits_source_event.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-e42ec522 && git add dashboard/orders.py tests/test_upsert_order_emits_source_event.py
git commit -m "feat(rec): emit acted-on source event at upsert_order (failure-isolated)"
```

---

## Self-review checklist (controller, before dispatch)

- Money-path safety: every change additive; `source` optional; event emission `try/except` + `commit=False` batched; a `record_event` failure can't break `upsert_order` (Task 4 test proves it).
- Whitelist: `source` copied into `_invoice_line_view` output (else dropped from invoice render).
- Acted-on (order placed, `origin_ref=order id`) is distinct from purchased (paid); idempotent on re-edit.
- Manual picker defaults to `self`; portal buttons stamp scan/intake.

## Not in 2a (later slices)

- 2b: the client portal UI (collapsible sections w/ remembered state, icon rows + counts, per-product hide + operator note + client note), and switching `process_strip` to prefer per-line `source`.
- 2c: product-page "add to my portal" (`self`) + product-page-visitor→client identity.
- 2d: reveal-click + on-page engagement capture for biofield/scan/chat.
