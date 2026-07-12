# CNS Tracking Harvest Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a CNS ship-to name doesn't resolve to a GHL contact, harvest the buyer from the order emails, add them to GHL (onboarding only genuine new storefront buyers), and fill the draft `To:` only on a precision-safe single-email match.

**Architecture:** New pure module `dashboard/order_harvest.py` (source detection + per-source customer-block parsers + a precision-gated `harvest_buyer`). `cns_tracking_watcher.py::handle_confirmation` gains two optional injected seams — `harvest_fn` (read-only identity lookup) and `persist_contact` (impure GHL upsert/onboard) — so all decision logic stays unit-testable with fakes. When `harvest_fn is None`, behavior is byte-identical to today.

**Tech Stack:** Python 3, stdlib `re`/`email`, sqlite3, existing `dashboard.tracking` + `dashboard.ghl`. Tests: pytest, synthetic fixtures, no network.

## Global Constraints

- No network in tests — inject fakes for gmail search / GHL / drafts.
- No real customer PII in committed fixtures — synthetic names/emails only.
- Never parse the MERCHANT block: `Healing Oasis` / `351 Wailuku Drive` / `support@remedymatch.com` / `(808) 217-9647`.
- Strict no-regression: no GHL match + no precise harvest → blank `To:`, `status="needs_review"` (unchanged).
- Onboarding (pipeline + workflow enroll) fires ONLY when `source == "neworder"` AND the GHL contact was newly created. Everything else is records-only.
- Dry-run performs read-only harvest preview but no `ghl_upsert`/onboarding/drafts/DB writes.
- GHL helper signature (verbatim): `ghl_upsert_contact(email, first_name="", last_name="", phone="", source_tag="", extra_tags=None, custom_fields=None) -> (contact_id, created_bool, error)`.

---

### Task 1: `order_harvest.py` — source detection + customer-block parsers

**Files:**
- Create: `dashboard/order_harvest.py`
- Test: `tests/test_order_harvest.py`

**Interfaces:**
- Produces:
  - `detect_source(sender: str) -> str | None` — returns one of `"eprocessing"`, `"authorizenet"`, `"neworder"`, `"invoice"`, else `None`.
  - `parse_order_email(source: str, body: str) -> dict | None` — returns
    `{"source": str, "name": str|None, "email": str|None, "phone": str|None, "products": list[str]}`
    from the CUSTOMER block, or `None` if `source` is unknown.
  - `_norm_name(s: str) -> str` — lowercase, collapse whitespace, strip punctuation (reused by Task 2).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_order_harvest.py
from dashboard.order_harvest import detect_source, parse_order_email, _norm_name

def test_detect_source():
    assert detect_source("Transactions@prod.eprocessingnetwork.com") == "eprocessing"
    assert detect_source("noreply@mail.authorize.net") == "authorizenet"
    assert detect_source("support@remedymatch.com") == "neworder"
    assert detect_source("Glen Swartwout <drglenswartwout@gmail.com>") == "invoice"
    assert detect_source("noreply-ecns@usps.com") is None

# eProcessing: merchant "Email:" block first, then customer "Name:"/"E-Mail:"
EPROC = """This message is to confirm that a transaction has been processed
Remedy Match LLC may be contacted at:
   Address: 351 Wailuku Drive
     Phone: (808) 217-9647
     Email: support@remedymatch.com
Order information is as follows:
    Invoice: 480
    Name: Jane Buyer
    E-Mail: jane.buyer@example.com
    Card Type: AX
"""

def test_parse_eprocessing_customer_block_not_merchant():
    r = parse_order_email("eprocessing", EPROC)
    assert r["name"] == "Jane Buyer"
    assert r["email"] == "jane.buyer@example.com"   # customer, NOT support@remedymatch.com

# remedymatch New-order (storefront): customer line + product remedy links
NEWORDER = """<p>New order : #1042</p>
<p>customer: Sam Storefront (sam.storefront@example.com)</p>
<p>Delivery address: Sam Storefront, 5 Main St, Reno, NV 89501</p>
<table><tr><td><a href="/remedies/204-ocuheal-eye-drops">OcuHeal Eye Drops</a></td><td>2</td></tr></table>
"""

def test_parse_neworder_customer_and_products():
    r = parse_order_email("neworder", NEWORDER)
    assert r["name"] == "Sam Storefront"
    assert r["email"] == "sam.storefront@example.com"
    assert r["products"] == ["OcuHeal Eye Drops"]

# Authorize.net Merchant Email Receipt: merchant block, then customer fields
AUTHNET = """Merchant: Remedy Match LLC
support@remedymatch.com
Customer Information
First Name: Carl
Last Name: Client
Email: carl.client@example.com
Phone: 555-222-3333
"""

def test_parse_authorizenet():
    r = parse_order_email("authorizenet", AUTHNET)
    assert r["name"] == "Carl Client"
    assert r["email"] == "carl.client@example.com"
    assert r["phone"] == "555-222-3333"

def test_parse_invoice_uses_to_header():
    body = "To: Deb Buyer <deb.buyer@example.com>\nSubject: Your Remedy Match invoice INH-77\n"
    r = parse_order_email("invoice", body)
    assert r["email"] == "deb.buyer@example.com"
    assert r["name"] == "Deb Buyer"

def test_parse_unknown_source_returns_none():
    assert parse_order_email("mystery", "whatever") is None

def test_norm_name():
    assert _norm_name("J. Morris  Williams") == "j morris williams"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-11a8b606 && python -m pytest tests/test_order_harvest.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.order_harvest`).

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/order_harvest.py
"""Harvest a buyer's contact from the order emails in the connected mailbox.

Pure parsing + precision gate; no Gmail/GHL calls here (those are injected by the
watcher). Always reads the CUSTOMER block, never the merchant block
(Healing Oasis / 351 Wailuku Drive / support@remedymatch.com / (808) 217-9647).
"""
from __future__ import annotations
import re
from typing import Callable, Optional

_MERCHANT_EMAIL = "support@remedymatch.com"


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def detect_source(sender: str) -> Optional[str]:
    s = (sender or "").lower()
    if "eprocessingnetwork.com" in s:
        return "eprocessing"
    if "mail.authorize.net" in s:
        return "authorizenet"
    if "support@remedymatch.com" in s:
        return "neworder"
    if "drglenswartwout@gmail.com" in s:
        return "invoice"
    return None


def _clean(v: Optional[str]) -> Optional[str]:
    v = (v or "").strip()
    return v or None


def parse_order_email(source: str, body: str) -> Optional[dict]:
    body = body or ""
    name = email = phone = None
    products: list[str] = []

    if source == "eprocessing":
        # Customer labels are "Name:" and "E-Mail:" (hyphen). The merchant block
        # uses "Email:" (no hyphen) = support@remedymatch.com — never matched here.
        m = re.search(r"^\s*Name:\s*(.+?)\s*$", body, re.M)
        name = _clean(m.group(1)) if m else None
        m = re.search(r"E-Mail:\s*([^\s<>]+@[^\s<>]+)", body)
        email = _clean(m.group(1)) if m else None

    elif source == "neworder":
        m = re.search(r"customer:\s*(.+?)\s*\(([^)]+@[^)]+)\)", body, re.I)
        if m:
            name = _clean(m.group(1))
            email = _clean(m.group(2))
        products = [re.sub(r"<[^>]+>", "", t).strip()
                    for t in re.findall(r'/remedies/[^"]+">([^<]+)</a>', body)]

    elif source == "authorizenet":
        f = re.search(r"First Name:\s*(.+?)\s*$", body, re.M)
        l = re.search(r"Last Name:\s*(.+?)\s*$", body, re.M)
        if f or l:
            name = _clean(" ".join(x.group(1).strip() for x in (f, l) if x))
        m = re.search(r"(?<!support@remedymatch\.com)\bEmail:\s*([^\s<>]+@[^\s<>]+)", body)
        email = _clean(m.group(1)) if m and m.group(1) != _MERCHANT_EMAIL else email
        m = re.search(r"Phone:\s*([0-9()+\-.\s]{7,})", body)
        phone = _clean(m.group(1)) if m else None

    elif source == "invoice":
        m = re.search(r"^To:\s*(.*?)\s*<([^>]+@[^>]+)>", body, re.M)
        if m:
            name = _clean(m.group(1))
            email = _clean(m.group(2))
        else:
            m = re.search(r"^To:\s*([^\s<>]+@[^\s<>]+)", body, re.M)
            email = _clean(m.group(1)) if m else None
    else:
        return None

    if email and email.lower() == _MERCHANT_EMAIL:
        email = None
    return {"source": source, "name": name, "email": email,
            "phone": phone, "products": products}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-11a8b606 && python -m pytest tests/test_order_harvest.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-11a8b606
git add dashboard/order_harvest.py tests/test_order_harvest.py
git commit -m "feat: order-email customer-block parsers + source detection

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `harvest_buyer` — precision gate

**Files:**
- Modify: `dashboard/order_harvest.py`
- Test: `tests/test_order_harvest.py`

**Interfaces:**
- Consumes: `detect_source`, `parse_order_email`, `_norm_name` (Task 1).
- Produces: `harvest_buyer(gmail_search, ship_to_name) -> dict | None`
  - `gmail_search: Callable[[str], list[dict]]` where each dict is `{"sender": str, "body": str}`.
  - Returns `{"email", "first", "last", "phone", "source", "products"}` or `None`.
  - Gate: among candidates whose `_norm_name(customer name) == _norm_name(ship_to_name)` and that have an email, accept ONLY if there is exactly **one distinct** email. Zero, name-mismatch-only, or ≥2 distinct emails → `None`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_order_harvest.py
from dashboard.order_harvest import harvest_buyer

def _search_returning(msgs):
    return lambda query: msgs

def test_harvest_single_exact_match_hits():
    msgs = [{"sender": "Transactions@prod.eprocessingnetwork.com",
             "body": "Name: J Morris Williams\nE-Mail: jmw@example.com\n"}]
    r = harvest_buyer(_search_returning(msgs), "J Morris Williams")
    assert r["email"] == "jmw@example.com"
    assert r["first"] == "J" and r["last"] == "Morris Williams"
    assert r["source"] == "eprocessing"

def test_harvest_name_mismatch_returns_none():
    msgs = [{"sender": "Transactions@prod.eprocessingnetwork.com",
             "body": "Name: Someone Else\nE-Mail: else@example.com\n"}]
    assert harvest_buyer(_search_returning(msgs), "J Morris Williams") is None

def test_harvest_two_distinct_emails_returns_none():
    msgs = [
        {"sender": "Transactions@prod.eprocessingnetwork.com",
         "body": "Name: Pat Lee\nE-Mail: pat1@example.com\n"},
        {"sender": "support@remedymatch.com",
         "body": "customer: Pat Lee (pat2@example.com)\n"},
    ]
    assert harvest_buyer(_search_returning(msgs), "Pat Lee") is None

def test_harvest_same_email_two_sources_prefers_neworder():
    msgs = [
        {"sender": "Transactions@prod.eprocessingnetwork.com",
         "body": "Name: Pat Lee\nE-Mail: pat@example.com\n"},
        {"sender": "support@remedymatch.com",
         "body": 'customer: Pat Lee (pat@example.com)\n<a href="/remedies/1-x">X</a>'},
    ]
    r = harvest_buyer(_search_returning(msgs), "Pat Lee")
    assert r["email"] == "pat@example.com"
    assert r["source"] == "neworder"          # storefront wins → enables onboarding
    assert r["products"] == ["X"]

def test_harvest_merchant_only_returns_none():
    msgs = [{"sender": "noreply-ecns@usps.com", "body": "irrelevant"}]
    assert harvest_buyer(_search_returning(msgs), "Anyone") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-11a8b606 && python -m pytest tests/test_order_harvest.py -q`
Expected: FAIL (`harvest_buyer` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/order_harvest.py

def _split_name(full: str) -> tuple[str, str]:
    parts = (full or "").split()
    if not parts:
        return "", ""
    return parts[0], " ".join(parts[1:])


def harvest_buyer(gmail_search: Callable[[str], list], ship_to_name: str):
    """Return the buyer's contact ONLY on a precision-safe match, else None."""
    target = _norm_name(ship_to_name)
    if not target:
        return None
    candidates = []
    for msg in gmail_search(ship_to_name) or []:
        src = detect_source(msg.get("sender", ""))
        if not src:
            continue
        parsed = parse_order_email(src, msg.get("body", ""))
        if not parsed or not parsed.get("email"):
            continue
        if _norm_name(parsed.get("name") or "") != target:
            continue
        candidates.append(parsed)
    if not candidates:
        return None
    distinct = {c["email"].lower() for c in candidates}
    if len(distinct) != 1:
        return None                      # ambiguous → never guess
    # Prefer a storefront "neworder" candidate (carries products + enables onboarding)
    best = next((c for c in candidates if c["source"] == "neworder"), candidates[0])
    first, last = _split_name(best.get("name") or ship_to_name)
    return {"email": best["email"], "first": first, "last": last,
            "phone": best.get("phone"), "source": best["source"],
            "products": best.get("products") or []}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-11a8b606 && python -m pytest tests/test_order_harvest.py -q`
Expected: PASS (13 tests total).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-11a8b606
git add dashboard/order_harvest.py tests/test_order_harvest.py
git commit -m "feat: precision-gated harvest_buyer (single-email-or-nothing)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire harvest into `handle_confirmation` + watcher closures

**Files:**
- Modify: `cns_tracking_watcher.py` (`handle_confirmation` + new `make_harvest_fn` / `make_persist_contact` + `main` wiring)
- Test: `tests/test_tracking_watcher.py` (create)

**Interfaces:**
- Consumes: `harvest_buyer` (Task 2); `dashboard.ghl.ghl_upsert_contact/ghl_add_to_pipeline/ghl_enroll_workflow`.
- Produces:
  - `handle_confirmation(html, msg_id, cx, find_contact, draft_fn, harvest_fn=None, persist_contact=None, dry_run=True)`
    - `harvest_fn: Callable[[str], dict|None]` — read-only identity lookup (returns Task-2 shape).
    - `persist_contact: Callable[[dict, str], dict]` — impure GHL; returns `{"contact_id": str|None, "onboarded": bool}`. Called only when `not dry_run` and an identity with an email was harvested.
  - `make_persist_contact() -> Callable` closure binding the GHL helpers with the onboarding rule.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tracking_watcher.py
import sqlite3
from cns_tracking_watcher import handle_confirmation
from dashboard.tracking import init_tracking_schema

CONF = """<html><body>
  <p>Order #: <a href="https://cnsb.usps.com/confirmation-page?orderUUID=019e0000-0000-7000-8000-000000000001">019e0000-0000-7000-8000-000000000001</a></p>
  <table class="item-contents-table"><tbody><tr><td class="item-contents-column"><table><tbody><tr><td>
    <p class="bold"> Priority Mail&#174;</p>
    <a href="x">4200000000009405530109355300000001</a>
    <p class="bold">Shipped To:</p>
    <p class="pt-5">New Buyer</p>
    <p class="pt-5">1 A St</p>
    <p class="pt-5">RENO NV 89501-0001 US</p>
  </td></tr></tbody></table></td></tr></tbody></table>
</body></html>"""

def _cx():
    cx = sqlite3.connect(":memory:"); init_tracking_schema(cx); return cx

def test_no_ghl_match_precise_harvest_fills_to_and_persists():
    calls = {}
    def find_contact(name): return None
    def harvest_fn(name):
        return {"email": "new@example.com", "first": "New", "last": "Buyer",
                "phone": None, "source": "eprocessing", "products": []}
    def persist(identity, name):
        calls["persist"] = (identity, name); return {"contact_id": "C1", "onboarded": False}
    drafts = []
    def draft_fn(to, subject, html, text): drafts.append(to); return "D1"
    res = handle_confirmation(CONF, "M1", _cx(), find_contact, draft_fn,
                              harvest_fn=harvest_fn, persist_contact=persist, dry_run=False)[0]
    assert res["to"] == "new@example.com"
    assert res["confidence"] == "harvested"
    assert res["status"] == "drafted"
    assert drafts == ["new@example.com"]
    assert calls["persist"][0]["email"] == "new@example.com"

def test_no_ghl_match_no_harvest_stays_needs_review():
    def find_contact(name): return None
    def harvest_fn(name): return None
    def draft_fn(to, subject, html, text): return "D2"
    res = handle_confirmation(CONF, "M2", _cx(), find_contact, draft_fn,
                              harvest_fn=harvest_fn, persist_contact=lambda i, n: {}, dry_run=False)[0]
    assert res["to"] == "(blank — needs review)"
    assert res["status"] == "needs_review"

def test_harvest_fn_none_is_legacy_behavior():
    def find_contact(name): return None
    def draft_fn(to, subject, html, text): return "D3"
    res = handle_confirmation(CONF, "M3", _cx(), find_contact, draft_fn, dry_run=False)[0]
    assert res["status"] == "needs_review"

def test_dry_run_previews_harvest_without_persisting():
    def find_contact(name): return None
    def harvest_fn(name):
        return {"email": "d@example.com", "first": "New", "last": "Buyer",
                "phone": None, "source": "neworder", "products": []}
    def persist(identity, name): raise AssertionError("must not persist in dry-run")
    res = handle_confirmation(CONF, "M4", _cx(), find_contact, lambda **k: None,
                              harvest_fn=harvest_fn, persist_contact=persist, dry_run=True)[0]
    assert res["to"] == "d@example.com"
    assert res["confidence"] == "harvested"
    assert res["action"] == "would draft (harvested)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-11a8b606 && python -m pytest tests/test_tracking_watcher.py -q`
Expected: FAIL (`handle_confirmation` has no `harvest_fn` kwarg).

- [ ] **Step 3: Implement — replace the match block in `handle_confirmation`**

In `cns_tracking_watcher.py`, change the signature and the per-shipment resolution.
Replace lines 103–119 (`def handle_confirmation...` through the `email = build_tracking_email(...)` line) with:

```python
def handle_confirmation(html, msg_id, cx, find_contact, draft_fn,
                        harvest_fn=None, persist_contact=None, dry_run=True):
    """Process one confirmation email's HTML. Returns a list of per-shipment
    result dicts. In dry-run, no drafts/GHL writes/DB writes happen."""
    parsed = parse_cns_confirmation(html)
    results = []
    for s in parsed["shipments"]:
        if shipment_exists(cx, s["tracking"]):
            results.append({"tracking": s["tracking"],
                            "recipient": s["recipient_name"],
                            "action": "skipped (already processed)"})
            continue

        match = find_contact(s["recipient_name"])
        conf = match["confidence"] if match else "none"
        to = match["email"] if (match and conf in ("high", "medium")) else None
        ghl_contact_id = (match or {}).get("contact_id")

        # No confident GHL match: try a precision-safe harvest from order emails.
        harvested = None
        if to is None and harvest_fn is not None:
            harvested = harvest_fn(s["recipient_name"])
            if harvested and harvested.get("email"):
                to = harvested["email"]
                conf = "harvested"

        status = "drafted" if to else "needs_review"
        email = build_tracking_email(s["tracking"], s["recipient_name"], resolved_email=to)
```

Then replace the draft/record/append tail (old lines 121–137) with:

```python
        draft_id = None
        onboarded = False
        if dry_run:
            action = "would draft (harvested)" if conf == "harvested" else "would draft"
        else:
            if harvested and conf == "harvested" and persist_contact is not None:
                pc = persist_contact(harvested, s["recipient_name"]) or {}
                ghl_contact_id = pc.get("contact_id") or ghl_contact_id
                onboarded = bool(pc.get("onboarded"))
            draft_id = draft_fn(to=to, subject=email["subject"],
                                html=email["html"], text=email["text"])
            record_shipment(
                cx, tracking_number=s["tracking"], order_uuid=parsed["order_uuid"],
                recipient_name=s["recipient_name"], address_block=s["address_block"],
                resolved_email=to, match_confidence=conf,
                ghl_contact_id=ghl_contact_id,
                draft_id=draft_id, status=status, source_msg_id=msg_id)
            action = "drafted"

        results.append({"tracking": s["tracking"], "recipient": s["recipient_name"],
                        "to": to or "(blank — needs review)", "confidence": conf,
                        "status": status, "action": action, "draft_id": draft_id,
                        "onboarded": onboarded})
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-11a8b606 && python -m pytest tests/test_tracking_watcher.py tests/test_tracking.py -q`
Expected: PASS (new 4 + existing tracking tests still green).

- [ ] **Step 5: Add the impure closures + wire `main`**

Add near the Gmail plumbing in `cns_tracking_watcher.py`:

```python
def make_gmail_search_fn(service):
    """Return search(query) -> [{'sender','body'}] over the connected mailbox."""
    import base64
    def _plain(payload):
        if payload.get("mimeType", "").startswith("text/"):
            data = payload.get("body", {}).get("data")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", "replace")
        for p in payload.get("parts", []) or []:
            t = _plain(p)
            if t:
                return t
        return ""
    def search(query):
        q = query  # caller passes the ship-to name; harvest widens as needed
        listing = service.users().messages().list(
            userId="me", q=q, maxResults=10).execute()
        out = []
        for m in listing.get("messages", []):
            full = service.users().messages().get(
                userId="me", id=m["id"], format="full").execute()
            headers = {h["name"].lower(): h["value"]
                       for h in full.get("payload", {}).get("headers", [])}
            out.append({"sender": headers.get("from", ""),
                        "body": _plain(full.get("payload", {}))})
        return out
    return search


def make_harvest_fn(gmail_search):
    from dashboard.order_harvest import harvest_buyer
    return lambda name: harvest_buyer(gmail_search, name)


def make_persist_contact():
    """Upsert the harvested buyer into GHL; onboard ONLY genuine new storefront
    buyers (source 'neworder' AND newly created). Returns {contact_id, onboarded}."""
    from dashboard.ghl import (ghl_upsert_contact, ghl_add_to_pipeline,
                               ghl_enroll_workflow)
    def persist(identity, ship_to_name):
        tag = ("source:gk-purchase" if identity.get("source") == "neworder"
               else "source:phone-email-order")
        cid, created, err = ghl_upsert_contact(
            identity["email"], first_name=identity.get("first") or "",
            last_name=identity.get("last") or "", phone=identity.get("phone") or "",
            source_tag=tag, extra_tags=["tracking-harvest"])
        onboarded = False
        if not err and cid and identity.get("source") == "neworder" and created:
            ghl_add_to_pipeline(cid, ship_to_name, identity["email"])
            ghl_enroll_workflow(cid)
            onboarded = True
        return {"contact_id": cid, "onboarded": onboarded}
    return persist
```

Then in `main()`, build the seams and pass them through. Replace the
`find_contact_by_name` import block + the `handle_confirmation(...)` call:

```python
    try:
        from dashboard.ghl import find_contact_by_name
    except Exception:
        find_contact_by_name = lambda name: None  # noqa: E731

    svc = gmail_service()
    draft_fn = make_draft_fn(svc) if not dry_run else (lambda **k: None)
    harvest_fn = make_harvest_fn(make_gmail_search_fn(svc))
    persist_contact = None if dry_run else make_persist_contact()
```

and the call inside the loop:

```python
            for r in handle_confirmation(html, mid, cx, find_contact_by_name,
                                         draft_fn, harvest_fn=harvest_fn,
                                         persist_contact=persist_contact,
                                         dry_run=dry_run):
```

- [ ] **Step 6: Run the full tracking suite**

Run: `cd /tmp/wt-deploy-chat-11a8b606 && python -m pytest tests/test_tracking.py tests/test_tracking_watcher.py tests/test_order_harvest.py -q`
Expected: PASS (all green).

- [ ] **Step 7: Commit**

```bash
cd /tmp/wt-deploy-chat-11a8b606
git add cns_tracking_watcher.py tests/test_tracking_watcher.py
git commit -m "feat: harvest resolver wired into CNS watcher (fill To + GHL add/onboard)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Live dry-run sanity check + PR

**Files:** none (validation)

- [ ] **Step 1: Dry-run against the real mailbox (read-only, no writes)**

Run: `cd /tmp/wt-deploy-chat-11a8b606 && doppler run -p remedy-match -c prd -- python3 cns_tracking_watcher.py --days 45`
Expected: prints per-shipment lines; previously-blank recipients now show either
`-> <email> (would draft (harvested))` for precision hits or stay
`(blank — needs review)`. Confirm no obviously-wrong email is attached to a
ship-to≠buyer case (e.g. a Kate/Jeff style shipment stays blank).

- [ ] **Step 2: Push branch + open PR**

```bash
cd /tmp/wt-deploy-chat-11a8b606 && git push -u origin sess/11a8b606
gh pr create --title "CNS tracking harvest resolver (close blank-To gap)" \
  --body "Precision-gated harvest of the buyer from order emails when the ship-to name doesn't resolve in GHL: adds the contact (onboards only genuine new storefront buyers), fills To only on a single-email match. No-regression: unmatched/ambiguous cases stay needs_review. Spec + plan in docs/superpowers/. 🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

## Self-Review

- **Spec coverage:** parsers (T1) ✓; precision gate (T2) ✓; handle_confirmation harvest + onboarding rule + dry-run (T3) ✓; watcher wiring (T3 step 5) ✓; no-regression `harvest_fn=None` (T3 test) ✓; live dry-run + PR (T4) ✓. Ship-to≠buyer safety is covered by the name-mismatch → None test (T2).
- **Placeholders:** none — every code/test step is concrete.
- **Type consistency:** identity dict shape `{email,first,last,phone,source,products}` is produced by `harvest_buyer` (T2) and consumed identically by `handle_confirmation`/`persist` (T3). `persist_contact` returns `{contact_id,onboarded}` in both its definition and its consumer. `ghl_upsert_contact` call matches the verbatim signature in Global Constraints.
- **Note for implementer:** the synthetic New-order/Authorize.net fixtures encode the documented format; before relying on live auto-fill, confirm the real emails match (the T4 dry-run is the check). If a real format differs, adjust the Task-1 regex — the precision gate makes a parse miss fail safe (stays needs_review), never wrong.
