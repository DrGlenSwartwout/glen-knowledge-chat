# Studio-credit Self-Serve Claim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewire the existing live `/coaching/studio-credit` POST so a buyer's self-serve submission creates a real `studio_credit_claims` row (`source='self_serve'`, pending) that lands in `/console/studio-credits` for one-click approve→grant, instead of writing a dead legacy table and emailing a manual-grant instruction.

**Architecture:** Add a dedupe upsert to the existing Flask-free store (`dashboard/studio_credit.py`); point the route at it; drop the legacy `studio_credit_intents` insert; reword the internal heads-up email to point at the console (sent only on a genuinely new claim); show the claim `source` in the console. No new public page, no new flag.

**Tech Stack:** Python 3.11, Flask, sqlite3, pytest; existing studio-credit store + console from PR #205.

## Global Constraints

- **Python:** use `python3` (not `python`); never `import app` in tests (Pinecone at import crashes the sandbox) — test `dashboard/*` helpers standalone; verify `app.py` with `python3 -m py_compile`.
- **Timestamps:** `datetime.utcnow().isoformat() + "Z"` (matches existing rows; the store's `_now()` already does this).
- **Dedupe is pending + self_serve only:** a re-submit while a `pending` `self_serve` claim exists updates it in place; an approved/rejected email gets a fresh pending claim. Never dedupe against approved/rejected rows.
- **Stop writing `studio_credit_intents`:** remove the INSERT only; leave its `CREATE TABLE` and historical rows untouched. Nothing reads that table.
- **No new flag.** `/coaching/studio-credit` is already live; this is a backend upgrade.
- **Heads-up email** goes to `RM_INBOUND_INQUIRY_EMAIL` (the internal `+rm-inquiry` sink), `reply_to=None`, best-effort (never 500s), and ONLY when the claim is new (`is_new=True`) so dedupe re-submits don't re-spam.
- **Worktree:** all edits/commits in `/tmp/wt-deploy-chat-selfserve` (branch `sess/selfserve-studio`).

---

### Task 1: Store — dedupe upsert (`dashboard/studio_credit.py`)

**Files:**
- Modify: `dashboard/studio_credit.py` (append one function)
- Test: `tests/test_studio_credit.py` (append cases)

**Interfaces:**
- Consumes: existing `migrate`, `add_claim`, `get`, `list_claims`, `reject_claim`, `_now` from this module (PR #205).
- Produces: `upsert_self_serve_claim(cx, *, email, invoice_ref="", proof_note="") -> (claim: dict, is_new: bool)`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_studio_credit.py`)**

```python
def test_upsert_self_serve_creates_pending(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    claim, is_new = m.upsert_self_serve_claim(cx, email="Buyer@X.com", invoice_ref="order 9")
    assert is_new is True
    assert claim["status"] == "pending" and claim["source"] == "self_serve"
    assert claim["email"] == "buyer@x.com" and claim["invoice_ref"] == "order 9"
    assert len(m.list_claims(cx, status="pending")) == 1


def test_upsert_self_serve_dedupes_pending(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    c1, n1 = m.upsert_self_serve_claim(cx, email="a@x.com", invoice_ref="first")
    c2, n2 = m.upsert_self_serve_claim(cx, email="a@x.com", invoice_ref="second")
    assert n1 is True and n2 is False
    assert c2["id"] == c1["id"]                     # same row updated
    assert c2["invoice_ref"] == "second"           # invoice_ref refreshed
    assert len(m.list_claims(cx, status="pending")) == 1   # no duplicate


def test_upsert_self_serve_after_reject_is_new(tmp_path):
    m = _mod(); cx = _cx(tmp_path)
    c1, _ = m.upsert_self_serve_claim(cx, email="a@x.com", invoice_ref="first")
    m.reject_claim(cx, c1["id"], decided_by="glen", reason="no proof")
    c2, n2 = m.upsert_self_serve_claim(cx, email="a@x.com", invoice_ref="retry")
    assert n2 is True and c2["id"] != c1["id"]     # fresh pending claim
    assert m.get(cx, c1["id"])["status"] == "rejected"
    assert len(m.list_claims(cx, status="pending")) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-selfserve && python3 -m pytest tests/test_studio_credit.py -k self_serve -v`
Expected: 3 FAIL — `module has no attribute 'upsert_self_serve_claim'`.

- [ ] **Step 3: Write the implementation (append to `dashboard/studio_credit.py`)**

```python
def upsert_self_serve_claim(cx, *, email, invoice_ref="", proof_note=""):
    """Public self-serve submission. Dedupe: if a pending self_serve claim already
    exists for this email, update it in place (refresh invoice_ref/proof_note and
    bump created_at); otherwise create one. Returns (claim, is_new). Pending-only:
    an approved/rejected email gets a fresh pending claim."""
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("valid email required")
    cur = cx.cursor()
    cur.row_factory = sqlite3.Row
    existing = cur.execute(
        "SELECT id FROM studio_credit_claims "
        "WHERE email=? AND status='pending' AND source='self_serve' "
        "ORDER BY created_at DESC LIMIT 1",
        (email,)).fetchone()
    if existing is not None:
        cx.execute(
            "UPDATE studio_credit_claims SET invoice_ref=?, proof_note=?, created_at=? "
            "WHERE id=?",
            (invoice_ref or "", proof_note or "", _now(), existing["id"]))
        cx.commit()
        return get(cx, existing["id"]), False
    claim = add_claim(cx, email=email, invoice_ref=invoice_ref, proof_note=proof_note,
                      source="self_serve", created_by="self_serve")
    return claim, True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-selfserve && python3 -m pytest tests/test_studio_credit.py -q`
Expected: PASS (10 passed total — 7 existing + 3 new).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-selfserve
git add dashboard/studio_credit.py tests/test_studio_credit.py
git commit -m "feat(studio-credit): self-serve dedupe upsert in store"
```

---

### Task 2: Route rewire + buyer confirmation copy (app.py, coaching.html)

**Files:**
- Modify: `app.py` — `coaching_studio_credit_post` (currently lines ~19815-19848)
- Modify: `static/coaching.html` — `studio_credit_submitted` block (lines ~196-204)

**Interfaces:**
- Consumes: `dashboard.studio_credit.upsert_self_serve_claim` (Task 1); existing `_db_lock`, `LOG_DB`, `_send_inquiry_email`, `RM_INBOUND_INQUIRY_EMAIL`, `_render_static_template`, `datetime`.
- Produces: no new symbols (route behavior change only).

- [ ] **Step 1: Replace the POST handler body**

Find this exact block in `app.py`:

```python
@app.route("/coaching/studio-credit", methods=["POST"])
def coaching_studio_credit_post():
    import uuid
    data = request.get_json(silent=True) or request.form or {}
    email = (data.get("email") or "").strip().lower()
    studio_ref = (data.get("studio_ref") or "").strip() or None
    if email and "@" in email:
        sid = str(uuid.uuid4())
        now_iso = datetime.utcnow().isoformat() + "Z"
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            cx.execute(
                "INSERT INTO studio_credit_intents (id, created_at, email, studio_ref) "
                "VALUES (?,?,?,?)",
                (sid, now_iso, email, studio_ref)
            )
        subject = "studio.com credit intent submitted"
        body = (
            f"A visitor reported a studio.com purchase and asked for the 30-day credit.\n\n"
            f"Email: {email}\n"
            f"studio_ref: {studio_ref or '(not provided)'}\n"
            f"Submitted: {now_iso}\n\n"
            f"To verify and grant 30 days, POST /admin/membership/grant with "
            f"source=studio_credit, email={email}, notes=studio_ref.\n"
        )
        try:
            _send_inquiry_email(
                to_email=RM_INBOUND_INQUIRY_EMAIL,
                subject=subject, body=body,
                reply_to=None,
            )
        except Exception as e:
            print(f"[studio-credit] glen notification failed: {e!r}", flush=True)
    html = _render_static_template("coaching.html", status="studio_credit_submitted")
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}
```

Replace it with:

```python
@app.route("/coaching/studio-credit", methods=["POST"])
def coaching_studio_credit_post():
    from dashboard import studio_credit as _sc
    data = request.get_json(silent=True) or request.form or {}
    email = (data.get("email") or "").strip().lower()
    studio_ref = (data.get("studio_ref") or "").strip()
    if email and "@" in email:
        with _db_lock, sqlite3.connect(LOG_DB) as cx:
            _sc.migrate(cx)
            claim, is_new = _sc.upsert_self_serve_claim(
                cx, email=email, invoice_ref=studio_ref)
        if is_new:
            subject = "New self-serve studio-credit claim"
            body = (
                f"A visitor reported a studio.com purchase and asked for the free month.\n\n"
                f"Email: {email}\n"
                f"studio_ref: {studio_ref or '(not provided)'}\n"
                f"Submitted: {datetime.utcnow().isoformat() + 'Z'}\n\n"
                f"Review and approve at /console/studio-credits.\n"
            )
            try:
                _send_inquiry_email(
                    to_email=RM_INBOUND_INQUIRY_EMAIL,
                    subject=subject, body=body,
                    reply_to=None,
                )
            except Exception as e:
                print(f"[studio-credit] glen notification failed: {e!r}", flush=True)
    html = _render_static_template("coaching.html", status="studio_credit_submitted")
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}
```

(Removes `import uuid`, the `studio_credit_intents` INSERT, and the manual-grant email; sends the reworded console-pointing heads-up only when `is_new`.)

- [ ] **Step 2: Reword the buyer confirmation copy in `static/coaching.html`**

Find:

```html
      <h2>Got it.</h2>
      <p>Glen will verify your studio.com purchase and email you a sign-in link within about seven days.</p>
```

Replace with:

```html
      <h2>Got it.</h2>
      <p>We'll review your studio.com purchase and email you your free month of coaching access. Keep an eye on your inbox.</p>
```

- [ ] **Step 3: Verify app compiles and the store tests still pass**

Run: `cd /tmp/wt-deploy-chat-selfserve && python3 -m py_compile app.py && echo OK`
Expected: `OK`.
Run: `python3 -m pytest tests/test_studio_credit.py -q`
Expected: 10 passed (Task 1's behavior unchanged).

- [ ] **Step 4: Confirm the legacy INSERT is gone and nothing else writes it**

Run: `cd /tmp/wt-deploy-chat-selfserve && grep -n "INSERT INTO studio_credit_intents" app.py || echo "no insert — good"`
Expected: `no insert — good`. (The `CREATE TABLE IF NOT EXISTS studio_credit_intents` at ~line 6255 stays.)

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-selfserve
git add app.py static/coaching.html
git commit -m "feat(studio-credit): self-serve POST creates a claim; drop legacy intents insert"
```

---

### Task 3: Console — show claim source (`static/console-studio-credits.html`)

**Files:**
- Modify: `static/console-studio-credits.html` — claim render (lines ~46-49)

**Interfaces:**
- Consumes: the `source` field already present on every claim from `GET /api/console/studio-credits` (the API returns full claim rows; `list_claims` includes `source`).
- Produces: nothing new.

- [ ] **Step 1: Add a source pill next to the status pill**

Find:

```javascript
        <strong>${esc(c.email)}</strong>
        <span class="pill">${esc(c.status)}</span>
        ${c.invoice_ref?`<span class="muted">inv: ${esc(c.invoice_ref)}</span>`:""}
```

Replace with:

```javascript
        <strong>${esc(c.email)}</strong>
        <span class="pill">${esc(c.status)}</span>
        ${c.source==="self_serve"?`<span class="pill" style="background:#e7f0ff">self-serve</span>`:""}
        ${c.invoice_ref?`<span class="muted">inv: ${esc(c.invoice_ref)}</span>`:""}
```

- [ ] **Step 2: Confirm the file still parses (no syntax break in the template literal)**

Run: `cd /tmp/wt-deploy-chat-selfserve && node --check static/console-studio-credits.html 2>/dev/null || echo "node check n/a — visually verify the added line is balanced"`
Expected: either a clean node check, or the fallback message (the file is HTML, so `node --check` may not apply — in that case eyeball that the inserted line uses matching backticks and `${...}` like its neighbors).

- [ ] **Step 3: Commit**

```bash
cd /tmp/wt-deploy-chat-selfserve
git add static/console-studio-credits.html
git commit -m "feat(studio-credit): show claim source pill in console"
```

---

## Self-Review

**Spec coverage:**
- Rewire POST → create `studio_credit_claims` (`source='self_serve'`, pending) → Task 2 + Task 1 upsert. ✓
- Trust approval gate + dedupe one pending self_serve claim per email → Task 1 `upsert_self_serve_claim` + tests. ✓
- Free-text proof, `studio_ref → invoice_ref` → Task 2 (passes `invoice_ref=studio_ref`). ✓
- Stop writing legacy `studio_credit_intents`, keep CREATE + rows → Task 2 Steps 1 & 4. ✓
- Reworded internal heads-up to console, only when `is_new` → Task 2 Step 1. ✓
- Reworded buyer thank-you → Task 2 Step 2. ✓
- Console `source` pill → Task 3. ✓
- No new flag / no new public page → nothing added; route already live. ✓
- Not in scope (double opt-in, upload, webhook, convert-to-paid) → absent. ✓

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `upsert_self_serve_claim(cx, *, email, invoice_ref="", proof_note="") -> (claim, is_new)` is defined in Task 1 and consumed identically in Task 2 (`claim, is_new = _sc.upsert_self_serve_claim(cx, email=..., invoice_ref=...)`). `source` field name consistent across store, API, and the Task 3 template. `is_new` boolean gates the email in Task 2.
