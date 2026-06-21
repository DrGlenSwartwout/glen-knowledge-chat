# Studio-credit Self-Serve Claim — Design

**Date:** 2026-06-21
**Status:** Approved (brainstorm) — ready for implementation plan
**Builds on:** the merged studio-credit feature (PR #205) — `dashboard/studio_credit.py`, `studio_credit_claims` table, `/console/studio-credits`.

## Problem / Goal

Studio-coaching-app purchasers should be able to **claim their free month themselves** instead of Glen/Rae logging every claim by hand. A primitive self-serve page already exists and is live — `/coaching/studio-credit` — but its POST writes a legacy `studio_credit_intents` row and emails Glen a "run `/admin/membership/grant` manually" notice. We want that submission to instead create a real `studio_credit_claims` row (`source='self_serve'`, `pending`) that lands in `/console/studio-credits` for one-click approve→grant.

## Decisions (locked during brainstorm)

1. **Trust the manual approval gate, not double opt-in.** The claim lands as `pending`; Glen eyeballs the proof and approves, and the free-month welcome email only sends on approval. Anti-abuse is handled by **deduping to one pending self-serve claim per email** (re-submits update, don't pile up), plus the existing rejection path.
2. **Free-text proof only.** No file upload. The existing `studio_ref` field maps to the claim's `invoice_ref`. If Glen needs the receipt he replies to ask.
3. **Stop writing the legacy `studio_credit_intents` table.** Verified nothing reads it (only `CREATE TABLE` + the one `INSERT`). Drop the INSERT; keep the `CREATE TABLE` and historical rows. `studio_credit_claims` + the console become the single source of truth.
4. **No new public page, no new flag.** `/coaching/studio-credit` is already live; this is strictly a better backend.

## Architecture

### 1. Store — dedupe upsert (`dashboard/studio_credit.py`)

New function (reuses `add_claim`):

```
upsert_self_serve_claim(cx, *, email, invoice_ref="", proof_note="") -> (claim: dict, is_new: bool)
```

- Look up an existing claim for `email` with `status='pending'` AND `source='self_serve'`.
- If found: update its `invoice_ref`, `proof_note`, and `created_at` (bump to now); return `(updated_claim, False)`.
- If not found: `add_claim(cx, email=email, invoice_ref=invoice_ref, proof_note=proof_note, source='self_serve', created_by='self_serve')`; return `(new_claim, True)`.

Pending-only dedupe: an email whose prior claim is approved/rejected gets a fresh pending claim on re-submit (a repeat purchaser next year is a legitimate new claim — the per-year guard in `approve_claim` still governs whether it can be granted).

### 2. Route — rewire `coaching_studio_credit_post` (app.py)

- Read `email` + `studio_ref` from `request.get_json(silent=True) or request.form` (unchanged inputs).
- Validate `email` contains `@` (unchanged guard).
- Inside `_db_lock`: `claim, is_new = studio_credit.upsert_self_serve_claim(cx, email=email, invoice_ref=studio_ref or "")`. (Run `studio_credit.migrate(cx)` first, defensively, as the console list route does.)
- Remove the `INSERT INTO studio_credit_intents` block.
- **Internal heads-up email (best-effort, only when `is_new`):** to `RM_INBOUND_INQUIRY_EMAIL`, reworded — subject `"New self-serve studio-credit claim"`, body names the email + studio_ref and says "Review and approve at /console/studio-credits". Wrapped in try/except; never 500s. (Not re-sent on a dedupe update, so re-submits don't re-spam.)
- Render the existing `coaching.html` `status="studio_credit_submitted"` thank-you (copy reworded to set expectations: "We'll review your purchase and email your free month").

### 3. Console — show `source` (`static/console-studio-credits.html`)

Add a small pill on each claim showing `source` (`self_serve` vs `console`) so Glen can tell at a glance which came from the public form. The list/approve/reject paths already handle self-serve claims unchanged (they flow through `list_claims`/`approve_claim`/`reject_claim`).

### 4. Form copy (`static/coaching.html`)

Minor: reword the `studio_credit_submitted` confirmation block to set expectations ("We'll review your purchase and email you your free month of coaching"). The form fields (`email`, `studio_ref`) are unchanged.

## Testing

Unit tests on `dashboard/studio_credit.py` (temp sqlite, no `import app`):

- `upsert_self_serve_claim` on a new email creates a `pending` `self_serve` claim and returns `is_new=True`.
- A second submit for the same email returns `is_new=False`, updates the existing row's `invoice_ref`, and `list_claims` shows exactly **one** pending claim for that email (no duplicate).
- After that claim is approved (or rejected), a new submit for the same email creates a fresh `pending` claim (`is_new=True`).

Route verified via `python3 -m py_compile app.py` (cannot `import app` — Pinecone at import). The console/template changes are static.

## Explicitly NOT in this phase (YAGNI)

- Double opt-in / email-ownership verification.
- Invoice file upload.
- Automated Studio.com webhook or in-funnel SKU signal (the `source` column remains the seam).
- Convert-to-paid logic.
