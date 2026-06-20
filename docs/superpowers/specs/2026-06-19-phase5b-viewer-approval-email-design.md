# Phase 5b — Triggering-viewer capture + "now Dr. Glen-reviewed" approval email

**Date:** 2026-06-19
**Status:** Approved (design — locked earlier this session); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`)
**Parent:** Phase 5 (console sales-page review, merged PR #177). Closes the in-funnel sales-pages arc.

---

## Problem

Phase 5 lets Glen/Rae approve a product page's AI draft copy (dropping the "pending review" banner). Phase 5b closes the loop on the people who looked at it while it was still a draft: capture every known-email viewer of a not-yet-approved page, and when Glen approves it, email each of them once as Dr. Glen that the page they looked at is now personally reviewed.

## Scope (Phase 5b)

A `sales_page_viewers` store; capture in the product page-data path (known-email viewers of a draft page); and an email step inside the existing `sales_pages.approve` action that sends each un-emailed viewer a "now reviewed" note as Dr. Glen, once. Ships **live, no flag**.

**Out of scope:** anonymous viewers (no address to mail); a viewer-facing dashboard; re-emailing on a later edit/re-approval (each viewer is emailed at most once, ever, per page).

---

## Confirmed decisions (Glen, 2026-06-19)

- **Recipients = EVERY known-email viewer of the draft** (not just the first), each emailed **once** on approval; re-approval emails nobody new.
- **Ship LIVE — no flag** (Phase 5 console is already live; this is the intended behavior).
- **Greeting = plain `Aloha,`** when there is no name; otherwise `Aloha {name},`.
- Capture only happens while the page is **not yet approved** and only for viewers whose email we know (authenticated / consented). Anonymous viewers are skipped.

---

## Architecture

### Store — `dashboard/sales_page_viewers.py` (SQLite `chat_log.db`)
- `sales_page_viewers(product_slug TEXT, email TEXT, name TEXT, first_seen_at TEXT, emailed_at TEXT DEFAULT '', PRIMARY KEY(product_slug, email))`.
- Functions:
  - `init_table(cx)`
  - `record_viewer(cx, slug, email, name="")` — `INSERT OR IGNORE` (a repeat view never resets `emailed_at` or overwrites the first capture).
  - `viewers_to_email(cx, slug) -> list[dict]` — rows for this slug with `emailed_at` empty (`{email, name}`).
  - `mark_emailed(cx, slug, emails)` — stamp `emailed_at=now` for the given emails of this slug.
  - `notify_on_approve(cx, slug, product_name, base_url, *, send, strip=lambda s: s) -> int` — for each `viewers_to_email`, send the "now reviewed" email via the injected `send` (default the caller passes `inbox.send_email`), then `mark_emailed`; per-recipient try/except (a send failure is logged, that recipient is NOT marked, others proceed); returns the count emailed.

### Capture (in `begin_product_page_data`)
Inside the existing `_SALES_AI_COPY_ENABLED` block (the same place `ai_state` is computed), after `_ai_state` is known: when `_ai_state != "approved"` and `get_authenticated_user(request)` yields an email, `sales_page_viewers.record_viewer(cx, slug, email, name)`. Best-effort try/except so a capture error never breaks page-data. (Anonymous viewers have no email and are skipped. A page already approved records nobody — there's nothing to notify.)

### Email on approve (extend `sales_pages.approve`)
- App startup adds `base_url=PUBLIC_BASE_URL` to the existing `sales_pages_actions.configure(...)` deps.
- `_exec_approve` (in `dashboard/sales_pages_actions.py`), after `set_state(..., "approved")`, calls — wrapped in its own try/except so it **never fails the approve** — `sales_page_viewers.notify_on_approve(cx, slug, product_name, base_url, send=inbox.send_email, strip=strip_dash)`, where `product_name = (get_product(slug) or {}).get("name", slug)` and `base_url = _DEPS.get("base_url", "")`.
- **Email content** (Glen voice, no AI-pleasantry filler, no em dashes):
  - subject: `f"Your {product_name} page is ready, reviewed by Dr. Glen"`
  - body: opens `Aloha {name},` (or `Aloha,` when no name), one line that the page they looked at has now been personally reviewed and is ready, the link `{base_url}/begin/product/{slug}`, and closes `In wellness,\nDr. Glen & Rae`. Run the body through `strip` (no em dashes).
  - sent via `inbox.send_email(email, subject, body, from_name="Dr. Glen Swartwout")`.
- **Idempotent:** `emailed_at` is stamped on send, and `viewers_to_email` only returns un-emailed rows, so a re-approval (or a second approve event) emails nobody already notified.

### No flag
Phase 5b ships live (Glen's decision). Capture only runs under `_SALES_AI_COPY_ENABLED` (already live in prod) and only for not-yet-approved pages with a known email; the email only fires on an OWNER/OPS approve action. There is no public-facing surface and no new env var.

---

## Data flow
1. A known-email viewer opens a product page whose copy is still a draft → `record_viewer` (INSERT OR IGNORE).
2. Glen approves the page in the console → `sales_pages.approve` sets `state=approved` and then emails every un-emailed viewer once as Dr. Glen, stamping `emailed_at`.
3. A later viewer of the now-approved page is not captured (nothing to notify); a re-approval emails nobody new.

## Error handling
- Capture wrapped so a viewer-store error never breaks page-data.
- The notify step is wrapped so a failure never fails the approve action (the state change has already committed).
- Per-recipient try/except in `notify_on_approve`: a failed send is logged and that viewer is left un-emailed (retried on a future approve); other recipients still go out.
- `record_viewer` is `INSERT OR IGNORE` (idempotent); `viewers_to_email` + `emailed_at` make the email at-most-once per viewer per page.

## Testing
- **Store:** `record_viewer` idempotent (repeat view doesn't reset); `viewers_to_email` returns only un-emailed; `mark_emailed` stamps; a second `viewers_to_email` after marking excludes them.
- **`notify_on_approve` (fake `send`):** emails every un-emailed viewer once, marks them, returns the count; a re-run sends nobody; a recipient whose `send` raises is left un-emailed while others succeed; the body has no em dash (assert via a dash-injecting `strip`); greeting is `Aloha,` with no name and `Aloha {name},` with one; subject + link correct.
- **Capture (page-data, Flask test client):** an authenticated/known-email viewer of a draft page records a viewer; an approved page records nobody; flag-off (`SALES_PAGES_AI_COPY` off) records nobody.
- **Approve integration:** `sales_pages.approve` on a page with un-emailed viewers sends each once (assert via a mocked send) and stamps them; the approve still succeeds even if the send raises.
- Follow deploy-chat test isolation (tmp `$DATA_DIR/chat_log.db`; mock Supabase; importorskip playwright; `importlib.reload`). NO emoji; no em dashes in the email.

## Notes
- Reuses `dashboard/inbox.send_email` (as Dr. Glen), the Phase-5 `_exec_approve` + `configure` deps, `get_authenticated_user`, `PUBLIC_BASE_URL`, and the page-data `ai_state`. No new external dependency, no new flag.
- Sending real email on approve is an outward-facing effect: once this deploys, the next approve of a page with prior draft-viewers emails them. That is the intended behavior (Glen chose live/no-flag).
