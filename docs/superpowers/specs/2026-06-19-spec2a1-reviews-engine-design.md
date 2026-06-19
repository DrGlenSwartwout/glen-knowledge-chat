# Spec 2a-1 — Product Reviews Engine (collect → AI-score → moderate → display)

**Date:** 2026-06-19
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`)
**Parent:** Spec 2 (reviews + referral-coupon engine). This is the first of three increments: **2a-1 core** (this doc) → 2a-2 (in-browser record + AI video scoring/transcript + auto-trim) → 2a-3 (AI-recommended gift → console-approve → next order). The referral-coupon reward is the separate Spec 2b.

---

## Problem

The in-funnel sales pages have no social proof and no way for buyers to review what they bought. We want verified buyers to rate and review products, earn store-credit points for genuine reviews (AI-scored), and have approved reviews shown on the product page — lifting conversion and feeding a reward loop.

## Scope (2a-1)

Verified-buyer reviews: a required star rating at reorder plus an optional written review and an optional video (by link or upload). Written reviews are **AI-scored for points (fully automated)** and **AI-gated for compliance**. Points credit the existing points ledger immediately on gate-pass. A console queue lets Glen/Rae gate **publication** (with an AI publish recommendation). Approved reviews render in a new "What people are saying" sales-page section; approved UGC videos feed the existing video section.

**Out of scope (later increments):** in-browser video recording, AI video scoring (transcript-based) and auto-trim → **2a-2**; AI-recommended gift → console-approve → next-order → **2a-3**; the referral-coupon reward → **2b**.

---

## Confirmed decisions (Glen, 2026-06-19)

- Video input methods: **link or upload** now; **record** in 2a-2.
- Reward = **points**, **fully AI-automated** (never human-set), **1 point = $1 store credit (100¢)**, **capped at 5 points total** per review. Written scores **0–2**; a strong **video alone can reach 5** (video scoring lands in 2a-2 with transcript extraction). Points auto-credit on the AI compliance-gate pass.
- **Humans gate only publication and gifts**, each with an **AI recommendation**. Points never wait on a human.
- **Star rating is required to reorder** a product (unless already reviewed).
- **Idempotent per item**: one review per (buyer, product); capture never re-prompts once done.
- Entry points: the **reorder page** and a **post-purchase email** link.
- Verified buyers only (must have a paid order containing that product).

---

## Architecture

### Data — `dashboard/product_reviews.py` (SQLite `chat_log.db`)
`product_reviews(id INTEGER PK, product_slug TEXT, email TEXT, name TEXT, rating INTEGER NOT NULL, body TEXT DEFAULT '', video_kind TEXT DEFAULT '', video_ref TEXT DEFAULT '', ai_score INTEGER DEFAULT 0, ai_verdict TEXT DEFAULT '', ai_recommend_publish INTEGER DEFAULT 0, points_awarded INTEGER DEFAULT 0, status TEXT DEFAULT 'pending', featured INTEGER DEFAULT 0, created_at TEXT, reviewed_at TEXT, reviewed_by TEXT, UNIQUE(product_slug, email))`.
- `status` ∈ {pending, approved, rejected}. `UNIQUE(product_slug,email)` is the **done-state**.
- `video_kind` ∈ {'', link, upload}; `video_ref` = URL (link) or stored filename (upload).
- Functions: `init_table`, `has_reviewed(cx, slug, email) -> bool`, `upsert_review(cx, slug, email, name, rating, body, video_kind, video_ref) -> id`, `set_ai_result(cx, id, ai_score, ai_verdict, recommend_publish)`, `set_points(cx, id, points)`, `set_status(cx, id, status, by="")`, `set_featured(cx, id, on)`, `approved_for_slug(cx, slug) -> [rows]`, `aggregate(cx, slug) -> {count, avg}`, `pending_queue(cx) -> [rows]`.

### AI scoring + compliance gate — `dashboard/review_scoring.py`
One Claude-haiku call (model `claude-haiku-4-5-20251001`, like Phase 2/5). Input: product name + the written review text. Returns a validated object: `{compliance_ok: bool, reasons: str, quality_points: int(0-2), recommend_publish: bool}`. Quality rewards specificity/authenticity/usefulness, **not** length or keyword stuffing. The gate **rejects** disease-cure/medical claims, PII, spam, and abusive content. All text run through `_strip_dash`. A pure prompt-builder `build_review_prompt(product, body)` keeps the call testable with a fake client. (Video scoring is **not** called in 2a-1.)

### Points — reuse `dashboard/points.py`
On AI gate **pass**, credit `min(5, quality_points)` dollars (× 100 = cents) to the buyer's points ledger, `source=f"review:{slug}:{review_id}"`, **idempotent** via `has_entry`. Gate **fail** → no points, `status` stays `pending` with the AI reasons recorded for the console. Points are independent of publication.

### Capture
- **Reorder gate:** the reorder page (`/reorder`, `api_reorder_items`) requires a **star rating** for each line item not yet reviewed (`has_reviewed` false) before that item can be reordered. Written + video are optional. Submits to `POST /api/reviews` (verified by `list_orders_by_email` having a paid order with the slug).
- **Post-purchase email:** a tokenized review link (reuse the existing `/invoice/<token>`-style token pattern) opening the same review form, pre-bound to (email, slug).
- **Upload storage:** video uploads saved to `DATA_DIR/review-media/<slug>/<file>`, served via a gated route; size/type guarded. Links stored as URLs. No AI/ffmpeg processing in 2a-1.

### Moderation — `dashboard/reviews_actions.py` on the dispatch spine
Actions (RBAC `(OWNER, OPS)`, `LOW_WRITE`), driven via the existing `/api/action/<key>`:
- `reviews.approve {id}` → `status=approved` (publishes). `reviews.reject {id}` → `status=rejected`. `reviews.feature {id, on}` → pin/unpin.
New console page `static/console-reviews.html` (modeled on `console-sales-pages.html`) + a BOS sub-tab in `op-nav.js`; lists the pending queue with the review text, rating, video link/preview, the **AI publish recommendation + reasons**, and the points already awarded. Read API `GET /api/console/reviews` (pending + recent), gated by `_sales_console_ok()` (same console-secret pattern as Phase 5).

### Display — sales page
- New accordion section **"What people are saying"** (after `research`, before `images`) in `begin_product_page_data` + `static/begin-product.html`: **aggregate avg stars + count**, then approved written reviews (first name or initial + **"Verified buyer"** badge), with an **"Individual results vary"** disclaimer. Only `status=approved` rows surface; featured pinned first.
- Approved **UGC videos** (approved review with a video) feed the **existing video** section as `kind:"ugc"` entries.

---

## Data flow
1. Buyer hits the reorder page (or post-purchase email link) → must star-rate the product → optionally writes a review / attaches a video → `POST /api/reviews`.
2. Server verifies the buyer, upserts the review (`pending`), runs `review_scoring` on the written text.
3. Gate pass → auto-credit points (capped 5) + store the AI publish recommendation; gate fail → no points, flagged for the console.
4. Glen/Rae open `/console/reviews`, see the AI recommendation, and `reviews.approve` / `reject` / `feature`.
5. Approved reviews render in the page's testimonials section (and approved videos in the video section).

## Error handling
- AI-scoring failure (Claude error) → review saved `pending`, no points, logged; never blocks the submit.
- Points credit wrapped so a ledger error never fails the review submit (best-effort, like Phase 4 credit).
- Upload guarded by size/extension; bad input → 400, no row.
- Page-data testimonials block wrapped so a read error never breaks page-data (degrade to no section).
- Re-submitting an existing (slug,email) updates the row (edit), re-runs scoring, but **never double-credits** (idempotent source key).

## Testing
- **Data layer:** `has_reviewed` done-state; `upsert_review` UNIQUE upsert; status/feature transitions; `aggregate` avg+count; `approved_for_slug` excludes pending/rejected.
- **AI scoring (mock client):** quality→points(0-2); compliance gate rejects a disease-claim sample (`compliance_ok=false`); em dashes stripped.
- **Points:** auto-credit on pass, capped at 5, idempotent (no double-credit on re-submit); no credit on gate-fail.
- **Capture:** verified-buyer gate (non-buyer rejected); star required to reorder an unreviewed item; no re-prompt once reviewed; upload size/type guard.
- **Moderation actions:** approve→approved, reject→rejected, feature toggles; RBAC denies non-OWNER/OPS.
- **Display/page-data:** only approved reviews + aggregates surface; featured first; disclaimer present; flag-off/no-reviews → no section.
- Follow deploy-chat test isolation (tmp `$DATA_DIR/chat_log.db`; mock Supabase; importorskip playwright). Console UI = manual visual pass. NO emoji.

## Flag
Ship behind **`REVIEWS_ENABLED`** (default off): capture endpoints 404, no testimonials section, console hidden until on. Lets it merge dark and go live when the first reviews are ready to moderate.

## Notes
- Compliance: customer testimonials carry FTC/FDA risk; the AI gate rejects disease-cure claims and the section carries an "individual results vary" disclaimer. Human approval is the final publication gate.
- Voice/UI conventions: NO emoji (SVG/text glyphs), "Order" not "Reorder" where customer-facing, no em dashes in generated text.
