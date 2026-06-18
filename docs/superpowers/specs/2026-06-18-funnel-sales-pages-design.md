# New-Format In-Funnel Sales Pages — Design

**Date:** 2026-06-18
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask app, deployed to Render as `glen-knowledge-chat`)
**Surface:** the `/begin/match` funnel at illtowell.com/begin/match

---

## Problem

When the `/begin/match` funnel identifies a remedy, it currently links out to the old
GrooveKart sales page on `remedymatch.com` (`_resolve_remedy_url` in `app.py`, fallback
`_store_search_url`). That breaks the funnel flow, loses session/consent/points/pricing,
ships an off-brand page, and routes buyers through a GrooveKart checkout that has been
unreliable.

We want **new-format sales pages served from inside the funnel**, rendered from our own
data, with a single global format we can evolve programmatically so every page updates at
once.

## Goals

1. **Brand/visual format** — pages match the illtowell.com funnel look and Glen's copy
   conventions (no emojis; SVG/text icons; "Order" not "Reorder").
2. **Keep buyers in the funnel** — CTA goes to the existing in-funnel `/begin/buy/<slug>`
   checkout (Stripe card auto-confirm + Zelle/Wise claim), retiring the GrooveKart path for
   our own products.
3. **Richer persuasion copy** — real sales-page structure (mechanism, proof, ingredients,
   comparison, research), generated via Glen's skills.
4. **Auto-generated per product** — content generated programmatically from our own data,
   not hand-built per SKU.
5. **Globally evolvable format** — one shared template; change it once → all pages
   re-render. Per-product content editable in the console; format/template evolved in code.

## Non-Goals (this spec)

- The **reviews + referral-coupon engine** (reorder → rate → review → approve →
  shareable tracked referral coupon). That is **Spec 2**, built next. This spec only
  defines the read-only interface the sales page uses to *display* approved reviews/videos.
- **External / off-catalog remedies** — unchanged. We do not generate pages for products
  we do not make; they keep today's trusted-link / search behavior.
- **Naming competitors** — never. Comparison archetypes are always anonymized.
- **Producing the bottle-science video** — async content task (slides → video); the page
  has the slot ready.

---

## Architecture

### Canonical destination

New internal route **`/begin/product/<slug>`** becomes the canonical sales page. The funnel
match card links here instead of `remedymatch.com` for in-catalog products. The page CTA
("Order →") points at the existing **`/begin/buy/<slug>`** in-funnel checkout.

- `_resolve_remedy_url` / match-event construction (`app.py` ~L2048) gains a
  `product_url = /begin/product/<slug>` for in-catalog slugs (resolved via the existing
  `_resolve_buy_slug`). The match card prefers `product_url` when present.
- Off-catalog / external remedies: unchanged (trusted link or `_store_search_url`).

### Render-from-data (two layers)

- **Template / format** — lives in code (Jinja/HTML + a display-component layer). Evolved
  by Claude. A template change is instantly global because pages render from data.
- **Per-product content** — a structured record per slug, stored in **SQLite (`chat_log.db`
  under `DATA_DIR`)**, not a flat file — so the state machine is queryable and the approve
  action + content persist in the same DB/transaction as the event log. Tables:
  - `sales_pages(product_slug PK, state, content_json, triggering_email, created_at,
    updated_at, approved_at, approved_by)`
  - `sales_page_viewers(person_id, product_slug, sections_viewed, image_ratings_json,
    credit_earned, …, UNIQUE(person_id, product_slug))`
  Editable in the console by Glen/Rae. (Follows the `client_portals(content_json)` pattern in
  `dashboard/client_portal.py`.)

### Content lifecycle (state machine)

```
none → generating → draft → approved
```

- **none** — no page record yet.
- **generating** — triggered on-demand when the matcher surfaces an in-catalog remedy with
  no page. The page reveals progressively (real streaming where content is being generated,
  simulated reveal where cached) behind the AI-generated caveat banner. Generation is
  enqueued using the queue + local-watcher pattern already proven in scan-notify.
- **draft** — AI content is live but wears the **caveat banner**; comparison table uses the
  safe generic archetype (#2, see below).
- **approved** — Glen approves in the console → banner drops, reviewed anonymized
  comparators (#1) swap in, and **the viewer who triggered generation gets an email** that
  the page is now Dr. Glen–reviewed.

### Generation (where it actually runs)

**Important:** the Render app **cannot invoke Glen's local skills** (formulation-analyzer,
sales-page-architect, formulation-image-studio) in-process — those are tools on Glen's Mac.
Generation is built as code, and split to keep slow work off Render's web workers (a past
cold-start burst wedged gunicorn → site-wide 502; slow image gen must not run in a web
request):

- **Copy — in-process on Render, streamed live.** A generator module builds the section
  prompts in Python from the product's structured data (`data/products.json` +
  `dashboard/product_content`) plus Glen's documented copy conventions, and streams each
  section to the browser using the existing `sse()` + `_cl.messages.stream(...)` pattern
  (`app.py`). This produces the live progressive reveal on first visit and the **safe generic
  comparison archetype (#2)**. (The skill *bodies* inform the prompts; they are not called.)
- **Images — off Render, async via a local watcher.** On first request the job is enqueued
  (reuse the `process_queue.enqueue` table pattern); a launchd watcher on Glen's Mac renders
  the **2** Flux images via the image-studio infra already local there (Replicate token +
  `deploy-chat311` venv), writes them to the media store + draft record, and the page
  backfills them when ready. Mirrors the proven sync-ghl-writes / scan-notify local-drain
  pattern (Render enqueues, the Mac does what Render can't).

**Compliance rails:** comparison rows limited to label-/material-based facts; the
hidden-excipient line stays category-level (never accuses a specific product); a
guardian-style pass injects Glen's verified credentials.

### Per-viewer state

Keyed off the funnel's existing member identity (the name+email soft opt-in already gating
advice/checkout). Stored in the viewer's profile:

- which sections they have opened (restored on return),
- their image 👍/👎 ratings,
- the "rated both images" credit (redeemed on order — see Rewards).

---

## Page section model

One ordered, render-from-data template. **The intro is open by default; every other section
is a collapsed drop-down that renders its content on first click** (real streaming where
generated, simulated reveal where cached). Each viewer's open/closed state is remembered.

| # | Section | Default | Notes |
|---|---------|---------|-------|
| 1 | **Intro — functional description** | **open** | One-paragraph "what this does for you," rendered first. |
| 2 | **Basic description** | collapsed (renders first among drop-downs) | Fuller plain-language overview. |
| 3 | **Video** | collapsed | A **multi-source video list**: product hero video, then **educational videos** (e.g. the bottle-science video), then **approved user-generated videos** about the product. The hero + educational entries are page content; the UGC entries are read from the reviews store (Spec 2) so an approved customer video appears here automatically. |
| 4 | **Ingredients** | collapsed | Label-format panel per Glen's convention (`Compound Class: Common Name (Latin) dose %DV`, elemental/IU rules, no-excipient ethos). |
| 5 | **Comparison table** | collapsed | See below. Strongest differentiator — earns the high spot. |
| 6 | **Research** | collapsed | Citations / links. |
| 7 | **Images + community feedback** | collapsed | The 2 rated images; placed low to avoid friction before the CTA. See below. |
| 8 | **Order CTA** | persistent / prominent | → `/begin/buy/<slug>` in-funnel checkout. |

Across the top of any unreviewed page: the **AI-generated caveat banner** — "Generated from
Dr. Glen's knowledge base — pending his personal review for final approval."

### Comparison table (section 5)

- You vs **two anonymized archetypes**: "a leading professional-channel formula" and "a
  top-selling mass-market formula." (No ranking claims like "#1 on Fullscript"; no named
  products.)
- **Green-check ✓ / red-X ✗ scoring** on: form quality, excipient-free, completeness
  (good ingredients present vs missing), **packaging**, and **microplastic exposure**.
- **Category-level hidden-excipient callout**: "the industry can legally include up to ~3%
  stearates with no label disclosure" — framed as category education, never as an accusation
  about a specific product.
- **Rows include:** mineral/ingredient form, named excipients (from the comparator's own
  published Supplement Facts panel only), dose, completeness, **packaging** (Miron
  biophotonic violet glass ✓ vs plastic/clear glass ✗), **microplastic exposure** (none ✓ vs
  likely ✗).
- **Draft → approved:** draft renders the generic archetype (safe, no review needed);
  approval swaps in the reviewed anonymized-but-representative comparators whose label facts
  Glen confirmed in the console review pass.

### Two image systems (distinct)

**System 1 — rated community-feedback images (section 7).** Exactly **2** images: one
Mode A botanical-lifestyle + one Mode B mechanism. Each has an optional 👍/👎. **Static while
being rated.** Framing: "You help shape Dr. Glen's program — rate these and earn a credit
toward this product." Rate both → **1 credit, redeemable only when the viewer orders that
product** (anti-gaming; self-funding). Ratings + credit stored in the viewer profile.

**System 2 — packaging / bottle-science visuals (brand assets, not rated).** The Miron
biophotonic violet-glass graphics + bottle shots, near the bottle/comparison area.
**Anti-fatigue rotator:** an auto-rotating slider alternating assets on a **≤10-second**
interval; the template may vary count/placement per page so the site does not feel samey.
Honor `prefers-reduced-motion` (no auto-rotate for those users). Rotation applies to these
brand assets **only**, never the rated block. The rotator renders whatever assets exist
(1 → static, 2+ → slider). Because it lives in the shared template, it is evolved in code.
Links to `skepticalreviews.com/bottles` (Glen's own page) as the "learn more about the
science" reference.

---

## Console editor + global template control

- The **Products module** (`dashboard/products.py` + console UI) gains a per-product
  **Sales Page** editor: view/edit the structured content fields, see state, edit the two
  comparators, manage images/videos, attach education videos.
- **Approve** action → state = `approved`, fires the triggering-viewer email, logs an event
  via the existing action/event spine (`dispatch_action`). A **Regenerate** action re-runs
  the pipeline.
- **Template = code, content = console.** Rae/Glen own per-product content in the console;
  Claude owns the global format in code. A template change re-renders all pages with no
  per-page editing — the "update all pages programmatically" capability.

---

## Interface to the future reviews/referral spec (Spec 2)

- The page's **testimonial** section and the **user-generated-video** entries of the video
  list (section 3) read from a **reviews store** (read-only contract here; Spec 2 populates
  it). Empty store → those entries simply don't render; the video list still shows the hero +
  educational videos.
- **Image 👍/👎, section-open state, and the "rate both → 1 credit" reward** are captured
  into the viewer profile now. The credit **redeems through the existing points/pricing
  engine on order**, using the same idempotent order-settlement path as points
  (`settle_order_points`-style), which is what protects against gaming.

---

## Open content items (not spec blockers)

- **Distinct Miron asset set** — Glen sourcing originals (e.g. `miron.com` `visual
  compare.png`, violetglass discovery visuals). **Self-host all assets** (never hotlink
  external CDNs — they rot / change / are someone else's bandwidth). **Replicate** the
  third-party retailer graphic (therealfoodcompany) via the image studio rather than copy
  it. Rotator handles however many exist.
- **Bottle-science video** — slides on Jakob Lorber, Yves Kraushaar, and Drs. Hugo Niggli &
  Max Bracher (source of the research images), turned into a video via Glen's slide →
  render workflow. Drops into the section-3 education-videos slot when ready.

---

## Reused building blocks (grounded in the codebase)

- **Match wiring:** `_resolve_buy_slug` (`app.py` ~L2765) already maps name→slug; add
  `"product_url": (f"/begin/product/{buy_slug}" if buy_slug else "")` to the match event
  (`app.py` ~L2052). `static/begin-match.html` `renderMatch()` (~L271-307) prefers
  `product_url` over `buy_url`/`url`/`search_url`.
- **Route/serve:** new `/begin/product/<slug>` serving `static/begin-product.html`
  (static HTML + client-side XHR), modeled on `begin_buy_page` (`app.py` ~L2778); content
  via an extended `/begin/product-data/<slug>` (already returns name/price/description/
  ingredients/benefits/how_it_works/`open_sections`).
- **Approve / regenerate:** new `dashboard/sales_pages.py` registering actions via the
  `dispatch.dispatch_action` → `actions.action` → `events.append_event` spine; copy the
  `products.mark_page_fixed` action shape (`dashboard/products.py` ~L115); RBAC permission
  `(OWNER, OPS)`, `risk_tier=LOW_WRITE` (AUTO for owner/ops).
- **Console editor:** `static/console-sales-pages.html` modeled on
  `static/console-biofield-portal.html`; add a BOS sub-tab entry in `static/op-nav.js`
  (`{ id: "sales", label: "Sales Pages", href: "/console/sales-pages" }`).
- **Image credit:** `dashboard/points.credit(cx, email, value_cents=100,
  reason="page_rate_images", order_ref=f"page_{slug}")` (idempotent via `has_entry`);
  realize it inside `_settle_order_points` (`app.py` ~L2636) gated on the order's product.
- **Approval email:** `dashboard.inbox.send_email(to, subject, body, from_name=...)`.
- **Viewer identity:** `dashboard/portal_identity` (email→`person_id`), keyed off the
  funnel `amg_session` cookie + `journey_state`/`people`.
- **Streaming + queue:** `sse()` / `stream_with_context` for reveal; `process_queue.enqueue`
  + a `sales_page_watcher.py` launchd worker for image generation.

## Testing

- **Route/resolution:** in-catalog match → card links `/begin/product/<slug>`; off-catalog
  match → unchanged external/search behavior.
- **State machine:** none→generating→draft→approved transitions; draft shows caveat banner
  + generic archetype; approved drops banner + shows reviewed comparators + sends viewer
  email (mock the mailer).
- **Progressive reveal:** intro open by default; other sections render on first click;
  open/closed state persists per viewer across reloads.
- **Image rewards:** rating both images grants exactly one credit; credit only realizes on
  an order for that product; idempotent (no double-grant).
- **Rotator:** 1 asset → static; 2+ → slider; `prefers-reduced-motion` → no auto-rotate;
  never rotates the rated block.
- **Comparison compliance:** no named competitors; excipient callout stays category-level.
- **Reviews interface:** empty reviews store → testimonial/video sections hide cleanly.
- Follow `deploy-chat` test-isolation conventions (mock live Supabase; seed tmp
  `$DATA_DIR/chat_log.db`; `pytest.importorskip` playwright).

---

## Sequencing

1. **Spec 1 (this doc):** new-format in-funnel sales pages, incl. the image-feedback element
   and the read-only reviews interface.
2. **Spec 2 (next):** reviews + referral-coupon engine (reorder → rate → review → approve →
   shareable tracked referral coupon) that *populates* the reviews store this page reads.
