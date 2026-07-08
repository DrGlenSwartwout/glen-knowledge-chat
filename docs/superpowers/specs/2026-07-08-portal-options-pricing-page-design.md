# Client-portal "Options & Pricing" card — design (teed up 2026-07-08)

**Status:** spec ready to build. Origin: Dana T asked for "one page with a list of options, links and pricing explaining each option," and her reply promises it's "being built." Every client should see the same simple orientation.

## Goal

A single, always-clear card on every client portal that answers "what are my options and what do they cost?" — so a client never has to email to understand the offering. Same content for everyone (with the client's own courtesy fee if one is set).

## The content (confirmed offers, 2026-07-08 — verify flags still true at build)

The whole live picture is three items (this session confirmed the `$1` biofield trial and `$99/mo` continuous-care are BOTH off — do NOT list them):

1. **Your Biofield Analysis & Remedies** — your voice-scan reading and personally matched remedies, in your private portal. Included with your scan. Order remedies as you wish (Functional Formulations ~**$69.97** each, 30-day supply).
2. **Personal Causal Biofield Analysis + Program + Consultation with Dr. Glen** (the hands-on / muscle-testing depth) — one-time **$300** (a $1,000 value). "Reply to arrange."
3. *(No monthly subscription. Nothing to sign up for.)*

## Data source — do NOT hardcode prices (avoid drift)

- The **$300** must come from the canonical `biofield-analysis` service product in `data/products.json` (`price_cents`), overridden by the client's courtesy price when set (`client_prices` by email+slug — same source the fee panel + invoice pricer already use). See [[reference_biofield_analysis_invoice]].
- The **FF remedy price** from `_FF_BASE_CENTS` (69.97) — or phrase it as "from ~$70" to stay robust.
- Rationale: prices changed twice this session; a hardcoded page would silently go stale. [[feedback_confirm_channel_live_before_fixing]]

## Placement & mechanism (mirror the invoice card just shipped, #701)

- **Payload**: add `payload["options"]` (or `["pricing"]`) in `api_client_portal` (app.py ~15194) — a small best-effort block: `{biofield_analysis_cents, biofield_analysis_value_cents, ff_from_cents}` resolved from the catalog + this client's `client_prices`. Best-effort, never breaks the load (same pattern as `payload["invoices"]`).
- **Render**: a new `<div class="card options-card">` in `static/client-portal.html` (near the report/invoice cards) that lists the three items with the resolved prices and a "reply to arrange" affordance for #2.
- **Gate**: a flag (e.g. `PORTAL_OPTIONS_CARD_ENABLED`) so it can ship dark then flip on, like the other portal features.
- Memory gotcha: a field only reaches the page if it's in BOTH the payload dict AND rendered ([[feedback_portal_content_payload_surface]]).

## Open decisions (resolve at build)

1. **Static copy vs live prices** — recommend LIVE prices (data-sourced) for the two dollar figures, static copy for the rest. Confirm with Glen.
2. **Show the client's courtesy price** if they have one (e.g. Karin's $100)? Recommend yes — it personalizes and matches the fee panel. Confirm.
3. **Also surface it on the Intake `/author/<id>/invoice-view` page** for the operator's reference? Optional; low cost since the tab bar exists.
4. **CTA for #2** — "Reply to this email/portal" vs a real book-a-consult link. Recommend "reply" for v1 (the $300 tier is `cta_label: "Book your consultation"` in `begin_funnel.TIER_CATALOG`).

## Testing

- Payload includes `options` with catalog-sourced cents; a client with a `client_prices` courtesy override shows that override.
- Card renders the three items; no dollar figure is hardcoded in the HTML.
- Flag off → no card; on → card present.

## Out of scope (v1)

- Per-client personalization beyond the courtesy price.
- A separate standalone public pricing page (this is the in-portal card).
- Listing the higher ascension rungs (certification/1:1/etc.) — this is the client-facing "what do I do next" trio, not the B2B ladder.
