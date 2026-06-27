# Portal Concierge — Design Spec

**Date:** 2026-06-27
**Status:** Approved (design), pending implementation plan
**Author:** Glen + Claude

## Context

Per the chat surface model (`project_chat_surface_model`), the concierge experience belongs in the **portal** (the client's post-TOS home), not the funnel. Today the post-purchase concierge is an anonymous funnel flow (`/begin/concierge/chat`) that only knows one just-bought product. The portal (`/portal/<token>`) already authenticates a *known* client by email and holds their **biofield findings + order history** — so a concierge living there can be grounded in the client's real data and serve them *ongoing*, not just in the seconds after one checkout.

This is **sub-project B** of the concierge→portal migration. Decision (Glen): build **B first** — the portal concierge for clients who *already* have portals (biofield clients, who have the richest context). The funnel concierge stays as-is for non-portal buyers in the interim.

**Sub-project A (separate, not this spec):** make portals *universal* — every remedy buyer must agree to TOS at purchase, and TOS agreement mints a portal — so eventually every buyer has a portal home and the funnel concierge can be retired. A is the bigger architectural lift (TOS-at-purchase gate, portal minting, link delivery, no-scan render state). Tracked separately.

## Goal

An **ongoing health concierge** chat *inside* the portal: a known, post-TOS client asks about their scan findings, their remedies & protocol (dosing/timing), reorders, and complements — answers grounded in *their* data — with a one-click path to the existing reorder/checkout.

## Architecture

A new token-authed SSE endpoint **`POST /api/portal/<token>/chat`** that reuses the existing concierge engine (the streaming + RAG retrieval + `_PAIRINGS` complement logic + suggestion-extraction at `app.py:6713-6789`), but swaps its inputs and widens its prompt. A chat panel is added to `static/client-portal.html`.

- **Auth:** the portal **token** in the URL, validated with the same `_portal_record_for(token)` helper the other `/api/portal/<token>/*` endpoints use (`app.py:11326-11344`) → resolves to the client's email/identity. **No separate TOS gate** — possessing a portal *is* the post-TOS authorization (do NOT re-run the funnel's `is_member` gate here).
- **Engine reuse:** lift the concierge's stream loop, RAG embedding+retrieval, pairings, and suggestion-extraction. Do NOT fork a new chat stack.

## Context grounding (the upgrade over the funnel concierge)

The system prompt is assembled from the client's **portal data** instead of one `bought_slug`:
- **Biofield findings / causal-chain layers** — latest scan + history (from `portal_biofield_reports` / the portal content blob). Enables "your scan flagged X at Layer 3 — the remedy is Y."
- **Order history** — the remedies the client actually owns (`portal_view._orders_block` → `list_orders_by_email`), for protocol/dosing/timing/reorder questions.
- **Widened prompt** — an "ongoing health concierge": scan interpretation, remedies & protocol, dosing/timing, reorders, and complement suggestions. (The funnel concierge's narrow "you just bought X, complete your protocol + here's a complement" becomes the broader ongoing role.)

## Buy path (reuse existing commerce)

Concierge suggestions — both complements and "reorder my X" — render as a **card with a button to the existing reorder/checkout**, using the **same link mechanism the portal's `reorder_items` already use**. No new cart, no portal-side invoice/add-to-order. The suggestion-extraction resolves a product → its reorder URL the same way the portal's reorder items are resolved.

## Components (isolated, testable)

- **`dashboard/portal_concierge.py`** (new): pure-ish logic —
  - `build_context(cx, email)` → assembles the grounding facts (biofield findings/layers, owned remedies from orders, scan history) into a structured context object.
  - `system_prompt(context)` → the ongoing-concierge instruction string built from that context.
  - `resolve_suggestion(...)` → maps an extracted complement/reorder product to its reorder URL (reusing the product/reorder resolver the portal uses).
  - Keeps the LLM-call/SSE wiring thin in `app.py`. No Flask import here.
- **`app.py`**: the thin `POST /api/portal/<token>/chat` route — token auth → email → `portal_concierge.build_context` → stream (RAG + Haiku) → emit `suggestion` card → `done`. Mirrors the existing concierge route's SSE shape (`{"token"}`, `{"suggestion"}`, `{"done"}`), minus the `{"gate"}` (no TOS gate).
- **`static/client-portal.html`**: a new **"Ask Dr. Glen" chat section** — inline in the portal page (consistent with its sectioned layout: account / video / audio / PDF / layers / reorder), NOT a floating bubble. Streams via the endpoint; renders the answer + suggestion→reorder cards.

## Data flow

1. Client opens their portal → sees the "Ask Dr. Glen" section.
2. Types a question → `POST /api/portal/<token>/chat` `{query, history}`.
3. Backend: validate token → email; `build_context(cx, email)` (biofield + orders); build prompt; RAG retrieve; stream the answer; extract a suggestion; emit a `suggestion` card carrying the reorder URL.
4. Frontend: render the streamed answer + (if any) a suggestion/reorder card whose button opens the existing reorder/checkout.

## Error handling / fail-open

- A chat or suggestion-extraction failure must NEVER break the portal page render or the other portal APIs. The chat section degrades gracefully (shows an error bubble), the rest of the portal is unaffected.
- Invalid/expired token → 401/404 as the other portal endpoints do.
- Suggestion resolution failure → no card, answer still streams.

## Testing

- **Unit (pure):** `portal_concierge.build_context` from sample portal/biofield/order rows → asserts the grounding facts are assembled correctly; `resolve_suggestion` → correct reorder URL; the no-data case (portal with no biofield/orders) degrades to a generic-but-valid context.
- **Eval (pass-rate, under Doppler):** the concierge prompt, sampled N times — answers stay grounded in the supplied context (e.g. reference the seeded finding/remedy), and when a complement is warranted a valid `suggestion` is emitted. Assert RATES, never single-shot (LLM-output lesson, `feedback_mock_masked_green_tests`).
- **End-to-end streaming:** if the answer carries any hidden directive, test `stream → accumulate → parse` end-to-end (the `feedback_streaming_directive_endtoend_test` trap). The concierge's `suggestion` is extracted by a *separate* call today (not a trailing directive), so confirm which mechanism is used and test it accordingly.
- **Render-verify (headless, gevent server):** the portal chat section renders, a question streams a reply, a suggestion card renders with a working reorder link, errors degrade gracefully, zero console errors.

## Out of scope

- The **funnel concierge** (`/begin/concierge/chat`, `concierge.html`) stays unchanged — it serves non-portal buyers until sub-project A lands. Not retired here.
- **No portal cart / add-to-invoice** — buy actions link out to the existing reorder/checkout.
- **Sub-project A** (universal portal: every-buyer-TOS gate → portal minting → link delivery → no-scan render) is separate.

## Open items for the plan

- Confirm the exact existing **reorder/checkout link** mechanism the portal `reorder_items` use, so `resolve_suggestion` reuses it verbatim.
- Confirm how the concierge currently extracts its `suggestion` (separate `_CONCIERGE_EXTRACT_SYSTEM` call vs inline) and whether to keep that or switch to a directive.
- Confirm the portal's chat-section placement in `client-portal.html` (which existing section it sits after) and whether the portal token is available to the page JS for the POST.
