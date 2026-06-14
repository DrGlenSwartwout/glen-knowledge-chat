# Practitioner-scoped support chat

A self-contained product-selection chat on the practitioner portal pages. It recommends
**only the practitioner's Functional Formulations** and **never emits RM-direct links or
pricing** — self-contained by construction.

## How it stays self-contained (no RM leakage)
`dashboard/practitioner_chat.scoped_reply(message, history, catalog)`:
1. The model is given **only** the practitioner's FF catalog (slug/name/description) — no RM
   knowledge base, no concierge synthesis.
2. `suggested_slugs` are **validated against the catalog** (hallucinated/out-of-catalog slugs dropped).
3. The reply is **scrubbed** of any URL and of the brand strings `truly.vip`/`truly.so`/
   `remedymatch`/`remedy match`/`illtowell` → "our selection".
On LLM error → a safe fallback reply with no suggestions.

## Endpoints
- `POST /api/practitioner/chat` — authed (practitioner session). Returns `{reply,
  suggestions:[{slug, name, price_cents}]}` priced at the practitioner's price.
- `POST /api/client/<code>/chat` — resolves the practitioner by dispensary code (404 unknown),
  **consent-gated** (patient ToS; 403 `need_optin`), same scoped reply + priced suggestions.

## Widget
A collapsible launcher at the **bottom** of each page (open/close). Suggestions render as
**add-to-cart chips** (with price) that reuse the page's own cart; **no links rendered**.
- **Drop-ship + wholesale pages:** available by default.
- **Client page:** mounted only when the practitioner's **`chat_enabled`** setting is true
  (stored in `practitioner_settings` branding, **default false**; exposed via the client
  catalog endpoint; toggled on the settings page).

## Done
Completes the practitioner drop-ship + white-label portal + scoped chat (Plans 1–5).
