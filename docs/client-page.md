# Branded client page (patient-paid)

A patient reaches a practitioner's `/dispensary/<code>` link and buys Functional
Formulations at the **practitioner's price**; we ship to the patient and credit the
practitioner's **margin** to their wallet. Replaces the old dispensary flow (patient bought
RM retail → flat $20/bottle).

## Flow
- **`GET /dispensary/<code>`** — sets the `rm_dispensary` cookie and serves
  `static/practitioner-client.html` (no longer redirects to the RM funnel). Unknown code → 404.
- **`GET /api/client/<code>/catalog`** — the practitioner's sellable FF, each at
  `dropship_checkout.practitioner_price_for(pid, slug)` (≥ MAP $67; defaults to RM retail
  until the Plan-4 price-setting UI). Pure Powders / info_only excluded.
- **`POST /api/client/<code>/checkout`** — patient pays the practitioner's price **S**:
  - **Consent-gated** (the patient ToS opt-in via `/begin/unlock` → `is_member`); 403
    `need_optin` otherwise.
  - `build_client_order` invoices the **patient** at S (flat per-bottle, **no volume**),
    ships to the patient (US only), `source="dispensary"`, GET recorded-not-charged.
  - Card → Stripe with metadata `kind="client", practitioner_id, margin_cents, invoice_id`.
- **On paid** (`/begin/checkout-return`, `kind="client"`) — `wallet.earn_dropship_margin`
  credits the practitioner the **margin** (S − base − fee), idempotent per invoice. (Replaces
  the flat $20/bottle `earn_dropship`; the old `rm_dispensary`→funnel path is superseded.)

## Pricing
S = practitioner's price (≥ MAP $67, default retail). base = blended wholesale at the order's
total bottles; fee = 33% of (S − base); margin = S − base − fee → practitioner wallet.

## Not here
The practitioner price-**setting** UI ($/%) + white-label branding = **Plan 4**. The
practitioner-scoped support chat (self-contained, no RM links) = **Plan 5**.
