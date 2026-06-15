# Biofield checkout + readiness gate (Phase 2a)

Sells the $300 Causal Biofield Analysis as a points-redeemable checkout item, then gates the
consult booking on a readiness checklist (photo / intake / fresh voice scan) before revealing a
booking link and dropping a 48-hour prep task. Ships dark behind `BIOFIELD_CHECKOUT_ENABLED`.

## Flow
1. **Pay.** `POST /biofield/checkout {email, name, points_to_redeem_cents?}` prices a single
   $300 service line through the engine (loyalty points apply, floor-protected at the points
   floor — $300 floors at $129), creates a one-line QBO invoice + a Stripe session
   (`metadata.kind="biofield"`), and on the checkout-return seeds the readiness record
   (`seed_paid via="stripe"`), settles points, and redirects to `/biofield/ready`.
   Alternatively a customer who paid in **Practice Better** reaches the gate via a magic link
   and confirms payment there (`POST /api/biofield/confirm {item:"payment"}` → `seed_paid via="pb"`).
2. **Readiness gate** (`/biofield/ready`, magic-link/member auth): three items, each green when
   confirmed, else an action:
   - **Photo** — a face photo (remote-biofield visualization), uploaded in the gate
     (`POST /api/biofield/photo`).
   - **Intake** — auto-detected from `inbound_leads` (scoreapp/practice-better/concierge),
     else "complete intake" (Truly.VIP/Join) + self-confirm.
   - **Fresh voice scan (within 7 days)** — self-confirm in 2a (the scan DB is local to Glen's
     Mac, see [[project_e4l_scan_ingestion]]); link is Truly.VIP/E4L. Auto-verify is 2b.
3. **Book.** When paid + all three green, `GET /api/biofield/ready` returns `booking_url`
   (`BIOFIELD_BOOKING_URL`) and the page shows "Book your session". `POST /api/biofield/book`
   marks booked and inserts one `todos` row ("Biofield prep due 48h — <email>",
   `dedup_key=biofield-prep-<email>-<order_ref>`, owner `glen`) so the team runs the analysis /
   program design / report inside the window. Idempotent.

## Auth
Magic-link mirrors the reorder flow: `POST /biofield/request` mints an `auth_tokens` row
(`purpose="biofield"`) and emails `…/biofield/auth/<token>`; that consumes the token and sets the
HttpOnly `rm_biofield_email` cookie. A member session also satisfies `_biofield_email()`.

## PHI / storage
Client **photos are PHI**. They persist to a **private** path
`DATA_DIR/biofield-photos/<sha256(email)>.<ext>` — never under `static/` and not web-served; only
a path reference + an on-file flag live in the `biofield_readiness` table. Upload validates
`image/*` + a 10 MB cap; the filename is a hash of the email (no user-controlled path). Review
access controls before go-live.

## Config / flags
- `BIOFIELD_CHECKOUT_ENABLED` (default off) — gates the checkout + every `/api/biofield/*` route.
- `BIOFIELD_BOOKING_URL` — the booking link revealed when ready (PB scheduler / Zoom).
- `_STRIPE_ACTIVE` required for the card checkout (mirrors the other paid flows).

## Deferred (2b)
Real E4L scan-freshness auto-verify (needs the scan data mirrored to a server-reachable store);
Practice Better intake/photo/payment API; auto-drafting the analysis via the
`dr-glen-swartwout-e4l-scan-remedy-matcher` agent; refund/expiry of an unbooked paid Biofield.
