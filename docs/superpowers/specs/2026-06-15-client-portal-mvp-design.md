# Tokenized Per-Client Portal — MVP (Brooke Webb)

**Date:** 2026-06-15
**Status:** Approved design → implementation
**Context:** First concrete step toward the "Create Your Own Healing Adventure" unified personal portal (north-star: one role-aware portal per human). This MVP is a **no-login, token-gated per-client page** that gives one client (Brooke Webb) a home for their biofield video, their causal-chain healing path, and one-click reorder of their remedies. Deferred to later phases: login/auth, AI concierge, role-awareness, scan history, full adventure-map UI.

## Goals
- Prove the "personal home for the whole healing process" feel with the least possible build.
- Reuse the existing `/invoice/<token>` pattern (token → static SPA → token-scoped API; no login).
- Be reusable for future clients via an admin seed endpoint (no code change per client).

## Non-Goals (explicitly out of scope)
- User accounts, passwords, magic-link login.
- AI concierge chat on the page.
- Role-awareness (student/practitioner/affiliate surfaces).
- Pulling content live from People Hub / E4L. Content is seeded per client.
- Full visual "adventure map." MVP is "clean & warm with light journey hints."

## Architecture

Mirrors `/invoice/<token>` exactly.

### Data: new module `dashboard/client_portal.py`
- New SQLite table `client_portals` in `LOG_DB`:
  - `id` INTEGER PK
  - `token_hash` TEXT UNIQUE (sha256 of the urlsafe token — never store raw token)
  - `email` TEXT
  - `name` TEXT
  - `content_json` TEXT (the full page payload)
  - `created_at`, `updated_at` TEXT
  - Durable: no short TTL (this is the client's home; token is the auth).
- Functions:
  - `init_client_portal_table(cx)` — idempotent `CREATE TABLE IF NOT EXISTS`.
  - `upsert_portal(email, name, content: dict) -> (token, portal_id)` — generates `secrets.token_urlsafe(32)`, stores its sha256, writes content_json. If a row already exists for `email`, update content + rotate/keep token (keep existing token on update so links don't break; only mint on first create).
  - `get_portal_by_token(token) -> dict | None` — sha256 lookup, returns `{name, email, content}` or None.

### `content_json` shape
```json
{
  "greeting": "Aloha Brooke. This is your personal healing home ...",
  "video": { "url": "https://app.heygen.com/share/<id>", "label": "Watch your personal Biofield walkthrough" },
  "layers": [
    { "n": 1, "title": "Calming your nervous system",
      "meaning": "The most recent, surface layer ...",
      "remedy": "Sedativa homeopathic in Terrain Restore",
      "dosing": "10 drops under the tongue or in water, 3x/day, before meals, with an affirmation." },
    ... 7 layers total, top (surface) -> bottom (deepest root) ...
  ],
  "reorder_items": [
    { "slug": "terrain-restore", "qty": 1 },
    { "slug": "mitochondrial-biogenesis", "qty": 1 },
    { "slug": "nous-energy", "qty": 1 },
    { "slug": "ozonated-olive-oil", "qty": 1 },
    { "slug": "lavender-msm-lotion", "qty": 1 },
    { "slug": "stress-release", "qty": 1 },
    { "slug": "brain-cleanse", "qty": 1 }
  ]
}
```

### Routes (in app.py, beside the invoice routes)
- `GET /portal/<token>` → `send_from_directory(STATIC, "client-portal.html")`. Always 200 (SPA resolves token via API; invalid token shows a friendly "link not found" state).
- `GET /api/portal/<token>` → `get_portal_by_token`; 200 with `{name, greeting, video, layers, reorder_items_display}` or 404 `{error}`. `reorder_items_display` enriches each slug with name + price_cents from `_PRODUCTS` for rendering.
- `POST /api/portal/<token>/checkout` → validates token, reads the seeded `reorder_items`, builds a **real live Stripe checkout** via the existing `_stripe_checkout_url_for_reorder(email, items)` helper; returns `{url}`. 404 on bad token; 502 if Stripe fails (graceful, like existing checkout routes).
- `POST /admin/portal/upsert` → gated by `CONSOLE_SECRET` (header/param, same pattern as other `/admin/*`). Body `{email, name, content}` → `upsert_portal` → returns `{token, url}`. The bridge for seeding the live Render DB (curl once per new client). Reusable.

### Page `static/client-portal.html`
Standalone SPA matching `invoice.html` style (self-contained `<style>`, brand color `#7c5cbf`, card layout, `max-width: 640px`, mobile-first). Sections:
1. **Warm greeting** — Aloha + one line framing this as their personal healing home.
2. **Video card** — "▶ Watch your personal Biofield walkthrough" button opening the HeyGen share link in a new tab (button, NOT iframe — HeyGen share pages set X-Frame-Options and won't embed reliably). Optional thumbnail.
3. **Your healing path** — the 7 layers rendered top→bottom as a gentle connected trail (subtle numbered nodes + connecting line), each a soft card: title, plain-English meaning, remedy, dosing. Light journey hint, not a game map.
4. **Reorder card** — lists the remedies (name + price from catalog), one **"Reorder my remedies"** button → POSTs to the checkout endpoint → redirects to Stripe.
5. **Footer** — "With aloha, Dr. Glen & Rae · Remedy Match LLC".
Invalid/expired token → friendly fallback ("We couldn't find that link — please check the link Dr. Glen sent you.").

## Remedy → catalog slug mapping (verified against data/products.json)
All 7 layers map to real catalog slugs; the custom essence/homeopathic additions (Sedativa, White Bleeding Heart, Hematite, Black Seed) are compounded into their base SKUs (Terrain Restore, Ozonated Olive Oil), so reorder needs no "contact us" fallback for Brooke.

## Testing (TDD)
`tests/test_client_portal_routes.py`, mirroring `tests/test_cert_portal_routes.py` (tmp_path `LOG_DB`, table init, token mint helper, stubbed Stripe):
- `get_portal_by_token` round-trips content for a seeded row.
- `GET /portal/<token>` serves the SPA (200).
- `GET /api/portal/<good>` returns enriched content; `GET /api/portal/<bad>` → 404.
- `POST /api/portal/<good>/checkout` calls the reorder-checkout helper with the seeded items and returns a URL (helper stubbed); `<bad>` → 404.
- `POST /admin/portal/upsert` with correct `CONSOLE_SECRET` creates a row + returns token/url; wrong/absent secret → 401/403.

## Safety / rollout
- **Additive + inherently dark:** no route is reachable until a token is minted and shared, so no feature flag needed. Nothing changes for existing users.
- Built in an isolated worktree on branch `sess/9f6936ca`; PR opened for Glen to merge (not self-merged).
- **Seeding Brooke (post-merge/deploy):** curl `POST /admin/portal/upsert` on the live Render app with Brooke's content (greeting + HeyGen share URL + 7 layers + 7 reorder slugs) → receive `{token, url}` → hand Glen the portal URL.

## Future phases (not now)
Magic-link auth · per-client AI concierge wired to their record · role-aware feature pages · E4L scan history · full adventure-map UI · subscriptions/rewards surfaced.
