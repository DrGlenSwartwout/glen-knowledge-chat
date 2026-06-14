# Practitioner settings: pricing + white-label branding

Where a practitioner sets their FF selling prices and customizes their portal/client-page
branding. Stored in a local sqlite `practitioner_settings` table (chat_log.db) keyed by
practitioner id — no Supabase migration.

## Store (`dashboard/practitioner_settings.py`)
- `practitioner_settings(practitioner_id PK, branding_json, pricing_json, updated_at)`.
- `get_settings(cx, pid)` → `{branding:{...}, pricing:{default_markup_pct, overrides:{slug:cents}}}` (defaults filled at read time).
- `set_branding` / `set_pricing` (upsert per column).
- `price_cents_for(cx, pid, slug, *, retail_cents, map_cents)`: per-SKU override → default markup % over retail → retail; **clamped up to MAP**.

## Pricing
- A **default markup %** over RM retail, with optional **per-SKU dollar overrides**. The
  client page prices each FF at this (via `dropship_checkout._practitioner_price_cents`,
  wired to the store; best-effort fallback to `max(retail, MAP)` so a settings error never
  breaks checkout).
- Anything below **MAP $67** is clamped up; the settings API reports clamps in `clamped[]`.

## Branding (white-label)
`practice_name, contact_details, web_link, logo_url, photo_url, brand_color_1,
brand_color_2`. **Images are URL fields in v1** (an upload widget reusing `/clips/upload`
is a later add). Applied: **client page** = full (photo+logo+name+contact+link+2 colors);
**drop-ship + wholesale pages** = logo + name + 2 colors. Missing asset/field → RM default
(broken images guarded with `onerror`; invalid hex skipped). Never breaks a page.

## API + page
- `GET/POST /api/practitioner/settings` (authed) — read/write branding + pricing.
- `GET /practitioner/settings` → `static/practitioner-settings.html` (markup %/per-SKU $ with
  live $↔% companion + MAP warning; branding fields + color pickers + live preview).
- Branding is surfaced to pages via `GET /api/client/<code>/catalog` (client) and
  `/api/practitioner/portal-data` (drop-ship/wholesale), both extended additively.

## Done
This completes the practitioner drop-ship + white-label portal (Plans 1-4). Plan 5 = the
practitioner-scoped support chat (self-contained, no RM links).
