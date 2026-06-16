# Spec: Auth Seam + Role-Aware Tokenized Client Portal

**Date:** 2026-06-16
**Status:** Approved (design) — pending implementation plan
**Slice owner:** Glen
**Approach:** Option A — adopt and generalize the existing `/portal/<token>`, do not rebuild.

---

## Goal

Ship the first slice of the Unified Personal Portal north-star: a **role-aware, person-connected client portal** reached by a tokenized link (no login required), plus a clean **auth seam** so real login drops in next slice without a rewrite.

This slice is deliberately **token-only for clients now, real auth scaffolded** (Glen's decision). It is the keystone that features #2 (role-aware sales pages) and #3 (biofield delivery) plug into later.

## Non-goals (YAGNI — explicitly OUT of this slice)

- Real client login turned **on** (next slice flips a flag).
- Role-aware sales-page *content* (feature #2).
- Native biofield *data entry* to replace FMP (feature #3 entry side).
- Per-client custom branding.
- Practitioner/affiliate role blocks beyond role badges — those portals already exist separately (`/practitioner/portal`, `/cert`).

---

## What already exists (reuse, do not rebuild)

Confirmed by codebase exploration on 2026-06-16:

- **`/portal/<token>` route** — `app.py:7177-7267`, backed by `dashboard/client_portal.py` (`client_portals` table). Non-expiring token, hashed SHA256, keyed by email. Renders `static/client-portal.html` with video + causal-chain layers + reorder items + pricing. This is the page used for Brooke's biofield walkthrough.
- **`/invoice/<token>`** — `app.py:17635+`, `dashboard/orders.py`. Tokenized pay page (view/edit/add/apply-points/pay/chat). Source of the reorder + pay JS and styling to reuse.
- **Token pattern** — `secrets.token_urlsafe(32)` + `hashlib.sha256(...).hexdigest()`, stored in `auth_tokens` (`token_hash`, `email`, `purpose`, `extra` JSON, `expires_at`, `consumed_at`). Mint/validate helpers in `dashboard/practitioner_portal.py`.
- **Practitioner magic-link auth** — `app.py:6034+`, `dashboard/practitioner_portal.py:305+`: email → 15-min magic-link token → consume → 30-day session cookie. **This is the code the client login routes will copy.**
- **People model** — `people` table, `app.py:11264+` (`_init_people_table`), upsert at `app.py:9343+` (`upsert_person`). Email-unique PK; `roles` / `tags` JSON arrays; address, points, order, and session fields already present.
- **Pricing/points engine + Stripe pay** — live; reused for reorder.
- **Email send** — Gmail API → SMTP fallback (`dashboard/inbox.py`).

---

## Architecture

Two new functions become the **only** code that knows about identity and portal content. The page and APIs call these and never touch tokens or roles directly.

### Component 1 — `resolve_identity(request) → Identity | None`
New module: `dashboard/portal_identity.py`

- **What it does:** Turns an incoming request into an `Identity` or `None`.
- **Identity shape:** `{ person_id, email, roles: [..], auth_method: "token" | "session" }`
- **Today (token branch):** read the path token → validate against `auth_tokens` (purpose `client_portal`) or the existing `client_portals` token → resolve to a `people` row → return `Identity`.
- **Tomorrow (session branch, scaffolded):** read a practitioner-style session cookie → return the **same** `Identity` shape with `auth_method:"session"`.
- **Why:** single choke point. "Real auth scaffolded" = this seam exists, is tested, and the session branch is the documented drop-in point. No page/API rewrite needed when login goes live.
- **Depends on:** `auth_tokens`, `client_portals`, `people`.

### Component 2 — `get_portal_view(person_id) → dict`
New module: `dashboard/portal_view.py`

Composes the role-aware payload from the unified `people` row:

| Block | Visibility | Source |
|-------|-----------|--------|
| `account` | always | `people` row (name/email/address/points/role badges) |
| `orders` | client role | order history + reorder at role price via live pricing/points engine |
| `biofield` | present-if-data | existing `client_portals` content (causal-chain layers + video) |
| `upgrade` | always (stub) | `{ enabled: false, placeholder: true }` — reserved seam for feature #2 |

- **Why:** one assembler replaces the standalone `client_portals` blob read. Biofield becomes one section of a larger view rather than the whole page.
- **Depends on:** `people`, `orders`, `client_portals`, pricing engine.

### Component 3 — People unification
- Add a `person_id` column to `client_portals`. On read, resolve by `person_id` (fallback: match by email, then backfill `person_id`).
- Lazy minimal `upsert_person` if a token's email has no `people` row, so the portal always has a person.
- Minimal migration; no new heavy tables.

### Component 4 — The page
- Generalize `static/client-portal.html` from a biofield-walkthrough into a **role-aware shell** that renders whichever blocks `get_portal_view` returns and hides absent ones.
- Reuse `invoice.html` styling + reorder/pay JS.
- Brooke's existing biofield content keeps working — now rendered as the `biofield` block.

### Component 5 — Mint + send
- Console action "Send portal link" on the People hub: mint a long-lived `client_portal` token for a person, email the `/portal/<token>` link via existing Gmail/SMTP infra.
- Idempotent: reuse an existing token for the person if present.

### Component 6 — Auth scaffold (built, flagged OFF)
- Copy practitioner magic-link routes as `/portal/login` + `/portal/login-verify`, gated by `CLIENT_LOGIN_ENABLED=false`.
- They exist and are tested but dark. Next slice = flip the flag + surface a login link. They target the `resolve_identity` session branch.

---

## Data flow

1. Client clicks emailed `/portal/<token>` → `resolve_identity` → `person_id` → `get_portal_view` → render shell → role blocks populate.
2. **Reorder:** client adds items → reuse the invoice/order API to create a `proposed` order → existing pay flow.
3. **Address edit:** writes back to the `people` row (feeds in-house order entry).

## Error handling

- Invalid/expired token → friendly "this link expired — request a new one" page, never a raw 403.
- Person-not-found → graceful message or lazy minimal upsert.
- Missing `orders` / `biofield` data → that section is simply hidden; never errors.

## Testing

- **Unit:** `resolve_identity` (token valid / invalid / expired; session branch returns correct shape) and `get_portal_view` (role flags drive which blocks appear; biofield present vs absent; bare client with no orders).
- **Integration:** `GET /portal/<token>` renders expected blocks for (a) a client with orders + biofield and (b) a bare client.
- **Invocation:** deploy-chat pytest via `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest`.
- **Isolation discipline:** mock live Supabase calls; seed a tmp `DATA_DIR` and monkeypatch `_default_db_path` to avoid ambient `chat_log.db` bleed. Work happens in a session git worktree.

## Definition of done

- A real client receives a portal link, opens it with no login, and sees account + order history + reorder + (if present) their biofield map, with a visible placeholder where sales/upgrade will go.
- `resolve_identity` and `get_portal_view` are unit-tested; the session branch is in place but inactive.
- Client login routes exist behind `CLIENT_LOGIN_ENABLED=false`.
- Full suite green under the isolation rules above.
