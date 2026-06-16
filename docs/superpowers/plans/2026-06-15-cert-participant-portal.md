# Certification Participant Personal Portal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Reuse the practitioner portal to give cert participants personal ordering at cert-level pricing immediately, with reselling (drop-ship + wholesale) gated behind a resale license + approval; plus a self-serve contact toggle and a cohort invite.

**Spec:** `docs/superpowers/specs/2026-06-15-cert-participant-portal-design.md`
**Branch:** `sess/b76661d9-cert-portal`.
**Test invocation:** `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <files> -v` (pure modules also run under the bare venv). Ignore the 2 known pre-existing failures.

**Key existing pieces to reuse (read before editing):**
- `app.py` practitioner routes: `_practitioner_session_pid()` (~6014), `/api/practitioner/quote` + `/api/practitioner/checkout` (~6172-6233, the clone source for personal checkout), dropship routes (~6235+), `/api/practitioner/settings` GET/POST (~6464-6537), register route (~6038-6096, the coach auto-unlock at ~6076).
- `dashboard/practitioner_portal.py`: `portal_data(pid)`, `cart_set/cart_clear`, `register_practitioner` (coach unlock), `unlock_wholesale`, `submit_wholesale_application`/`decide_application`/`list_pending_applications`, `create_magic_link_token`.
- `dashboard/wholesale_checkout.py` `build_order(items, prac, method, ship_to_state, resale_ok)` — pricing+tax+wallet-redeem engine (personal checkout reuses it).
- `dashboard/wallet.py` — earn/redeem rates + functions (`EARN_FEE_FREE_PCT=0.03`, `ORDER_REDEEM_PCT=0.50`).
- `dashboard/practitioner_settings.py` — branding/pricing get/set.
- `static/practitioner-portal.html`, `static/practitioner-settings.html`.

---

## Task 1: Wallet — personal-order fee-free earn rate (3.5%)

**Files:** Modify `dashboard/wallet.py`; Test `tests/test_wallet_personal_earn.py`.

Read `wallet.py` to find the existing fee-free earn (`EARN_FEE_FREE_PCT = 0.03`) and the function that computes/credits earned credit for an order.

- [ ] **Add** `PERSONAL_EARN_FEE_FREE_PCT = 0.035` (do NOT change the existing 0.03).
- [ ] **Add** a pure helper `personal_earn_cents(charged_cents, method)` returning `round(charged_cents * 0.035)` when `method in ("zelle","wise")` else `0`. (If wallet.py already has a generic earn helper that takes a rate, reuse it; otherwise add this thin one.)
- [ ] Test: `personal_earn_cents(10000,"zelle")==350`; `("wise")==350`; `("card")==0`; `(0,"zelle")==0`.
- [ ] Commit `feat(wallet): personal-order fee-free earn rate (3.5%)`.

## Task 2: Personal ordering path (open to cert participants)

**Files:** Modify `app.py` (new routes near the wholesale checkout); maybe `dashboard/practitioner_portal.py` (a personal-cart helper if not reusing `wholesale_cart`); Test `tests/test_practitioner_personal_order.py`.

Clone the wholesale quote/checkout but WITHOUT the `wholesale_unlocked` gate, taxed (resale_ok=False), with personal wallet earn.

- [ ] `POST /api/practitioner/personal/quote` — session-gated (`_practitioner_session_pid`); 401 if not signed in. Returns the cert-level quote for the participant's cart (reuse `portal_data(pid)`'s quote, which already prices at `modules_completed`) + `wallet_balance_cents`. (Reuse the same cart as the wholesale portal — `portal_data` already returns it.)
- [ ] `POST /api/practitioner/personal/checkout` — session-gated. Clone `api_practitioner_checkout` EXACTLY except:
  - **Remove** the `if not data.get("wholesale_unlocked")` 403 block (personal ordering is always allowed for a valid practitioner session).
  - Force `resale_ok = False` (personal purchase is taxed; never resale-exempt).
  - Call `build_order(items, prac, method=…, ship_to_state=…, resale_ok=False)`.
  - On success, in addition to the existing `record_order` + `_ingest_order(channel="personal", …)`, credit personal wallet earn: `wallet.personal_earn_cents(charged_cents, method)` and apply it via the same wallet-credit path the system already uses (read wallet.py for the credit function; if `build_order` already credits fee-free earn at 3%, override/replace with the 3.5% personal rate for this path — do NOT double-credit). Use `channel="personal"` in `_ingest_order`.
  - Keep the method handling (zelle/wise/card), Stripe URL for card, pay_instructions for zelle/wise — identical to the wholesale route.
- [ ] Tests (stub `_wc.build_order`, `_pp.portal_data`, `_ingest_order`, wallet credit, and the session via monkeypatching `_practitioner_session_pid`): personal/checkout works WITHOUT wholesale_unlocked (no 403); passes `resale_ok=False` to build_order; empty cart → 400; not-signed-in → 401; zelle/wise path credits 3.5% personal earn, card path credits 0.
- [ ] Commit `feat(portal): personal ordering at cert-level price (no resale gate, taxed, 3.5% fee-free earn)`.

## Task 3: Resale activation gate (drop-ship + wholesale)

**Files:** Modify `app.py` (registration coach auto-unlock; a portal resale-submit route); `dashboard/practitioner_portal.py` (a submit-for-existing-record helper if needed); Test `tests/test_practitioner_resale_activation.py`.

- [ ] **Stop coach auto-unlock at registration:** in the register route (~app.py:6076) the coach branch calls `_pp.unlock_wholesale(pid)`. Change so coaches are NOT auto-unlocked (licensed still unlock via `register_practitioner` setting `wholesale_unlocked_at`). Keep building the first-module invoice if that logic is separate; just don't flip the unlock. (Read the block; remove only the `unlock_wholesale` call for the coach path.)
- [ ] **Portal resale submission:** `POST /api/practitioner/resale-apply` — session-gated. Body `{resale_license_number, license_state}`. For the logged-in pid, set `application_status='pending'`, `resale_license_number`, `license_state` on their EXISTING record (add a `practitioner_portal.submit_resale_for_pid(pid, resale_license_number, license_state)` helper that UPDATEs by id; do NOT create a new row). Email Rae (reuse the admin-notify pattern from the existing apply route). Returns `{ok, status:"pending"}`.
- [ ] The existing `/admin/wholesale` approve flow already sets `wholesale_unlocked_at` on approve via `decide_application(pid, approve=True)` — confirm it works for a coach record (it updates by pid regardless of role). The drop-ship + wholesale checkout gates (`wholesale_unlocked`) already block until then — no change needed there.
- [ ] Tests: resale-apply sets pending + resale_license_number for the pid (monkeypatch supabase cursor); not-signed-in → 401; missing resale_license_number → 400. Confirm (unit) that a coach with `wholesale_unlocked=False` is still blocked from `/api/practitioner/dropship/checkout` (the existing gate).
- [ ] Commit `feat(portal): resale-license activation gate for reselling (coaches no longer auto-unlock)`.

## Task 4: Self-serve contact toggle in settings

**Files:** Modify `app.py` (`/api/practitioner/settings` GET/POST); `static/practitioner-settings.html`; Test append to a settings test or new `tests/test_practitioner_show_contact_setting.py`.

- [ ] GET `/api/practitioner/settings` also returns `show_contact` (read from the practitioners row for the pid).
- [ ] POST accepts `show_contact` (bool) and UPDATEs `practitioners.show_contact` for the pid (parameterized).
- [ ] `practitioner-settings.html`: add a checkbox "Show my contact info in the public finder (default off)" bound to `show_contact`, wired into the existing settings load/save JS.
- [ ] Test: POST show_contact=true sets it for the pid; GET returns it; default false.
- [ ] Commit `feat(portal): self-serve finder contact-visibility toggle in settings`.

## Task 5: Portal landing UI for cert participants

**Files:** Modify `static/practitioner-portal.html`; (uses the APIs from Tasks 2-3). Smoke test: route serves 200.

- [ ] In the portal page, render for a cert participant: cert level (N/12), wallet balance, a **personal-order catalog/cart** (calls `/api/practitioner/personal/quote` + `/api/practitioner/personal/checkout`), and a **Reselling** section: if `wholesale_unlocked` → link to `/practitioner/dropship`; else → an "Activate reselling" form (resale license # + state) posting `/api/practitioner/resale-apply`, with copy explaining a resale license is required. Reuse existing portal styling + the existing `/api/practitioner/portal-data` for `wholesale_unlocked`/`modules_completed`/`wallet_balance_cents`. Escape all interpolated values.
- [ ] Smoke: `GET /practitioner/portal` returns 200 (existing route).
- [ ] Commit `feat(portal): cert-participant portal view (personal catalog + reselling activation)`.

## Task 6: Cohort portal invite

**Files:** Modify `app.py` (console-gated invite route); Test `tests/test_cert_portal_invite.py`.

- [ ] `POST /api/cert/portal-invite` — console-gated (mirror `/api/cert/show-contact`). Body `{email}`. Resolve pid via `_pp.id_for_email(email)`; if found, mint `create_magic_link_token(pid, email)` and email via `_send_practitioner_magic_link(email, name, link)`. Return `{ok, sent:bool}`. (No enumeration concern — console-gated.)
- [ ] Test: missing email → 400; unknown email → `{ok:true, sent:false}`; known email → mints token + sends (monkeypatch id_for_email + the mailer). Console-gated 401 without key.
- [ ] Commit `feat(cert): console-gated portal-invite (magic link to a cert participant)`.

## After all tasks
- [ ] Full new-suite run green; `import app` smoke.
- [ ] Open PR (Glen merges). Then: apply nothing new to DB (no schema change in this plan — `show_contact` column already live). Glen flips no flags (portal isn't flag-gated).
- [ ] Post-merge: Glen merges → deploy → send the cohort invites via `/api/cert/portal-invite` for the 11 (+ Mona).

## Notes
- No new DB columns needed (uses existing portal_role/tier/modules_completed/wholesale_unlocked_at/resale_license_number/application_status/show_contact/wallet_balance_cents).
- Reuse `escapeHtml`, existing styling, existing session/auth, existing pricing/tax. Don't restructure the portal; extend it.
- The personal checkout MUST NOT be resale-exempt (resale_ok=False) and MUST NOT require wholesale_unlocked — those two differences from the wholesale route are the crux.
