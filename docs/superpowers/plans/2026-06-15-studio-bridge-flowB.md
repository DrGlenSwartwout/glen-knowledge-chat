# Studio.com Bridge — Flow B (free group month) Plan — Mechanic 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A customer who joined Studio.com at **studio.com/drglen** can claim their **first month of live group coaching free**, then auto-continue at $99/mo (cancel anytime). Confirmation of the Studio signup is by receipt upload / self-attest (the dashboard exposes no emails). Ships dark behind `STUDIO_BRIDGE_ENABLED`.

**Architecture:** Reuse the Mechanic 1 membership rail (`subscriptions.create_membership` + the charge cron) and the chat receipt-upload. New: a `mode=setup` Stripe Checkout to vault a card with **no charge** (the customer paid Studio.com, not us), then create a `kind='membership'` subscription whose first charge is **one month out** (the free month), $99/mo thereafter. A consent gate (`is_member`) doubles as Flow A's win — claiming makes the Studio user our free member. A small `studio_bridge_claims` table makes the grant idempotent (one free-month grant per email).

**Tech Stack:** Python 3.11, Flask, Stripe (setup-mode vault + off-session via the cron), sqlite, pytest.

**Spec:** `docs/superpowers/specs/2026-06-15-upgrade-incentive-ladder-design.md` (Mechanic 2, Flow B). Studio.com pays us rev-share on every participant incl. referrals → this pays twice (rev-share + the $0-cost free group month is an acquisition carrot). Confirmation = receipt/self-attest (no email export). Compliance: auto-charge after the free month is a negative-option offer → explicit opt-in + disclosure + reminder + one-click cancel (same as Mechanic 1).

**Reuse:** `subscriptions.create_membership(...)` / `active_memberships_by_email` / `add_months` / the charge cron membership branch (all merged, #113); `group_bundle.MEMBERSHIP_AMOUNT_CENTS` (9900); `is_member` consent gate + `_normalize_*` (client checkout pattern); the chat OCR upload for the receipt; `stripe_pay.get_session`; the heads-up/cancel/portal from Mechanic 1.

**Flow A is OUT of scope here** (deferred as positioning — a Studio-user landing reusing the existing free funnel + Truly.VIP/E4L scan).

**Test invocation:** pure → `~/.venvs/deploy-chat311/bin/python -m pytest <path> -q`. App → `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path> -q` (worktree; ignore the 2 known pre-existing failures).

---

### Task 1: Stripe `mode=setup` $0 card vault

**Files:** Modify `dashboard/stripe_pay.py`; Test `tests/test_stripe_setup_session.py`

- [ ] **Step 1: Failing test** — stub `stripe_pay._post`/`_get` (monkeypatch) and assert:
  - `create_setup_session(customer_email, metadata, success_url, cancel_url)` POSTs to `/checkout/sessions` with `mode=setup`, the email, metadata, and the urls; returns the session dict (`{id, url}`).
  - `get_setup_intent(si_id)` GETs `/setup_intents/<id>` and returns it (with `customer`, `payment_method`).

```python
import dashboard.stripe_pay as sp

def test_create_setup_session(monkeypatch):
    captured = {}
    monkeypatch.setattr(sp, "_post", lambda path, params: captured.update(path=path, params=params) or {"id": "cs_1", "url": "https://stripe/x"})
    out = sp.create_setup_session(customer_email="p@x.com", metadata={"kind": "studio_bridge", "email": "p@x.com"},
                                  success_url="https://h/return", cancel_url="https://h/cancel")
    assert out["url"].startswith("https://stripe/")
    assert captured["path"] == "/checkout/sessions"
    assert captured["params"]["mode"] == "setup"
    assert captured["params"]["customer_email"] == "p@x.com"
    assert captured["params"]["success_url"] == "https://h/return"
    assert captured["params"]["metadata[kind]"] == "studio_bridge"

def test_get_setup_intent(monkeypatch):
    monkeypatch.setattr(sp, "_get", lambda path: {"id": "si_1", "customer": "cus_1", "payment_method": "pm_1"} if path == "/setup_intents/si_1" else {})
    si = sp.get_setup_intent("si_1")
    assert si["customer"] == "cus_1" and si["payment_method"] == "pm_1"
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** in `dashboard/stripe_pay.py` (mirror `create_checkout_session`'s param-flattening for `metadata[...]`):
```python
def create_setup_session(*, customer_email, metadata, success_url, cancel_url) -> dict:
    """Stripe Checkout in mode=setup — vaults a card with NO charge (for off-session use)."""
    params = {
        "mode": "setup",
        "customer_email": customer_email,
        "success_url": success_url,
        "cancel_url": cancel_url,
        "payment_method_types[0]": "card",
    }
    for k, v in (metadata or {}).items():
        params[f"metadata[{k}]"] = v
    return _post("/checkout/sessions", params)


def get_setup_intent(si_id: str) -> dict:
    return _get(f"/setup_intents/{si_id}")
```
(Match the existing `_post`/`_get` signatures + the metadata-flatten style already used by `create_checkout_session`.)

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(studio-bridge): stripe setup-mode card vault`

---

### Task 2: claims store (idempotent grant)

**Files:** Create `dashboard/studio_bridge.py`; Test `tests/test_studio_bridge_store.py`

- [ ] **Step 1: Failing test** — `init_table(cx)`; `record_pending(cx, email, *, signup_via)` upserts a claim (status pending); `mark_granted(cx, email, sub_id)`; `get(cx, email)`; `already_granted(cx, email)` True only after grant. Idempotent: a second `record_pending` keeps the same row; `already_granted` stays False until `mark_granted`.

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** `dashboard/studio_bridge.py` (pure; cx passed). Table:
```sql
CREATE TABLE IF NOT EXISTS studio_bridge_claims (
  email TEXT PRIMARY KEY, signup_via TEXT, status TEXT NOT NULL DEFAULT 'pending',
  sub_id INTEGER, created_at TEXT, granted_at TEXT
)
```
`record_pending` = INSERT OR IGNORE then UPDATE signup_via; `mark_granted` = UPDATE status='granted', sub_id, granted_at; `already_granted(cx,email)` = status=='granted'; `get` returns dict or None.

- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(studio-bridge): claims store`

---

### Task 3: claim endpoints + page + grant on return

**Files:** Modify `app.py`; Create `static/studio-claim.html`; Test `tests/test_studio_bridge_routes.py`

- [ ] **Step 1: Failing test** — `STUDIO_BRIDGE_ENABLED=1`, tmp `LOG_DB`. 
  - `POST /api/studio/claim {email, name, attest|receipt}` consent-gated (`is_member` → 403 `need_optin` if not); on pass, `record_pending(via="self-attest"|"receipt")` + returns a Stripe setup `stripe_url` (stub `stripe_pay.create_setup_session`).
  - `GET /studio/claim-return?session_id=...` with a stubbed `get_session` → `{setup_intent: "si_1", ...}` and stubbed `get_setup_intent` → `{customer, payment_method}`: creates a `kind='membership'` subscription for the email with `amount_cents=9900`, `cadence_months=1`, `next_charge_date == add_months(today,1)`; `studio_bridge.mark_granted`; idempotent (second return → no 2nd membership). 
  - Flag off → routes disabled. Unconsented → 403.
  - `GET /studio/claim` page → 200 HTML, contains "Studio".

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** in `app.py` (flag-gate all on `STUDIO_BRIDGE_ENABLED`):
  - `POST /api/studio/claim`: consent gate (mirror `api_client_checkout`'s `is_member` block); `studio_bridge.record_pending(cx, email, signup_via=<"receipt" if a receipt uploaded else "self-attest">)` (best-effort OCR the receipt via the chat attachment path, optional); create a setup session `stripe_pay.create_setup_session(customer_email=email, metadata={"kind":"studio_bridge","email":email}, success_url=f"{PUBLIC_BASE_URL}/studio/claim-return?session_id={{CHECKOUT_SESSION_ID}}", cancel_url=f"{PUBLIC_BASE_URL}/studio/claim")`; return `{ok, stripe_url}`.
  - `GET /studio/claim-return`: get_session → `setup_intent` id → `get_setup_intent` → `customer` + `payment_method`; guard `if not studio_bridge.already_granted(cx,email)`: `sid=subscriptions.create_membership(cx, email=email, stripe_customer_id=cus, stripe_payment_method_id=pm, amount_cents=group_bundle.MEMBERSHIP_AMOUNT_CENTS, next_charge_date=subscriptions.add_months(date.today().isoformat(),1))`; `studio_bridge.mark_granted(cx,email,sid)`. Redirect to `/studio/claim?done=1`. Best-effort; never 500 the redirect.
  - `GET /studio/claim` page (`static/studio-claim.html`): served no-store; explains "Joined at studio.com/drglen? Claim your first month of live group coaching free." Collects name+email+ToS (consent), an upload-receipt or "I joined at studio.com/drglen" self-attest, and an **explicit opt-in** disclosure: "After your free month, live group continues at $99/mo unless you cancel." Posts `/api/studio/claim` → redirects to the Stripe setup URL. The word "Studio" must appear. No em dashes / ALL CAPS.

- [ ] **Step 4: Run → pass.** Regression: `… -m pytest tests/test_studio_bridge_routes.py tests/test_membership_charge_cron.py tests/test_subscriptions_membership.py -q`.
- [ ] **Step 5: Commit** — `feat(studio-bridge): claim flow + setup-vault grant of free group month`

---

### Task 4: flag + doc + suite

**Files:** Create `docs/studio-bridge.md`

- [ ] **Step 1:** Confirm every route is gated by `STUDIO_BRIDGE_ENABLED` (default off).
- [ ] **Step 2:** `docs/studio-bridge.md`: Flow B (claim → consent + signup confirm + $0 card vault → free first month → $99/mo auto-continue), the rev-share rationale (pays twice), receipt/self-attest confirmation (no Studio email export), the negative-option opt-in/disclosure/reminder/cancel, the `STUDIO_BRIDGE_ENABLED` flag, and that Flow A (Studio-user clinical wedge) is deferred as positioning.
- [ ] **Step 3:** Suite green: `… -m pytest tests/test_stripe_setup_session.py tests/test_studio_bridge_store.py tests/test_studio_bridge_routes.py tests/test_membership_charge_cron.py -q`.
- [ ] **Step 4:** Commit — `docs(studio-bridge): Flow B free group month`

---

## Self-review
- **Spec coverage:** studio.com/drglen claim → first month live group free → $99/mo auto-continue (Task 1 vault + Task 3 grant on the Mechanic 1 rail); confirmation by receipt/self-attest (Task 3); consent gate doubles as the member win; idempotent one-grant-per-email (Task 2); negative-option opt-in/disclosure (Task 3 page); `STUDIO_BRIDGE_ENABLED` dark flag.
- **Type consistency:** `create_setup_session`/`get_setup_intent`; `studio_bridge` (init/record_pending/mark_granted/already_granted/get); membership via `create_membership(...,next_charge_date=add_months(today,1))`; metadata `kind="studio_bridge"`.
- **Deferred:** Flow A clinical-wedge onboarding (positioning); auto-verify of the Studio signup (no email export — receipt/self-attest only); refund/clawback.
- **Risk:** money + auto-renewal. Mitigations — dark flag; explicit opt-in + disclosure + the Mechanic 1 reminder/cancel; idempotent grant; the $0 setup vault charges nothing now; the only auto-charge is the $99 after the free month via the audited cron. The free month is ~$0 marginal cost.

## Done
A studio.com/drglen joiner can claim a free first month of live group coaching (card vaulted, $99/mo auto-continue after), shipped dark behind `STUDIO_BRIDGE_ENABLED`.
