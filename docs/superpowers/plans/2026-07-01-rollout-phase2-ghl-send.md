# Rollout Phase 2 — Full-Roster Staged GHL Send (SP0 recipient list + SP4)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Checkbox steps track progress.

**Goal:** Email a personal portal link to the entire known roster, throttled via GHL for deliverability, at a comfortable pace — starting after today's attendee wave.

**Prereqs already live:** SP2 combined RM+E4L TOS (PR #446) is deployed, so any link we send lands on a gate that covers E4L. Portal mint (`/admin/portal/upsert`) exists and is idempotent for NEW portals.

## Global Constraints
- Never a Gmail burst — the large send goes through GHL (Glen's decision).
- Idempotent + windowed: a `portal_send_log` prevents double-emailing; re-runs are safe.
- Throttle: small daily waves to protect sender reputation (recommend 150/day; tunable).
- Link retrieval must be idempotent: do NOT rotate an existing client's token on every run.

## The recipient roster (SP0 recipient-list) — FINAL, tiered
Assembled locally from the real **Practice Better export** (174 active clients) ∪
FileMaker CSV ∪ e4l.db, deduped by lowercased email →
`scratchpad/rollout-roster-tiered-v2.csv` (`tier,tier_label,name,email,source,confidence`,
tier-sorted). This is the send list. (Populating the prod `people` hub is a SEPARATE
concern — NOT required to send — and is deferred.)

**Send tiers (final), total 2,986:**

| Order | Tier | Label | Count | Cumulative |
|---|---|---|---|---|
| 1 | 1 | pb_member (real PB accounts, all active; +41 PB-only added) | 174 | 174 |
| 2 | 2 | e4l_no_scan | 125 | 299 |
| 3 | 3 | e4l_with_scan | 238 | 537 |
| 4 | 4 | other (FileMaker historical, not PB/E4L) | 2,449 | 2,986 |

At 150/day: Tier 1 in ~1–2 days, E4L by ~day 4, then the long Tier-4 tail (~16 days).
Tier 4 is the oldest/coldest — highest bounce/spam risk; sent last, and the wave-by-wave
sender lets Glen halt before Tier 4 if earlier waves show trouble.

## Design note — push (pre-mint + link) vs pull (request-a-portal button)
Two delivery models; the right one differs by tier:
- **Push** (email a working portal link; portal pre-minted): frictionless, highest
  conversion. Best for **warm** tiers who know Glen — **Tier 1 PB** and **Tiers 2/3 E4L**.
- **Pull** (email a soft "Request your portal" CTA → mint on click): only mints for the
  interested (no orphan portals), softer optics, better deliverability, explicit opt-in.
  Best for the **cold Tier-4** tail (2,449 old FileMaker contacts).
- **Recommended: hybrid** — push Tiers 1–3 (~537), pull Tier 4 (2,449).
- Pull needs a small build: a GHL email button → public `/portal/claim?t=<signed-email>`
  endpoint that verifies the signed token, mints via `ensure_token`, and redirects into
  `/portal/<token>` (TOS gate). Mints only on click. (Not yet built.)

## Blocking inputs needed from Glen (the only things stopping the send)
1. **GHL custom field** for the portal URL — Glen creates a contact custom field
   (e.g. "Portal URL") in GHL and gives me its **field id/key**.
2. **GHL workflow** — trigger on tag `portal-invite` (or enrollment), action = send
   an email containing the portal link (`{{contact.<portal_url_field>}}`). Glen
   creates it and gives me the **workflow id**.
   (The app already has `ghl_upsert_contact(..., custom_fields=...)` and
   `ghl_enroll_workflow(contact_id)`; see app.py ~8092/8205.)

## Mechanism (per recipient)
1. **Get-or-create link** (idempotent) → the client's working portal URL.
2. `ghl_upsert_contact(email, first, last, custom_fields={PORTAL_URL_FIELD: url}, extra_tags=["portal-invite"])` → contact_id.
3. `ghl_enroll_workflow(contact_id, PORTAL_INVITE_WORKFLOW)` → GHL emails the link.
4. Record `portal_send_log(email, cohort, sent_at)`.

---

### Task 1: Idempotent get-or-create portal link endpoint  [buildable now]

**Why:** `/admin/portal/upsert` returns `token=None` for an EXISTING portal (no url), and `/admin/portal/reissue-link` ROTATES the token (breaks any prior link) — neither is safe to call repeatedly in a throttled rollout. We need "give me this email's current working link, minting one only if absent."

**Files:** Modify `app.py` (new route near the other portal admin routes ~12914); the raw token is already cached in `portal_notify_state.portal_token`.

**Interface:** `POST /admin/portal/get-or-create-link {email}` (console-key gated) →
`{ok, email, url, created: bool}`. Returns the cached link if present; else mints via `client_portal.upsert_portal` and returns the fresh link. Never rotates.

- [ ] Step 1: Write failing test — existing portal returns its cached link, `created:false`, and a second call returns the SAME url (idempotent). New email mints, `created:true`.
- [ ] Step 2: Run it, confirm fail.
- [ ] Step 3: Implement the route: look up `portal_notify_state.portal_token` for the email; if present, return `PUBLIC_BASE_URL + /portal/ + token`; else `upsert_portal` and cache the token in `portal_notify_state`, return the url.
- [ ] Step 4: Run tests green.
- [ ] Step 5: Commit.

### Task 2: GHL staged rollout sender  [needs the 2 GHL ids to RUN; codeable now]

**Files:** Create `scripts/portal_ghl_rollout.py`; Test `tests/test_portal_ghl_rollout.py`.

**Interface:** `next_wave(rows, sent_emails, wave_size)` (pure) → the next N unsent recipients. CLI: `--roster rollout-roster.csv --wave-size 150 --dry-run|--send`; env `PORTAL_URL_FIELD`, `PORTAL_INVITE_WORKFLOW`, `CONSOLE_SECRET`, `PUBLIC_BASE_URL`, GHL creds.

- [ ] Step 1: Write failing test for `next_wave` — excludes already-sent emails, caps at wave_size, skips blank/low-confidence.
- [ ] Step 2: Run, confirm fail.
- [ ] Step 3: Implement `next_wave` + a `send_one(email,name)` that calls get-or-create-link → `ghl_upsert_contact(custom_fields)` → `ghl_enroll_workflow`; append to `portal_send_log`; continue-on-error.
- [ ] Step 4: Tests green.
- [ ] Step 5: Commit.

### Task 3: Run the waves (ops, after Glen supplies the GHL ids)
- [ ] Dry-run wave 1 → eyeball the 150 recipients.
- [ ] `--send` wave 1; verify a couple land in GHL enrolled + emailed.
- [ ] Repeat daily (or on a cron) until the roster is drained; `portal_send_log` guarantees no repeats.

## Deferred
Populate the `people` hub from the roster; GHL-tagged PB-account tranche (SP0-v2);
SP3 E4L onboarding-on-first-access.

## Self-review
- SP0 recipient-list ⇒ roster CSV (built). SP4 ⇒ Tasks 1–3. Blocking inputs named explicitly (GHL field + workflow id). Idempotency covered by get-or-create-link (Task 1) + portal_send_log (Task 2). No placeholders in the buildable tasks; Task 2 run-step is gated on the two GHL ids, called out up front.
