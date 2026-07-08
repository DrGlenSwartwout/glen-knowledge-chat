# Automated client workflow (one place, Rae-simple) — design (teed up 2026-07-08)

**Status:** spec ready to build. Origin (Glen, 2026-07-08): "automate this process to simplify it and control it from one place for Rae to be able to do it smoothly." **Extends** the To-Do board spec (#708) from *status + deep-links-to-manual-tools* into *one-click automation of each step*.

## Why (pain points this session exposed — the requirements are these bugs)

Getting one client (Dana, then Bobbi) from scan → live portal took an expert an hour of app-switching and hit every sharp edge. The automation must remove each:
1. **Two apps** — local Intake `:8011` (author/invoice) + prod composer (publish). Constant landing on the wrong one.
2. **JSON seed-pasting** — building a portal meant hand-generating a `portal-seed.json`, opening it in TextEdit, copy-pasting into the composer. Non-starter for Rae.
3. **Wrong-source content** — the composer loaded a **stale reveal** (old remedies, even a deprecated SKU) instead of the client's authored analysis. Silent and dangerous.
4. **Seed-format bugs** — nested `remedy` object → "[object Object]" on the portal.
5. **Flaky publish/email** — "Publish failed," then a save-but-no-email ("Updated" with `notified_at` empty).
6. **No single status** — nothing showed what was done vs next.

## Goal

**One control surface where Rae runs a client end-to-end with one button per step, each auto-doing the work with a review-and-confirm — no app-switching, no files, no JSON.**

## The surface

The To-Do board (#708) on `:8011`, made **actionable**: each client card shows the ordered steps with live status (auto-checked), and each not-done step is a **button that performs the action**, not a link to a manual tool.

## The automated steps (each = one button, auto-work + review)

1. **Pull scan** — trigger the scan-pull for the client (no email-trigger wait). Auto.
2. **Draft reveal** — run synthesis → stage a reveal draft. Auto (the pipeline already exists: `e4l_synthesis` → `build_payload`).
3. **Approve reveal** — one click: un-blur top remedy + email the reveal link (reuse `biofield_reveal.approve`+`.send`).
4. **Build & publish portal** — **the big one.** One click that:
   - builds the portal **from the client's authored analysis if one exists (`biofield_auth_chain`), else the approved reveal** — never a stale draft (fixes pain #3);
   - generates the seed **server-side in the correct flat format** (fixes #2, #4) — no human JSON;
   - publishes **and** emails, with the send verified (fixes #5);
   - shows a preview + a "looks right? confirm" gate before it goes live.
5. **Raise invoice** — one click: create the order from the authored remedies (already built: `/author/<id>/invoice`), optionally publish it to the portal to pay.
6. **Done** — the card shows all green + the live links.

## Rae-simplicity requirements

- Plain-language step labels ("Send Dana her reading", "Put her invoice on her portal"), not system terms.
- One primary action per step; the previous step must be green before the next is offered (guided, not a wall of buttons).
- Every write is **review-then-confirm** (preview the reveal / portal / invoice before it sends), and **reversible or re-runnable** (re-publish keeps the same link).
- Zero app-switching, zero files, zero keys typed. The board carries the client and the console key.

## Build order (slices — do NOT build as one blob)

- **Slice 1:** the actionable board (steps + live status), reusing #708's `workflow_state(email)`.
- **Slice 2:** server-side **portal auto-build+publish** button (steps 4) — the highest-value fix (kills the seed-paste + wrong-content + flaky-publish path). Includes a `build_portal_seed_from_authored(test_id)` helper (the correct flat format) + a verified publish+email.
- **Slice 3:** the reveal + invoice buttons (steps 1–3, 5), each wrapping an existing endpoint.

## Open decisions (resolve at build)

1. **Source precedence for the portal** — authored chain > approved reveal > drafted reveal. Confirm the order.
2. **How much auto vs confirm** — recommend every *client-facing send* (reveal email, portal publish+email, invoice) is behind a one-click **confirm**, everything upstream (pull, synth, draft) auto-runs. Confirm.
3. **Who can run it** — Rae's scoped console access (already exists) vs owner-only. Recommend Rae-scoped (that's the whole point).
4. **Board home** — `:8011/todo` (operator Mac) is fine for Glen but Rae may work elsewhere; may need a prod-hosted version. Flag: if Rae isn't on the Mac, the localhost-only pieces (#707 wrinkle) block her — decide whether the board + actions must be **prod-hosted** so Rae can use them from anywhere.

## Out of scope (v1)

- Fully hands-off (no human confirm) publishing.
- Bulk/multi-client runs.
- Rebuilding the underlying reveal/portal engines — this orchestrates the existing ones behind one button each.

## Related

Supersedes #707 (cross-app nav) and subsumes #708 (To-Do board) — build this as the umbrella; #708's `workflow_state` is Slice 1.
