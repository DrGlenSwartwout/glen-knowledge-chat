# Portal Hub Shell — Rollout & Verification Note

**Date:** 2026-07-22
**Branch:** `sess/c9f1462d` (off `main` at `90b20316`)
**Plan:** `docs/superpowers/plans/2026-07-22-portal-hub-shell.md`
**Spec:** `docs/superpowers/specs/2026-07-22-portal-hub-ia-design.md`
**Mockup:** https://claude.ai/code/artifact/b225762d-a68d-4eee-b989-a90f48d01871

## What shipped on this branch (Phase 0 + 1)

The client portal (`static/client-portal.html`, driven by `render(d, v)`) gains a
journey-grouped **hub** landing, gated behind a new flag and OFF by default. When
`PORTAL_HUB_ENABLED` is on, the portal becomes a drill-in hub: a status banner +
tile grid is the default view (`data-panel="hub"`); each tile drills into its own
panel with a "Back to hub" control. When the flag is off, the portal renders
byte-identically to before (verified per task against the untouched render path).

Panels wired: `hub` (default), `current` (My Analysis), `history`, `orders`,
`ask`, `finder`, `bodymap`, `classes`, `account`, `refer`, `referrals`. The old
`#portalTabs` tab bar is suppressed whenever the hub is on (no double-nav).

Commits (oldest first): `72d4b307` finder characterization + finder-on staged ·
`e1a1edaa` PORTAL_HUB_ENABLED flag + payload · `977fa461` banner + tile grid ·
`b23e9ad2` theme-aware CSS + showTab unknown-panel fallback · `9d9279d5`
current/history/orders panels + default panel · `afcb7766` drill-in hub +
Back-to-hub · `a60f3638` seven secondary panels + showTab hub fallback. (Plus the
Task-1 fix `cf7c569e` restoring city-fallback test coverage.)

## The two feature flags

| Flag | State | Notes |
|---|---|---|
| `PORTAL_FINDER_ENABLED` | **STAGED (not flipped)** | Turns the practitioner-finder card on for all portals. Prod set command staged for Glen: `doppler set PORTAL_FINDER_ENABLED=on -c prd` (prod secret write is classifier-blocked for automation). Independent of the hub — can flip anytime. |
| `PORTAL_HUB_ENABLED` | **OFF (ships dark)** | Turns the whole hub landing on. Same truthy set `("1","true","yes","on")`. Reversible by unsetting. Do NOT flip until the render-verify checklist below passes. |

Both are plain env flips, fully reversible. No data migration; nothing about a
client's records changes.

## Deploy caution

Every portal deploy 502s illtowell for ~4-8 minutes. Batch changes and warn
before deploying. Merging this branch self-ships (autoDeploy on); a flag flip is a
second, separate deploy.

## Consolidated render-verification (RUN before flipping `PORTAL_HUB_ENABLED` on)

Per-task gates were server-side pytest + a server-free Node parse/logic check; no
live browser render was run in the worktree. Before enabling the hub in any
environment, drive a real portal URL headless (webapp-testing) with the flag on
and confirm:

- [ ] Flag OFF on an existing portal → looks exactly as today (no `.portal-hub`).
- [ ] Flag ON → the hub landing shows: status banner + tile grid; no old tab bar.
- [ ] Tap **My Analysis** → analysis content shows, with a working "Back to hub".
- [ ] Tap **Scan History** / **Orders & Invoices** → correct panel, Back to hub works.
- [ ] Tap **Find a Practitioner** → finder panel (embed card) if finder enabled, else the "not enabled yet" message.
- [ ] Tap **Ask Dr. Glen** → chat panel; confirm the chat input actually works (init found `#chatCard`/`#chatMsgs` inside the now-hidden-until-shown panel).
- [ ] Tap **Body Map**, **MasterClasses**, **Account** → each shows its panel + Back to hub.
- [ ] Tap **Refer a Friend** (ambassador links / become-ambassador) and **Referrals** (activity list) → both panels render; ambassador form + share-page controls still work.
- [ ] Tap **Account** (top-bar link in the hub) → account panel shows + Back to hub.
- [ ] **No duplicate scan-history / receipts** in the My Analysis (current) panel when the hub is on (see deferred item below — this is the known gap to watch).
- [ ] Both flags on together → hub grid present, NO old tab bar (no double-nav).
- [ ] Dark mode: tiles, banner, and the `.next` chip are all readable.
- [ ] Runtime (not just parse): confirm the nested `${_hub ? \`…panels…\` : ""}` actually renders real DOM in a browser, and that the Ask-Dr.-Glen chat + ambassador/share-page form handlers bind correctly even though their panels start `hidden`.

## Deferred (documented, not lost)

- **NOT DELIVERED from Phase-1's plan: "remove the two duplicate cards."** This
  step was intentionally deferred (see below), so the scan-history teaser and the
  History-&-receipts card still render under the hub. Do not mark that plan step
  complete. It is the one gap between the plan's Phase-1 text and what shipped, and
  it is safe (it protects OFF-state byte-identity and the request-analysis path).
- **Legacy duplicate-card suppression + request-analysis handling.** The inline
  "Scan history" block carries the request-analysis action; the legacy
  history/receipts and unpaid-invoice inline cards render only when
  `!scan_history_enabled`. Under hub + scan-history-off (a rare combo) these can
  duplicate the History/Orders panels. Left untouched deliberately to avoid
  risking the request-analysis path. Follow-up: gate those inline blocks on
  `!(scan_history_enabled || hub_enabled)` AND confirm the Scan History panel
  surfaces request-analysis before removing anything.
- **Full card migration for My Analysis vs My Remedies** and the genuinely new
  pages (My Health Profile, My Healing Oasis, My Remedies external list,
  Ambassador dashboard data) are Phases 2-4 — their own specs/plans. Reuse note:
  the "request a review" mechanic for My Remedies' external list already exists
  (`REVIEWS_ENABLED` + `dashboard/supplement_reviews.py`).

## Minor polish items (from reviews, non-blocking)

- Recent-referrals list appears in both the `refer` and `referrals` panels (they
  are mutually hidden; only invite links were required non-duplicated). Optional
  later cleanup.
- The `refer` panel's `|| fallback` string is unreachable (share-page card is
  always present under hub). Harmless.
- The `.hub-banner .next` chip uses a fixed gold background; readable in dark but
  brighter than the themed surfaces. Optional.
- Under the hub, the **notification-preferences and scan-preferences** cards still
  land in the My Analysis (current) panel rather than the Account panel (only the
  account snapshot was migrated). Move them into `data-panel="account"` in a
  follow-up so Account holds all settings, per the spec.
