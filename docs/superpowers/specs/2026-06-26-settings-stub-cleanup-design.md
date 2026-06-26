# Settings Stub Cleanup (Sub-project B4) — Design Spec

**Date:** 2026-06-26
**Status:** Approved + executed inline (small change). Final increment of sub-project B (console
consolidation). B1 (Money), B2 (Pages), B3 (Approvals) shipped.

## Goal

Remove the eight "Coming soon" stub sections from the console Settings page so it shows only working
controls, and preserve their roadmap descriptions in a doc.

## Change

- **`static/console-settings.html`** — delete the 8 stub `<details class="section">` blocks
  (Maintenance Mode, Voice Routing, Notifications, Quiet Hours, AI Model Defaults, Cron Schedule, Sync
  Health, Environment), each of which was only a description paragraph with a "Coming soon" badge and
  no controls. Keep the 2 **Live** sections (Shipping, Active Write Mac). Drop the now-unused
  `.badge.soon` CSS rule. Leave a one-line HTML comment pointing to the roadmap doc.
- **`docs/console-settings-roadmap.md`** (new) — the 8 feature descriptions, lifted verbatim, kept as
  a roadmap.

## Out of scope

- Building any of the 8 planned features.
- The Settings sub-row from sub-project A (Pricing/Shipping-config/Tax/Write-Mac), the 2 Live
  controls, op-nav, and all backends — untouched.

## Testing

Headless render of `/console/settings`: only the 2 Live sections render, no "Coming soon" badge is
visible, two `.badge.live` present, zero JS console/page errors. `docs/console-settings-roadmap.md`
lists all 8 planned features. (Verified: sections = [Shipping, Active Write Mac], comingSoon=False,
2 live badges, 0 JS errors.)

## Rollout

A deletion + a doc. No backend/data change, no new route, no nav change.
