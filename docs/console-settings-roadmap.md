# Console Settings — Roadmap

Planned Settings features, preserved when the "Coming soon" stub sections were removed from
`static/console-settings.html` (sub-project B4, 2026-06-26). These were never built — each was a
description placeholder. Kept here so the planning intent isn't lost.

**Live today** (still on the Settings page): **Shipping** (USPS flat-rate prices, bottle catalog,
box-fit matrix → `/admin/shipping`) and **Active Write Mac** (which Mac may run shared-state writers).
Plus the Settings sub-row from sub-project A: Pricing, Shipping-config, Tax, Write-Mac.

## Planned

- **Maintenance Mode** — Global toggle to pause all perpetual cron agents (rae-reconciliation,
  knowledge-base-hygienist, mentor-keeper, etc.) for travel, debugging, or holidays. Currently each
  agent has its own schedule; this would add a single off-switch.
- **Voice Routing** — Default voice for outbound content: Justus (the AI integrator) vs Glen (Glen's
  clone) vs per-context. Today this is decided per-task; surfacing it as a setting would let you flip
  it for a whole campaign.
- **Notifications** — Routing matrix: who (Glen / Rae / Shaira) gets pinged for which event type
  (E4L scan completed, payment received, support ticket, agent failure). Reduces inbox spam and
  clarifies ownership.
- **Quiet Hours** — Time window (default 21:00–07:00 HST) during which no agent sends notifications
  to Glen. Critical alerts can override.
- **AI Model Defaults** — Per-task model picker: Haiku for chatbot retrieval, Sonnet for follow-up
  drafts, Opus for synthesis/distinction work. Cost lever — change without redeploying.
- **Cron Schedule** — Read-only view (initially) of every perpetual agent's next-run time, last-run
  status, and last error. Editable in v2.
- **Sync Health** — One-click "Run sync test" — drops a timestamped file in `00 System/`, watches for
  round-trip on all 3 Macs, reports mesh health. Plus last-backup timestamps for PB / QB / Pinecone /
  Supabase.
- **Environment** — Doppler config selector (prd / dev / staging) — for safer toggling between live
  and test environments without editing shell config.
