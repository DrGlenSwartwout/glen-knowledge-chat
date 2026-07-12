#!/bin/bash
# launchd wrapper for the CNS tracking watcher.
# launchd runs with a minimal PATH that excludes /opt/homebrew/bin, so doppler
# and python3 won't resolve unless we export PATH first.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

cd /Users/remedymatch/deploy-chat || exit 1

# --days 1: only process shipments from the last day (going-forward). The
# shipments table makes re-runs idempotent, so the 15-min cadence never repeats.
# --auto-send: email high-confidence (exact GHL match) + harvested recipients
# outright; fuzzy/unresolved recipients still land as drafts for Glen/Rae review.
exec doppler run -p remedy-match -c prd -- \
  python3 cns_tracking_watcher.py --live --auto-send --days 1 \
    --db /Users/remedymatch/deploy-chat/chat_log.db \
  >> /Users/remedymatch/Library/Logs/cns-tracking-watcher.log 2>&1
