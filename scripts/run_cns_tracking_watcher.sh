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
#
# env -u DATA_DIR: the harvest→onboard path imports app.py, whose module init
# opens $DATA_DIR/chat_log.db. Doppler's prd config injects DATA_DIR=/data (the
# Render disk), which doesn't exist on this Mac, so the import would crash. app.py
# ALSO treats a set DATA_DIR as "I'm on Render" and starts its background cron
# scheduler (hourly pushes, image queues, etc.) — which must NOT run from this
# watcher. UNSETTING DATA_DIR after doppler resolves fixes both: LOG_DB falls back
# to app.py's own dir (this checkout's chat_log.db, which exists) AND the
# scheduler stays off. The watcher's own DB is still set explicitly via --db.
exec doppler run -p remedy-match -c prd -- \
  env -u DATA_DIR \
  python3 cns_tracking_watcher.py --live --auto-send --days 1 \
    --db /Users/remedymatch/deploy-chat/chat_log.db \
  >> /Users/remedymatch/Library/Logs/cns-tracking-watcher.log 2>&1
