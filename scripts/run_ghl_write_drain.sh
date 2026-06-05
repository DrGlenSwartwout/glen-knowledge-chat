#!/bin/bash
# launchd wrapper for the GHL write-queue drain (sync-ghl-writes.py).
# launchd runs with a minimal PATH that excludes /opt/homebrew/bin, so doppler
# and python3 won't resolve unless we export PATH first.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

cd /Users/remedymatch/deploy-chat || exit 1

# Drains the GHL write-queue (CRM tag/note/opportunity/workflow) to GHL from the
# Mac's residential IP (Render's AWS IP is WAF-blocked). The server marks each
# item done/failed, so re-runs are safe and a usually-empty queue is a cheap no-op.
exec doppler run -p remedy-match -c prd -- \
  python3 sync-ghl-writes.py \
  >> /Users/remedymatch/Library/Logs/ghl-write-drain.log 2>&1
