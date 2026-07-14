#!/usr/bin/env bash
# Push the local e4l.db (catalog + PHI client findings) to prod over the owner-
# authenticated channel. The db-sync endpoint installs it at $DATA_DIR/e4l.db so the
# pattern glossary + scan-trends light up in prod. Run AFTER the db-sync endpoint is
# deployed. Usage:  bash scripts/push_e4l_to_prod.sh [path-to-e4l.db]
set -euo pipefail
DB="${1:-$HOME/AI-Training/e4l.db}"
BASE="${E4L_PROD_URL:-https://glen-knowledge-chat.onrender.com}"
[ -f "$DB" ] || { echo "e4l.db not found: $DB"; exit 1; }
KEY="$(doppler run -p remedy-match -c prd -- printenv CONSOLE_SECRET 2>/dev/null || true)"
[ -n "$KEY" ] || KEY="$(doppler run -p remedy-match -c prd -- printenv WEBHOOK_SECRET 2>/dev/null || true)"
[ -n "$KEY" ] || { echo "no console key in prd doppler (CONSOLE_SECRET/WEBHOOK_SECRET)"; exit 1; }
echo "pushing $(du -h "$DB" | cut -f1) e4l.db -> $BASE/api/console/e4l/db-sync"
curl -fsS -X POST "$BASE/api/console/e4l/db-sync" \
  -H "X-Console-Key: $KEY" -H "Content-Type: application/octet-stream" \
  --data-binary "@$DB" | python3 -m json.tool
