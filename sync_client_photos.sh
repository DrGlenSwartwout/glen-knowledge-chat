#!/bin/bash
# On-demand: bulk-sync FMP-exported client photos (folder arg, default ~/Desktop/fmp-photos).
# DRY=1 previews without writing.  Usage: bash ~/deploy-chat/sync_client_photos.sh [folder]
doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" PYTHONPATH="$HOME/deploy-chat" python3 "$HOME/deploy-chat/scripts/sync_client_photos.py" "$@"
