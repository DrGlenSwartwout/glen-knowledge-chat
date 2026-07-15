#!/bin/bash
# On-demand: fulfill pending portal analysis-requests from local FMP data.
# Add DRY=1 before the command to preview without writing.
doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" PYTHONPATH="$HOME/deploy-chat" python3 "$HOME/deploy-chat/scripts/fulfill_requests.py"
