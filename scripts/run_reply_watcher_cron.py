#!/usr/bin/env python3
"""Reply watcher cron entry. Runs on Render every 15 minutes."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reply_watcher import process_inbox_replies


if __name__ == "__main__":
    counts = process_inbox_replies()
    print(
        f"Reply watcher: "
        f"processed={counts['processed']} "
        f"skipped_nonuser={counts['skipped_nonuser']} "
        f"errored={counts['errored']}"
    )
