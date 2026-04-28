#!/usr/bin/env python3
"""Daily Personal email cron entry point. Run on Render's cron worker."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incentive_engine import run_daily_send_for_beta_cohort


if __name__ == "__main__":
    n = run_daily_send_for_beta_cohort()
    print(f"Personal email cron: sent {n} email(s)")
