"""System health — per-integration status grid."""

import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from .cache import last_success

LOG_DB = str(Path(__file__).parent.parent / "chat_log.db")


def _check_db():
    try:
        with sqlite3.connect(LOG_DB) as cx:
            cx.execute("SELECT 1").fetchone()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _env_present(*keys):
    return all(os.environ.get(k) for k in keys)


def status_grid():
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "systems": {
            "sqlite_chat_log": _check_db(),
            "practice_better": {"configured": _env_present("PRACTICE_BETTER_CLIENT_ID",
                                                            "PRACTICE_BETTER_CLIENT_SECRET"),
                                "last_success": last_success("money.pb")},
            "authorize_net":   {"configured": _env_present("AUTHNET_API_LOGIN_ID",
                                                            "AUTHNET_TRANSACTION_KEY"),
                                "last_success": last_success("money.an")},
            "wise":            {"configured": _env_present("WISE_API_TOKEN"),
                                "last_success": last_success("money.wise")},
            "quickbooks":      {"configured": _env_present("QUICKBOOKS_PROD_CLIENT_ID",
                                                            "QUICKBOOKS_PROD_REFRESH_TOKEN"),
                                "last_success": last_success("money.qb_banks")},
            "ghl":             {"configured": _env_present("GHL_API_KEY", "GHL_LOCATION_ID"),
                                "last_success": last_success("ghl.pipelines")},
            "heygen":          {"configured": _env_present("HEYGEN_API_KEY"),
                                "last_success": last_success("heygen.recent")},
            "facebook_ads":    {"configured": _env_present("META_ACCESS_TOKEN"),
                                "last_success": last_success("facebook.boulder")},
            "pinecone":        {"configured": _env_present("PINECONE_API_KEY"),
                                "last_success": last_success("pinecone.stats")},
        }
    }
