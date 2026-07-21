"""Map a former-SQLite file path to a Postgres schema name (Option B)."""
import os
import re
from typing import Optional

def schema_for_path(db_path: str) -> str:
    if not db_path or db_path == ":memory:":
        return "public"
    base = os.path.basename(db_path)
    base = re.sub(r"\.(db|sqlite|sqlite3)$", "", base, flags=re.I)
    name = re.sub(r"[^a-z0-9_]+", "_", base.lower()).strip("_")
    return name or "public"
