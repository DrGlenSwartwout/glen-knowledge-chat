"""Projects kanban — reads a server-side cached snapshot of
00 System/PROJECTS.md, uploaded via /api/projects/upload by the writer Mac's
hourly cron (mirrors the brain.py upload pattern).

Pure functions; routes are wired in app.py.

Environment variables:
    PROJECTS_SNAPSHOT_PATH — server-side path for the cached PROJECTS.md
                              (default: /tmp/projects.md)
"""

import os
import re
from pathlib import Path
from datetime import datetime, timezone


SNAPSHOT = Path(os.environ.get("PROJECTS_SNAPSHOT_PATH", "/tmp/projects.md"))


# ─── Read / write ──────────────────────────────────────────────────────────

def write_projects(payload_bytes: bytes) -> dict:
    """Persist uploaded PROJECTS.md. Validates UTF-8 + non-empty + has at least one section header."""
    text = payload_bytes.decode("utf-8")
    if not text.strip():
        raise ValueError("empty payload")
    if "## In Process" not in text and "## Ideas" not in text and "## Completed" not in text:
        raise ValueError("payload doesn't look like PROJECTS.md (no recognized section headers)")
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(text)
    return {
        "saved": True,
        "path": str(SNAPSHOT),
        "bytes": len(text),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }


def read_projects_md():
    """Read the cached snapshot. Returns None if not yet uploaded."""
    if not SNAPSHOT.exists():
        return None
    return SNAPSHOT.read_text()


# ─── Parser ────────────────────────────────────────────────────────────────

_SECTION_MAP = {
    "in process": "in_process",
    "ideas":      "ideas",
    "completed":  "completed",
}

_H2_RE    = re.compile(r"^##\s+(.+?)\s*$")
_ENTRY_RE = re.compile(r"^-\s+\*\*(.+?)\*\*\s*[—–-]?\s*(.*)$")
_FIELD_RE = re.compile(r"^\s+-\s+\*([^*]+?)\*:\s*(.+)$")


def parse_sections(md: str) -> dict:
    """Parse PROJECTS.md → kanban-shaped dict.

    Returns {sections: {in_process, ideas, completed}, counts}.
    Each entry: {name, description, fields: {status, where, eta, blockers, sessions, ...}}
    """
    sections = {"in_process": [], "ideas": [], "completed": []}
    current_section = None
    current_entry = None

    def flush():
        nonlocal current_entry
        if current_entry and current_section:
            sections[current_section].append(current_entry)
        current_entry = None

    for line in md.splitlines():
        h2 = _H2_RE.match(line)
        if h2:
            flush()
            current_section = _SECTION_MAP.get(h2.group(1).strip().lower())
            continue
        if current_section is None:
            continue

        entry = _ENTRY_RE.match(line)
        if entry:
            flush()
            current_entry = {
                "name": entry.group(1).strip(),
                "description": entry.group(2).strip(),
                "fields": {},
            }
            continue

        if current_entry is None:
            continue

        field = _FIELD_RE.match(line)
        if field:
            current_entry["fields"][field.group(1).strip().lower()] = field.group(2).strip()

    flush()
    return {
        "sections": sections,
        "counts": {k: len(v) for k, v in sections.items()},
    }


def kanban_payload() -> dict:
    """Top-level entry point used by the /api/projects route.

    If no snapshot yet, returns an empty board with a friendly meta note so
    the page renders cleanly until the first cron push lands.
    """
    md = read_projects_md()
    if md is None:
        return {
            "sections": {"in_process": [], "ideas": [], "completed": []},
            "counts": {"in_process": 0, "ideas": 0, "completed": 0},
            "source": {"status": "no snapshot uploaded yet — run console-push to populate"},
        }
    payload = parse_sections(md)
    try:
        st = SNAPSHOT.stat()
        payload["source"] = {
            "path": str(SNAPSHOT),
            "uploaded_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "bytes": st.st_size,
        }
    except Exception:
        payload["source"] = {"path": str(SNAPSHOT)}
    return payload
