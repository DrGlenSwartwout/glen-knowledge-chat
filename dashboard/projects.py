"""Projects kanban — reads a server-side cached snapshot of
00 System/PROJECTS.md, uploaded via /api/projects/upload by the writer Mac's
hourly cron (mirrors the brain.py upload pattern).

Pure functions; routes are wired in app.py.

Environment variables:
    PROJECTS_SNAPSHOT_PATH — server-side path for the cached PROJECTS.md
                              (default: /tmp/projects.md)
"""

import json
import os
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone


SNAPSHOT = Path(os.environ.get("PROJECTS_SNAPSHOT_PATH", "/tmp/projects.md"))
# Page-submitted ideas awaiting fold-in to PROJECTS.md by the Mac sync job.
PENDING = Path(os.environ.get("PROJECTS_PENDING_PATH",
                              "/tmp/pending_project_ideas.json"))


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


# ─── Pending ideas (page-submitted, awaiting fold-in to PROJECTS.md) ─────────

def _read_pending() -> list:
    if not PENDING.exists():
        return []
    try:
        return json.loads(PENDING.read_text())
    except Exception:
        return []


def _write_pending(items: list) -> None:
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    PENDING.write_text(json.dumps(items, indent=2))


def add_pending_idea(text: str) -> dict:
    """Queue an idea submitted from the page. The Mac sync job folds it into
    PROJECTS.md's ## Ideas section, then clears it."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty idea")
    if len(text) > 500:
        raise ValueError("idea too long (500 character max)")
    items = _read_pending()
    items.append({
        "id": uuid.uuid4().hex[:12],
        "text": text,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    _write_pending(items)
    return {"added": True, "pending_count": len(items)}


def pending_ideas() -> list:
    return _read_pending()


def clear_pending_ideas(ids: list) -> dict:
    items = _read_pending()
    drop = set(ids or [])
    keep = [it for it in items if it.get("id") not in drop]
    _write_pending(keep)
    return {"cleared": len(items) - len(keep), "remaining": len(keep)}


# ─── Parser ────────────────────────────────────────────────────────────────

_SECTION_MAP = {
    "in process": "in_process",
    "queued":     "queued",
    "planning":   "planning",
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
    sections = {"in_process": [], "queued": [], "planning": [],
                "ideas": [], "completed": []}
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


def _merge_pending(payload: dict) -> dict:
    """Append page-submitted pending ideas to the ideas column so they show
    immediately, before the Mac sync job folds them into PROJECTS.md."""
    pend = _read_pending()
    for p in pend:
        payload["sections"]["ideas"].append({
            "name": p.get("text", ""),
            "description": "",
            "fields": {},
            "pending": True,
        })
    if pend:
        payload["counts"]["ideas"] = len(payload["sections"]["ideas"])
    return payload


def kanban_payload() -> dict:
    """Top-level entry point used by the /api/projects route.

    If no snapshot yet, returns an empty board with a friendly meta note so
    the page renders cleanly until the first cron push lands.
    """
    md = read_projects_md()
    if md is None:
        empty = ["in_process", "queued", "planning", "ideas", "completed"]
        return _merge_pending({
            "sections": {k: [] for k in empty},
            "counts": {k: 0 for k in empty},
            "source": {"status": "no snapshot uploaded yet — run console-push to populate"},
        })
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
    return _merge_pending(payload)
