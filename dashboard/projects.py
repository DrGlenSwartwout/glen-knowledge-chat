"""Projects kanban — reads a server-side cached snapshot of
00 System/PROJECTS.md, uploaded via /api/projects/upload by the writer Mac's
hourly cron (mirrors the brain.py upload pattern).

Pure functions; routes are wired in app.py.

Environment variables:
    PROJECTS_SNAPSHOT_PATH — server-side path for the cached PROJECTS.md
                              (default: $DATA_DIR/projects.md, i.e. the persistent
                              disk; falls back to /tmp only when DATA_DIR is unset)
"""

import json
import os
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone


# Default to the persistent disk (DATA_DIR=/data on Render) so the cached snapshot
# + pending queue survive deploys/restarts. Only an explicit env var or a missing
# DATA_DIR (local dev) falls back to /tmp. Previously the default was /tmp, so
# every deploy wiped the kanban until the next 10-min sync.
_DATA_DIR = os.environ.get("DATA_DIR") or "/tmp"
SNAPSHOT = Path(os.environ.get("PROJECTS_SNAPSHOT_PATH") or os.path.join(_DATA_DIR, "projects.md"))
# Page-submitted ideas awaiting fold-in to PROJECTS.md by the Mac sync job.
PENDING = Path(os.environ.get("PROJECTS_PENDING_PATH") or os.path.join(_DATA_DIR, "pending_project_ideas.json"))


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


_VALID_SECTIONS = ("in_process", "queued", "planning", "ideas", "completed")


def _normalize_effort(raw: str) -> str:
    """Accept S/M/L/XL, 1–5, '3 stars', '★★★' — return canonical star string."""
    s = (raw or "").strip()
    if not s:
        return s
    legacy = {"S": "★★", "M": "★★★", "L": "★★★★", "XL": "★★★★★"}
    if s.upper() in legacy:
        return legacy[s.upper()]
    # Already stars
    stars = s.count("★")
    if stars and stars <= 5 and set(s) <= {"★", "☆", " "}:
        return "★" * stars
    # Numeric: "3", "4 stars", "5★"
    m = re.match(r"^\s*([1-5])\b", s)
    if m:
        return "★" * int(m.group(1))
    return s  # let it through; sync layer will write it verbatim


def _enqueue(edit: dict) -> dict:
    """Validate + queue any typed edit (add_idea / move / set / drop)."""
    t = edit.get("type")
    if t == "add_idea":
        text = (edit.get("text") or "").strip()
        if not text: raise ValueError("empty idea")
        if len(text) > 500: raise ValueError("idea too long (500 char max)")
        item = {"type": "add_idea", "text": text}
    elif t == "move":
        name = (edit.get("name") or "").strip()
        target = (edit.get("target") or "").strip().lower()
        if not name: raise ValueError("missing name")
        if target not in _VALID_SECTIONS:
            raise ValueError(f"invalid target: {target!r} (use one of {_VALID_SECTIONS})")
        item = {"type": "move", "name": name, "target": target}
    elif t == "set":
        name = (edit.get("name") or "").strip()
        field = (edit.get("field") or "").strip().lower()
        value = str(edit.get("value") or "").strip()
        if not (name and field and value):
            raise ValueError("missing name / field / value")
        if field == "effort":
            value = _normalize_effort(value)
        item = {"type": "set", "name": name, "field": field, "value": value}
    elif t == "drop":
        name = (edit.get("name") or "").strip()
        if not name: raise ValueError("missing name")
        item = {"type": "drop", "name": name}
    else:
        raise ValueError(f"unknown edit type: {t!r}")
    item["id"] = uuid.uuid4().hex[:12]
    item["created_at"] = datetime.now(timezone.utc).isoformat()
    items = _read_pending()
    items.append(item)
    _write_pending(items)
    return {"queued": True, "id": item["id"], "type": item["type"]}


def add_pending_idea(text: str) -> dict:
    """Back-compat shim — queues an add_idea edit."""
    return _enqueue({"type": "add_idea", "text": text})


def add_pending_edit(edit: dict) -> dict:
    """Queue a typed edit (add_idea / move / set / drop)."""
    return _enqueue(edit)


def pending_ideas() -> list:
    """Back-compat: returns add_idea-typed items in the legacy shape."""
    return [{"id": it["id"], "text": it.get("text", ""),
             "created_at": it.get("created_at", "")}
            for it in _read_pending() if it.get("type", "add_idea") == "add_idea"]


def pending_edits() -> list:
    """Returns all pending edits (typed). Legacy untyped items show as add_idea."""
    items = _read_pending()
    for it in items:
        it.setdefault("type", "add_idea")
    return items


def clear_pending_ideas(ids: list) -> dict:
    items = _read_pending()
    drop = set(ids or [])
    keep = [it for it in items if it.get("id") not in drop]
    _write_pending(keep)
    return {"cleared": len(items) - len(keep), "remaining": len(keep)}


# Alias for clarity — same op, used by the Mac sync after applying any edit
clear_pending_edits = clear_pending_ideas


def _find_entry(payload: dict, name: str):
    """Case-insensitive substring match. Returns (section, entry) or (None, None)."""
    needle = (name or "").strip().lower()
    if not needle:
        return None, None
    # prefer exact name match first
    for sec, entries in payload["sections"].items():
        for e in entries:
            if e["name"].lower() == needle:
                return sec, e
    for sec, entries in payload["sections"].items():
        for e in entries:
            if needle in e["name"].lower():
                return sec, e
    return None, None


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
    """Apply all pending edits to the kanban payload so the page reflects them
    immediately (Mac sync folds them into PROJECTS.md within ~10 min). Edited
    entries get `pending: true` for a dashed visual treatment."""
    for edit in _read_pending():
        t = edit.get("type", "add_idea")
        if t == "add_idea":
            payload["sections"]["ideas"].append({
                "name": edit.get("text", ""),
                "description": "",
                "fields": {},
                "pending": True,
            })
        elif t == "move":
            sec, entry = _find_entry(payload, edit.get("name", ""))
            target = edit.get("target")
            if entry and target in payload["sections"] and target != sec:
                payload["sections"][sec].remove(entry)
                entry["pending"] = True
                payload["sections"][target].append(entry)
        elif t == "set":
            _, entry = _find_entry(payload, edit.get("name", ""))
            if entry:
                entry["fields"][edit.get("field", "")] = edit.get("value", "")
                entry["pending"] = True
        elif t == "drop":
            sec, entry = _find_entry(payload, edit.get("name", ""))
            if entry and sec:
                payload["sections"][sec].remove(entry)
    # recompute counts after mutations
    for k, v in payload["sections"].items():
        payload["counts"][k] = len(v)
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
