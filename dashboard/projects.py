"""Projects kanban — reads 00 System/PROJECTS.md from the AI-Training vault on
GitHub raw and parses it into a JSON kanban board.

Pure functions; routes are wired in app.py (matches the convention used by
brain.py / settings.py / inbox.py / etc.).

Drop into ~/deploy-chat/dashboard/projects.py.

Environment variables (set on Render):
    GITHUB_TOKEN     — optional; only needed if AI-Training repo becomes private
    PROJECTS_REPO    — optional override (default: DrGlenSwartwout/AI-Training)
    PROJECTS_BRANCH  — optional override (default: main)
    PROJECTS_PATH    — optional override (default: 00 System/PROJECTS.md)
"""

import os
import re
import time

import requests


PROJECTS_REPO   = os.environ.get("PROJECTS_REPO",   "DrGlenSwartwout/AI-Training")
PROJECTS_BRANCH = os.environ.get("PROJECTS_BRANCH", "main")
PROJECTS_PATH   = os.environ.get("PROJECTS_PATH",   "00 System/PROJECTS.md")
GITHUB_TOKEN    = os.environ.get("GITHUB_TOKEN", "")

PROJECTS_URL = (
    f"https://raw.githubusercontent.com/{PROJECTS_REPO}/{PROJECTS_BRANCH}/"
    f"{requests.utils.quote(PROJECTS_PATH)}"
)

# Simple in-process TTL cache (5 min). Mirrors dashboard/cache.py pattern but
# kept inline here to avoid adding a dependency on its API surface.
_CACHE_TTL = 300
_cache = {"value": None, "fetched_at": 0.0}


def fetch_projects_md(force_refresh: bool = False) -> str:
    """Fetch PROJECTS.md from GitHub raw, with 5-min TTL cache."""
    now = time.time()
    if (
        not force_refresh
        and _cache["value"] is not None
        and (now - _cache["fetched_at"]) < _CACHE_TTL
    ):
        return _cache["value"]
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    r = requests.get(PROJECTS_URL, headers=headers, timeout=10)
    r.raise_for_status()
    _cache["value"] = r.text
    _cache["fetched_at"] = now
    return r.text


# ─── Parser ───────────────────────────────────────────────────────────────

_SECTION_MAP = {
    "in process": "in_process",
    "ideas":      "ideas",
    "completed":  "completed",
}

_H2_RE    = re.compile(r"^##\s+(.+?)\s*$")
_ENTRY_RE = re.compile(r"^-\s+\*\*(.+?)\*\*\s*[—–-]?\s*(.*)$")
_FIELD_RE = re.compile(r"^\s+-\s+\*([^*]+?)\*:\s*(.+)$")


def parse_sections(md: str) -> dict:
    """Parse PROJECTS.md → {sections: {in_process: [...], ideas: [...], completed: [...]}, counts: {...}, source: {...}}.

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
            key = h2.group(1).strip().lower()
            current_section = _SECTION_MAP.get(key)
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
        "source": {
            "repo":   PROJECTS_REPO,
            "branch": PROJECTS_BRANCH,
            "path":   PROJECTS_PATH,
        },
    }


def kanban_payload() -> dict:
    """Top-level entry point used by the /api/projects route."""
    return parse_sections(fetch_projects_md())
