"""Intelligence briefings — markdown reports written by the Mac runner, served to the dashboard.

5 briefing types:
- daily-briefing       — one-page morning summary
- eyes-on-every-street — anomaly detection across all systems
- founders-radar       — attention allocator (focus on / ignore lists)
- revenue-x-ray        — revenue-per-founder-hour analysis
- pattern-which-connects — cross-system pattern detection

Each briefing is uploaded as markdown by intelligence-runner.py and stored
under DATA_DIR/intelligence/{slug}.md. The dashboard renders the markdown
verbatim in the Intelligence row.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp")) / "intelligence"
DATA_DIR.mkdir(parents=True, exist_ok=True)

VALID_SLUGS = {
    "daily-briefing",
    "eyes-on-every-street",
    "founders-radar",
    "revenue-x-ray",
    "pattern-which-connects",
}


def _slug_path(slug):
    if slug not in VALID_SLUGS:
        raise ValueError(f"Unknown slug: {slug}. Valid: {sorted(VALID_SLUGS)}")
    return DATA_DIR / f"{slug}.md"


def read_briefing(slug):
    """Return the latest briefing markdown for slug, plus metadata."""
    p = _slug_path(slug)
    if not p.exists():
        return {
            "slug": slug,
            "empty": True,
            "message": f"No {slug} briefing yet. First run pending.",
        }
    stat = p.stat()
    return {
        "slug": slug,
        "empty": False,
        "markdown": p.read_text(),
        "generated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "bytes": stat.st_size,
    }


def write_briefing(slug, markdown_bytes):
    """Persist a briefing markdown payload uploaded by the Mac runner."""
    p = _slug_path(slug)
    text = markdown_bytes.decode("utf-8") if isinstance(markdown_bytes, bytes) else markdown_bytes
    p.write_text(text)
    return {"slug": slug, "saved": True, "path": str(p), "bytes": len(text)}


def list_all():
    """Index of all briefings with their freshness."""
    out = {}
    for slug in sorted(VALID_SLUGS):
        out[slug] = read_briefing(slug)
    return out
