"""Intelligence briefing runner.

Gathers live system stats from the existing dashboard modules, asks Claude to
compose each of the five briefings (daily-briefing, eyes-on-every-street,
founders-radar, revenue-x-ray, pattern-which-connects), and writes each
markdown payload to /data/intelligence/<slug>.md so the dashboard cards serve
fresh content.

Triggered by POST /cron/regenerate-briefings (which is curled by the Render
cron container). Lives in the web container so it has access to the persistent
disk, ANTHROPIC_API_KEY, and all the same internal stat functions the
dashboard already uses.
"""

import os
import json
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from . import intelligence as _intel
from . import money as _money
from . import ghl as _ghl
from . import pinecone_stats as _pc_stats
from . import scoreapp as _scoreapp
from . import heygen as _heygen


MODEL = os.environ.get("BRIEFING_MODEL", "claude-haiku-4-5-20251001")

VALID_SLUGS = [
    "daily-briefing",
    "eyes-on-every-street",
    "founders-radar",
    "revenue-x-ray",
    "pattern-which-connects",
]


SLUG_PROMPTS = {
    "daily-briefing": (
        "You write Glen Swartwout's one-page morning Daily Briefing. Voice: "
        "calm, scannable, lead with money + overdue client work, end with a "
        "single Bottom Line action. Section headers with emoji (💰 Money, "
        "🚨 Client Messages, 📋 Pipeline & Systems, then Bottom line). Bold "
        "names and amounts. No filler. Max ~250 words."
    ),
    "eyes-on-every-street": (
        "You write the Eyes On Every Street briefing — anomaly detection "
        "across every system. Lead with the most overdue / aging items. Note "
        "blind spots explicitly. Voice: observational, structural, names the "
        "pattern rather than itemizing. End with a Bottom Line that names the "
        "single most-urgent thing on Glen's desk."
    ),
    "founders-radar": (
        "You write Founder's Radar — an attention allocator. For each domain "
        "(client messages, revenue, pipeline, ads), list aging items with "
        "specific days-overdue, then say what to clear today vs tomorrow. "
        "Voice: triage-officer; explicit ages, explicit deadlines."
    ),
    "revenue-x-ray": (
        "You write Revenue X-Ray — a revenue-per-founder-hour analysis. "
        "Lead with Cash Position (Wise + bank totals). Then Pipeline Reality "
        "(GHL counts that could convert this week). Then Client Response "
        "Backlog (retention risk). Name the constraint (Schwerpunkt). End "
        "with a Bottom Line that names the single revenue lever to pull today."
    ),
    "pattern-which-connects": (
        "You write Pattern Which Connects — Bateson/Sheldrake voice, "
        "cross-system pattern detection. Find the META-pattern across "
        "domains (the structural thing that recurs everywhere). Quote "
        "specific numbers as evidence but stay one level up. Note Data Gaps "
        "explicitly. End with a Bottom Line that names the structural fact "
        "every stuck item traces back to."
    ),
}


def _safe(fn, default=None, label=""):
    """Call fn() and return its value; on exception return a marker dict
    instead of crashing the whole snapshot."""
    try:
        return fn()
    except Exception as e:
        return {"_error": f"{label}: {type(e).__name__}: {e}"} if default is None else default


def gather_snapshot():
    """Pull live stats from every dashboard module. Partial failures are
    captured as `_error` markers so Claude can note 'data unavailable'
    rather than the whole run dying."""
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "money": {
            "banks":   _safe(_money.qb_banks,      label="qb_banks"),
            "today":   _safe(_money.today_summary, label="today_summary"),
            "week":    _safe(_money.week_summary,  label="week_summary"),
            "wise":    _safe(_money.wise_data,     label="wise"),
            "pb":      _safe(lambda: _money.pb_data(days=30), label="pb_data"),
            "an":      _safe(lambda: _money.an_data(days=30), label="an_data"),
        },
        "ghl":        _safe(_ghl.opportunities_by_stage, label="ghl"),
        "pinecone":   _safe(_pc_stats.index_stats,       label="pinecone"),
        "scoreapp":   _safe(lambda: _scoreapp.recent_signups(limit=20), label="scoreapp"),
        "heygen":     _safe(lambda: _heygen.recent_videos(limit=5),     label="heygen"),
    }


def _build_user_prompt(snapshot, slug):
    """The user-side prompt embeds the JSON snapshot + a hint about today's
    date so Claude doesn't reuse stale dates from its training data."""
    today = datetime.now(timezone.utc)
    return (
        f"Today is {today.strftime('%A, %B %d, %Y')} ({today.isoformat()} UTC).\n\n"
        f"Live system snapshot (JSON):\n```json\n{json.dumps(snapshot, indent=2, default=str)}\n```\n\n"
        f"Write the {slug} briefing as markdown. Start with an H1 title and an "
        f"italic dateline. Do not wrap in code fences. Do not invent numbers "
        f"that aren't in the snapshot — if a source has an `_error` key, say "
        f"that source is unavailable. Be specific about names, amounts, and ages."
    )


def _generate_one(client, slug, snapshot):
    """Generate a single briefing's markdown via Anthropic."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SLUG_PROMPTS[slug],
        messages=[{"role": "user", "content": _build_user_prompt(snapshot, slug)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def regenerate_all():
    """Generate all 5 briefings in parallel and write each to disk. Returns a
    summary dict the cron handler echoes back."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    snapshot = gather_snapshot()
    client = anthropic.Anthropic()

    results = {}
    with ThreadPoolExecutor(max_workers=len(VALID_SLUGS)) as pool:
        futures = {pool.submit(_generate_one, client, s, snapshot): s for s in VALID_SLUGS}
        for fut in as_completed(futures):
            slug = futures[fut]
            try:
                markdown = fut.result()
                _intel.write_briefing(slug, markdown)
                results[slug] = {"ok": True, "bytes": len(markdown)}
            except Exception as e:
                results[slug] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return {
        "model": MODEL,
        "snapshot_as_of": snapshot["as_of"],
        "results": results,
        "ok_count": sum(1 for v in results.values() if v["ok"]),
        "fail_count": sum(1 for v in results.values() if not v["ok"]),
    }
