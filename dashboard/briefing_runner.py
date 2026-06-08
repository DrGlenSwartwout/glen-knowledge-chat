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
from . import inbox as _inbox
from . import briefing_actions as _ba


MODEL = os.environ.get("BRIEFING_MODEL", "claude-haiku-4-5-20251001")

VALID_SLUGS = [
    "money-cash",
    "clients-pipeline",
    "signals-patterns",
]


# Three domain cards. Each OWNS its slice of the snapshot so a statistic appears
# on exactly one card (no cross-card repetition). Shared format rules (H1 +
# dateline, the acronym/retired-pipeline guards, and the [HIGH]/[MED]/[LOW]
# action rubric) live in _build_user_prompt.
SLUG_PROMPTS = {
    "money-cash": (
        "You write the Money & Cash card for Glen Swartwout's Command Center. "
        "Cover ONLY money, from the snapshot's `money` block: the combined cash "
        "position (bank balances + Wise together), money in today and over 7 "
        "days, accounts receivable / overdue, and any Practice Better or "
        "Authorize.net processor issues. Name the single revenue constraint "
        "(the Schwerpunkt) limiting cash this week. Do NOT cover pipeline, "
        "leads, or system health — other cards own those. State each figure "
        "once. Then one short '## Insight' line on runway / trend. Voice: "
        "calm, precise, money-first."
    ),
    "clients-pipeline": (
        "You write the Clients & Pipeline card. Cover ONLY client relationships "
        "and the sales pipeline, from the snapshot's `inbox`, `gohighlevel` "
        "(pipeline opportunities by stage), and `scoreapp` (recent quiz signups) "
        "blocks. Lead with the client-message backlog: `inbox.awaiting_reply` is "
        "the count of unread Primary-category mail from the last 30 days (the "
        "real retention-risk queue), and `inbox.oldest` lists the longest-waiting "
        "with ages — name a few. Mention `inbox.inbox_unread_total` only as a "
        "one-line aside ('untriaged inbox: N unread'); it is context, NOT the "
        "backlog. Then pipelines holding opportunities that could convert this "
        "week, then new leads worth contacting. Do NOT cover cash / accounts "
        "receivable or system health. If a source has an `_error`, say it is "
        "unavailable rather than inventing. Then one short '## Insight' line on "
        "the biggest retention risk or best conversion bet. Voice: triage-officer; "
        "name the oldest waiting senders, pipelines, stages, counts, and lead names."
    ),
    "signals-patterns": (
        "You write the Signals & Patterns card. Cover cross-system health from "
        "the snapshot's `pinecone` (knowledge base) and `heygen` (video) blocks "
        "plus any `_error` markers anywhere in the snapshot (those are the blind "
        "spots). Then, one level up, name the structural meta-pattern that "
        "recurs across the systems — Bateson / Sheldrake voice. Do NOT restate "
        "the cash or pipeline figures the other two cards own. Call out data "
        "gaps explicitly. Then one short '## Insight' line naming the "
        "structural fact. Voice: observational, structural."
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
    rather than the whole run dying.

    KEY NAMING: use the full, unambiguous product name (e.g. `practice_better`,
    not `pb`) so the LLM doesn't have to expand acronyms — it has historically
    guessed wrong (e.g. "PayBlade" instead of "Practice Better")."""
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "money": {
            "banks":           _safe(_money.qb_banks,      label="qb_banks"),
            "today":           _safe(_money.today_summary, label="today_summary"),
            "week":            _safe(_money.week_summary,  label="week_summary"),
            "wise":            _safe(_money.wise_data,     label="wise"),
            "practice_better": _safe(lambda: _money.pb_data(days=30), label="pb_data"),
            "authorize_net":   _safe(lambda: _money.an_data(days=30), label="an_data"),
        },
        "gohighlevel": _safe(_ghl.opportunities_by_stage, label="ghl"),
        "inbox":       _safe(_inbox.backlog_summary,      label="inbox"),
        "pinecone":    _safe(_pc_stats.index_stats,       label="pinecone"),
        "scoreapp":    _safe(lambda: _scoreapp.recent_signups(limit=20), label="scoreapp"),
        "heygen":      _safe(lambda: _heygen.recent_videos(limit=5),     label="heygen"),
    }


def _build_user_prompt(snapshot, slug):
    """The user-side prompt embeds the JSON snapshot + a hint about today's
    date so Claude doesn't reuse stale dates from its training data."""
    today = datetime.now(timezone.utc)
    return (
        f"Today is {today.strftime('%A, %B %d, %Y')} ({today.isoformat()} UTC).\n\n"
        "Glossary (use these exact names — do NOT invent alternatives):\n"
        "  • practice_better → Practice Better (clinical-practice billing/EHR; PB)\n"
        "  • authorize_net   → Authorize.net (storefront card processor; AuthNet)\n"
        "  • gohighlevel     → GoHighLevel (CRM + pipelines; GHL)\n"
        "  • wise            → Wise (multi-currency banking)\n"
        "  • scoreapp        → ScoreApp (quiz-funnel intake at healing.scoreapp.com)\n"
        "  • heygen          → HeyGen (AI video renders)\n"
        "  • pinecone        → Pinecone (vector knowledge base)\n\n"
        f"Live system snapshot (JSON):\n```json\n{json.dumps(snapshot, indent=2, default=str)}\n```\n\n"
        f"Write the {slug} briefing as markdown. Start with an H1 title and an "
        f"italic dateline. Do not wrap in code fences. Do not invent numbers "
        f"that aren't in the snapshot — if a source has an `_error` key, say "
        f"that source is unavailable. Be specific about names, amounts, and ages. "
        f"Spell out every acronym in full the first time it appears in each "
        f"briefing, with the short form in parentheses after it — e.g. "
        f"'End of Day (EOD)'. Only reference pipelines and sources that appear "
        f"in the snapshot; never mention retired ones (e.g. MCTB, Email Paramedic).\n\n"
        f"End the card with a '## Recommended actions' section: 1 to 3 actions, "
        f"each on its own line, each STARTING with a severity tag in square "
        f"brackets chosen from urgency AND potential financial impact — "
        f"[HIGH] (act now / high dollars at stake), [MED] (this week / moderate), "
        f"[LOW] (maintenance / low impact). Put the most significant action "
        f"first. Example line: '[HIGH] Collect the $4,210 overdue from Acme — "
        f"30+ days aged.' If there is genuinely nothing to do, write "
        f"'[LOW] No action needed.'"
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
                _ba.reset_slug(slug)   # fresh briefing => clear handled-action state
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
