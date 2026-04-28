"""
Incentive engine — content selector + send orchestrator + feedback
processor for the Phase 0 beta. Imported into app.py at module load.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


ANTI_STALE_DAYS = 30
HIGH_AFFINITY_DAYS = 14


def _parse_state_field(state: dict, key: str, default):
    raw = state.get(key)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def select_topic_for_user(
    user_state: dict,
    candidate_topics: list,
    audience: str = "client",
) -> Optional[str]:
    """Pick the next topic for this user from the candidate pool.

    Rules (MVP):
      1. Eliminate topics sent within the last 30 days (anti-stale).
      2. Among remaining, prefer the topic with highest click count
         in the user's topic_engagement_history.
      3. If no engagement history, pick the first candidate
         (deterministic fallback).
      4. Return None if no candidate survives anti-stale filter.
    """
    now = datetime.now(timezone.utc)
    send_history = _parse_state_field(user_state, "topic_send_history", [])
    engagement = _parse_state_field(user_state, "topic_engagement_history", [])

    recent = set()
    for entry in send_history:
        try:
            sent_at = datetime.fromisoformat(entry["last_sent_at"])
            if (now - sent_at) <= timedelta(days=ANTI_STALE_DAYS):
                recent.add(entry["topic"])
        except Exception:
            continue

    fresh = [t for t in candidate_topics if t not in recent]
    if not fresh:
        return None

    clicks = {e["topic"]: e.get("click_count", 0) for e in engagement}

    fresh.sort(key=lambda t: (-clicks.get(t, 0), t))
    return fresh[0]


# ── Personal email generator (Task 7) ────────────────────────────────

_TPL_DIR = Path(__file__).parent / "templates"
_jinja_env = None


def _get_jinja_env():
    global _jinja_env
    if _jinja_env is None:
        from jinja2 import Environment, FileSystemLoader
        _jinja_env = Environment(
            loader=FileSystemLoader(str(_TPL_DIR)),
            autoescape=False,  # plain text — no HTML escaping
            keep_trailing_newline=True,
        )
    return _jinja_env


def _llm_complete(prompt: str, max_tokens: int = 500) -> str:
    """Call Claude Haiku for the teaching body. Replaceable in tests."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def generate_personal_email(
    user: dict,
    topic: str,
    topic_source_text: str,
    product: dict,
    is_beta: bool = False,
    audience: str = "client",
    monthly_window: bool = False,
) -> dict:
    """Build a Glen-voice plain-text Personal email for one user.

    Returns:
      {"subject": str, "body": str}
    """
    audience_framing = (
        "Write at a clinical-practitioner level — assume the reader can "
        "interpret mechanism and dosage. Lean into mechanism-of-action depth."
        if audience == "practitioner"
        else
        "Write in warm, accessible language. No clinical jargon. "
        "Speak to the reader's own healing intelligence."
    )

    prompt = (
        "You are writing a short plain-text email FROM Dr. Glen Swartwout "
        "to a single subscriber, in his personal voice. ~150-300 words.\n\n"
        "Today's topic: " + topic + "\n\n"
        "Source passage to ground the teaching (synthesize, don't quote):\n"
        + topic_source_text[:2000] + "\n\n"
        "Style: " + audience_framing + "\n\n"
        "Open with ONE teaching nugget — a distinction, a question, or a "
        "surprising connection. Then naturally transition to recommending "
        "the product " + product["name"] + ". Keep it warm and direct. No "
        "subject line, no greeting, no signature — just the body. ~150-300 "
        "words."
    )
    teaching_body = _llm_complete(prompt, max_tokens=500)

    subj_prompt = (
        f"Write a short email subject line (max 50 chars) for a personal "
        f"email about {topic}. Should sound like a question or a "
        f"distinction, NOT a sale. Output only the subject line."
    )
    subject = _llm_complete(subj_prompt, max_tokens=30).strip().strip('"')[:60]

    env = _get_jinja_env()
    template = env.get_template("personal_email.txt.j2")
    body = template.render(
        user=user,
        teaching_body=teaching_body,
        product=product,
        is_beta=is_beta,
        monthly_window=monthly_window,
        unsubscribe_url=(
            f"https://glen-knowledge-chat.onrender.com/"
            f"unsubscribe?email={user['email']}&channel=personal"
        ),
    )

    return {"subject": subject, "body": body}


# ── Engagement-gated send decision (Task 8) ──────────────────────────

DORMANT_THRESHOLD_DAYS = 14


def should_send_today(state: dict, paused: bool = False) -> bool:
    """Decide whether to send Personal email to this user today.

    Rules:
      1. If admin paused → no.
      2. If never sent before → yes (welcome moment).
      3. If last send was today (UTC) → no (don't double-send).
      4. If last_open_at OR last_click_at is at-or-after last_send_at
         → yes (they engaged).
      5. Otherwise → no (engagement gate closed).
    """
    if paused:
        return False
    last_send = state.get("last_send_at")
    if not last_send:
        return True  # first send for this user

    now = datetime.now(timezone.utc)
    try:
        last_send_dt = datetime.fromisoformat(last_send)
    except Exception:
        return True  # corrupted timestamp; default to send

    if last_send_dt.date() == now.date():
        return False

    last_open = state.get("last_open_at")
    last_click = state.get("last_click_at")

    def _at_or_after(ts_str, ref_dt):
        if not ts_str:
            return False
        try:
            return datetime.fromisoformat(ts_str) >= ref_dt
        except Exception:
            return False

    return _at_or_after(last_click, last_send_dt) or _at_or_after(
        last_open, last_send_dt
    )


# ── Reply ingestion + categorization (Task 9) ────────────────────────

ROUTE_BY_CATEGORY = {
    "suggestion":    "glen-review",
    "correction":    "pinecone-correction",
    "topic-request": "glen-review",
    "complaint":     "glen-review",
    "praise":        "archive",
    "question":      "glen-review",
}


def process_reply(
    user_id: int,
    original_send_id: Optional[int],
    raw_text: str,
) -> dict:
    """Run a Claude Haiku call on the reply, return structured fields.

    Returns:
      {ai_summary, ai_category, extracted_topics, extracted_products,
       extracted_conditions, routed_to}
    """
    prompt = (
        "Analyze this email reply from a wellness-newsletter subscriber. "
        "Output STRICT JSON with these fields:\n"
        "  summary: 1-2 sentence summary\n"
        "  category: one of "
        "    suggestion | correction | topic-request | complaint | praise | question\n"
        "  topics: list of topic labels mentioned (e.g. 'leaky-gut',\n"
        "    'wet-AMD', 'glaucoma', 'EMF', 'omega-3')\n"
        "  products: list of formulation names mentioned\n"
        "  conditions: list of health conditions / symptoms mentioned\n\n"
        f"Reply text:\n{raw_text[:4000]}"
    )
    raw = _llm_complete(prompt, max_tokens=500)
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "summary": raw[:200],
            "category": "question",
            "topics": [],
            "products": [],
            "conditions": [],
        }

    category = parsed.get("category", "question")
    return {
        "ai_summary":           parsed.get("summary", "")[:500],
        "ai_category":          category,
        "extracted_topics":     parsed.get("topics", []),
        "extracted_products":   parsed.get("products", []),
        "extracted_conditions": parsed.get("conditions", []),
        "routed_to":            ROUTE_BY_CATEGORY.get(category, "glen-review"),
    }


# ── Reply-as-personalization update loop (Task 10) ───────────────────

import sqlite3 as _sqlite3

LOG_DB = str(
    Path(os.environ.get("DATA_DIR", str(Path(__file__).parent))) / "chat_log.db"
)

REPLY_BOOST_WEIGHT = 2  # a reply counts as 2 clicks (stronger signal)


def update_personalization_from_reply(
    user_id: int,
    extracted_topics: list,
    extracted_products: list,
) -> None:
    """Boost the user's topic + product affinity based on what they
    mentioned in their reply. Replies are stronger signals than clicks
    because the user invested effort to write."""
    with _sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = _sqlite3.Row
        row = cx.execute(
            "SELECT topic_engagement_history, product_affinity "
            "FROM personal_email_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            cx.execute(
                "INSERT INTO personal_email_state (user_id, "
                "topic_engagement_history, product_affinity) VALUES (?, ?, ?)",
                (user_id, "[]", "{}"),
            )
            history, affinity = [], {}
        else:
            history = json.loads(row["topic_engagement_history"] or "[]")
            affinity = json.loads(row["product_affinity"] or "{}")

        by_topic = {h["topic"]: h for h in history}
        now_iso = datetime.now(timezone.utc).isoformat()
        for t in extracted_topics:
            if t in by_topic:
                by_topic[t]["click_count"] = (
                    by_topic[t].get("click_count", 0) + REPLY_BOOST_WEIGHT
                )
                by_topic[t]["last_clicked_at"] = now_iso
            else:
                by_topic[t] = {
                    "topic": t,
                    "click_count": REPLY_BOOST_WEIGHT,
                    "last_clicked_at": now_iso,
                }
        history = list(by_topic.values())

        for p in extracted_products:
            affinity[p] = affinity.get(p, 0) + REPLY_BOOST_WEIGHT

        cx.execute(
            "UPDATE personal_email_state SET topic_engagement_history = ?, "
            "product_affinity = ? WHERE user_id = ?",
            (json.dumps(history), json.dumps(affinity), user_id),
        )
        cx.commit()


# ── Newsletter renderer (Task 11) ────────────────────────────────────


def generate_newsletter(
    kind: str,                          # 'weekly' or 'monthly'
    user: dict,
    title: str,
    body_html: str,
    offer: dict,                        # {product_url, code, pct, cta_text, deadline, ...}
    is_beta: bool = False,
    personal_note: Optional[str] = None,
    closeouts: Optional[list] = None,
) -> dict:
    """Render a per-subscriber newsletter HTML body.

    The broadcast body (body_html, title, offer) is the same for everyone
    in this send; personal_note differs per user.

    Returns: {"subject": str, "body": str}
    """
    template_name = (
        f"newsletter_{kind}.html.j2"
        if kind in ("weekly", "monthly")
        else "newsletter_weekly.html.j2"
    )
    env = _get_jinja_env()
    template = env.get_template(template_name)

    body = template.render(
        title=title,
        body=body_html,
        offer_pitch=offer.get("pitch", ""),
        product_url=offer.get("product_url", ""),
        cta_text=offer.get("cta_text", "Shop now"),
        coupon_code=offer.get("code", ""),
        discount_pct=offer.get("pct", 10),
        deadline=offer.get("deadline", "this Friday"),
        headline_offer=offer.get("headline", ""),
        window_days=offer.get("window_days", 3),
        month_label=offer.get("month_label", ""),
        closeouts=closeouts or [],
        personal_note=personal_note,
        is_beta=is_beta,
        unsubscribe_url=(
            f"https://glen-knowledge-chat.onrender.com/"
            f"unsubscribe?email={user['email']}&channel={kind}"
        ),
    )

    return {"subject": title, "body": body}


# ── Per-subscriber personal closing note (Task 12) ───────────────────

DEFAULT_PERSONAL_NOTE = (
    "Reply with anything — questions, requests, corrections, what you'd "
    "want to read about. I read every reply. And if you'd like a personal "
    "email cadence (adaptive to what interests you), reply with 'personal' "
    "and I'll set it up."
)


def build_personal_note_for_user(user_state: dict) -> str:
    """Generate the per-user personal note section for newsletters.

    Returns DEFAULT_PERSONAL_NOTE if no engagement data exists; otherwise
    asks the LLM for a 1-2 sentence note that references the user's
    top-affinity topic without being creepy.
    """
    history = _parse_state_field(user_state, "topic_engagement_history", [])
    high_affinity = [h for h in history if h.get("click_count", 0) > 0]

    if not high_affinity:
        return DEFAULT_PERSONAL_NOTE

    high_affinity.sort(key=lambda h: -h.get("click_count", 0))
    top_topic = high_affinity[0]["topic"]

    prompt = (
        f"Write a 1-2 sentence personal note in Dr. Glen Swartwout's warm "
        f"direct voice, addressed to a subscriber whose recent reading "
        f"interest is '{top_topic}'. Reference the topic without being "
        f"creepy about it; tease something coming up that ties in. "
        f"Output ONLY the note, no greeting, no signature."
    )
    return _llm_complete(prompt, max_tokens=120).strip()


# ── Beta send orchestrator + cron worker (Task 13) ───────────────────


def _init_test_state(db_path: str, rows: list) -> None:
    """Test helper — seed users + personal_email_state rows."""
    with _sqlite3.connect(db_path) as cx:
        cx.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                email TEXT,
                name TEXT
            );
            CREATE TABLE IF NOT EXISTS personal_email_state (
                user_id INTEGER PRIMARY KEY,
                last_send_at TEXT,
                last_open_at TEXT,
                last_click_at TEXT,
                consecutive_no_engagement_days INTEGER DEFAULT 0,
                topic_engagement_history TEXT,
                topic_send_history TEXT,
                product_affinity TEXT,
                paused_until TEXT
            );
            CREATE TABLE IF NOT EXISTS personal_email_sends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                sent_at TEXT NOT NULL,
                channel TEXT NOT NULL,
                topic TEXT,
                product_name TEXT,
                coupon_code TEXT,
                subject TEXT,
                body_snippet TEXT,
                opened_at TEXT,
                clicked_at TEXT
            );
        """)
        for r in rows:
            cx.execute(
                "INSERT OR REPLACE INTO users (id, email, name) VALUES (?,?,?)",
                (r["user_id"], r["email"], r["name"]),
            )
            cx.execute(
                """INSERT OR REPLACE INTO personal_email_state
                   (user_id, last_send_at, last_open_at, last_click_at)
                   VALUES (?,?,?,?)""",
                (
                    r["user_id"],
                    r.get("last_send_at"),
                    r.get("last_open_at"),
                    r.get("last_click_at"),
                ),
            )
        cx.commit()


def _send_email(user: dict, subject: str, body: str) -> None:
    """Plug-point for sending. Phase 0 default: SMTP via env vars; falls
    back to stdout. Replaceable in tests."""
    smtp_host = os.environ.get("SMTP_HOST")
    if smtp_host:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", ""))
        msg["To"] = user["email"]
        with smtplib.SMTP(smtp_host, int(os.environ.get("SMTP_PORT", "587"))) as s:
            s.starttls()
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            s.sendmail(msg["From"], [user["email"]], msg.as_string())
    else:
        print(
            f"\n[email] TO: {user['email']}\nSUBJECT: {subject}\n\n{body}\n",
            flush=True,
        )


def _load_incentive_config() -> dict:
    cfg_path = Path(__file__).parent / "data" / "incentive-config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {"beta_cohort_emails": []}


def _list_beta_cohort_users() -> list:
    """Phase 0: identify beta users by email-list in incentive-config.json.
    Phase 1+ will switch to GHL tag membership."""
    config = _load_incentive_config()
    cohort_emails = list(config.get("beta_cohort_emails", []))
    if not cohort_emails:
        return []

    placeholders = ",".join(["?"] * len(cohort_emails))
    with _sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = _sqlite3.Row
        rows = cx.execute(
            f"SELECT id, email, name FROM users WHERE email IN ({placeholders})",
            tuple(cohort_emails),
        ).fetchall()
    return [
        {"id": r["id"], "email": r["email"], "name": r["name"]} for r in rows
    ]


def _load_user_state(user_id: int) -> dict:
    with _sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = _sqlite3.Row
        row = cx.execute(
            "SELECT * FROM personal_email_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else {}


def _record_send(
    user_id: int,
    channel: str,
    topic: str,
    product_name: str,
    coupon_code: str,
    subject: str,
    body: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _sqlite3.connect(LOG_DB) as cx:
        cx.execute(
            """INSERT INTO personal_email_sends
               (user_id, sent_at, channel, topic, product_name,
                coupon_code, subject, body_snippet)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                user_id, now, channel, topic, product_name, coupon_code,
                subject, body[:500],
            ),
        )
        cx.execute(
            "INSERT OR IGNORE INTO personal_email_state (user_id) VALUES (?)",
            (user_id,),
        )
        # Update last_send_at + topic_send_history (anti-stale tracking)
        cx.row_factory = _sqlite3.Row
        row = cx.execute(
            "SELECT topic_send_history FROM personal_email_state "
            "WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        history = (
            json.loads(row["topic_send_history"] or "[]")
            if row and row["topic_send_history"] else []
        )
        # Replace existing entry for this topic, or append
        history = [h for h in history if h.get("topic") != topic]
        history.append({"topic": topic, "last_sent_at": now})
        cx.execute(
            "UPDATE personal_email_state "
            "SET last_send_at = ?, topic_send_history = ? "
            "WHERE user_id = ?",
            (now, json.dumps(history), user_id),
        )
        cx.commit()


def run_daily_send_for_beta_cohort() -> int:
    """Iterate the beta cohort, apply engagement gate, generate + send +
    record for each pass-through user. Returns number sent."""
    config = _load_incentive_config()
    sent = 0
    for user in _list_beta_cohort_users():
        state = _load_user_state(user["id"])
        if not should_send_today(state, paused=False):
            continue

        # Audience routing — Phase 0 default 'client'; Phase 1 enhancement
        # could read GHL tag from a cached column. For now, default 'client'
        # for the broad cohort.
        audience = state.get("audience_tag") or "client"

        # Adaptive topic selection from Pinecone audience-filtered pool
        from pinecone_content_pool import (
            candidate_topics_for_audience,
            fetch_source_text_for_topic,
        )
        try:
            candidate_topics = candidate_topics_for_audience(audience)
        except Exception as e:
            print(
                f"[orch] Pinecone fetch failed for user {user['id']}: {e}",
                flush=True,
            )
            candidate_topics = []

        if not candidate_topics:
            # Pool empty — fall back to the original stub so beta
            # subscribers still receive *something* during the warm-up window.
            topic = "leaky-gut"
            topic_source_text = "(content pool unavailable — fallback)"
        else:
            chosen = select_topic_for_user(state, candidate_topics, audience)
            if chosen is None:
                # All candidates stale for this user — skip them today.
                continue
            topic = chosen
            topic_source_text = (
                fetch_source_text_for_topic(chosen, audience)
                or "(no source text fetched)"
            )

        product = {
            "name": "Terrain Restore",
            "url":  "https://truly.vip/terrain-restore",
            "code": config.get("beta_shared_code", "BETA5"),
        }

        email = generate_personal_email(
            user=user,
            topic=topic,
            topic_source_text=topic_source_text,
            product=product,
            is_beta=True,
            audience=audience,
        )
        _send_email(user, email["subject"], email["body"])
        _record_send(
            user["id"], "personal", topic, product["name"],
            product["code"], email["subject"], email["body"],
        )
        sent += 1
    return sent
