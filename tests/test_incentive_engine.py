"""Tests for the Phase 0 incentive engine.

These tests exercise the additive schema migration that creates the
incentive engine tables alongside the existing query_log table.
"""

import subprocess
import sqlite3
import sys
from pathlib import Path


def test_incentive_schema_creates_required_tables(tmp_path):
    """The schema migration should create personal_email_state,
    personal_email_sends, personal_email_feedback, and
    holdout_assignments tables.

    Runs in a subprocess with a fresh DATA_DIR so the assertion
    actually exercises the migration logic, not preexisting tables
    in the dev chat_log.db.
    """
    db_dir = tmp_path
    db_path = db_dir / "chat_log.db"

    repo_root = Path(__file__).resolve().parent.parent
    code = (
        "import sys, os\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        f"os.environ['DATA_DIR'] = {str(db_dir)!r}\n"
        "import app\n"
        "app._init_log_db()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"

    with sqlite3.connect(str(db_path)) as cx:
        tables = {
            r[0]
            for r in cx.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert "personal_email_state" in tables
    assert "personal_email_sends" in tables
    assert "personal_email_feedback" in tables
    assert "holdout_assignments" in tables


def test_resolve_channel_tags_includes_personal_newsletter_beta():
    """_resolve_channel_tags should produce tags for each opt-in flag."""
    from app import _resolve_channel_tags
    tags = _resolve_channel_tags(personal=True, newsletter=True, is_beta=True)
    assert "personal-email-opt-in" in tags
    assert "newsletter-opt-in" in tags
    assert "beta-personal-email" in tags
    assert "chatbot-lead" in tags  # baseline tag

def test_resolve_channel_tags_omits_unchecked():
    """If a channel is not opted in, its tag should not appear."""
    from app import _resolve_channel_tags
    tags = _resolve_channel_tags(personal=False, newsletter=True, is_beta=False)
    assert "personal-email-opt-in" not in tags
    assert "newsletter-opt-in" in tags
    assert "beta-personal-email" not in tags


# ── Task 6: content selector ─────────────────────────────────────────

import json
from datetime import datetime, timedelta, timezone


def test_select_topic_respects_anti_stale():
    """If a topic was sent in last 30 days, selector should skip it."""
    from incentive_engine import select_topic_for_user
    user_state = {
        "topic_send_history": json.dumps([
            {"topic": "wet-AMD", "last_sent_at":
             (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()},
            {"topic": "leaky-gut", "last_sent_at":
             (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()},
        ]),
        "topic_engagement_history": json.dumps([
            {"topic": "wet-AMD", "click_count": 5},
            {"topic": "leaky-gut", "click_count": 1},
        ]),
    }
    candidate_topics = ["wet-AMD", "leaky-gut", "EMF"]
    chosen = select_topic_for_user(user_state, candidate_topics, "client")
    assert chosen != "wet-AMD"  # too recent (anti-stale)
    assert chosen in ["leaky-gut", "EMF"]


def test_select_topic_prefers_high_affinity():
    """Among non-stale candidates, the one with most clicks wins."""
    from incentive_engine import select_topic_for_user
    user_state = {
        "topic_send_history": json.dumps([]),
        "topic_engagement_history": json.dumps([
            {"topic": "leaky-gut", "click_count": 7},
            {"topic": "EMF", "click_count": 1},
        ]),
    }
    chosen = select_topic_for_user(user_state, ["leaky-gut", "EMF", "vision"], "client")
    assert chosen == "leaky-gut"


def test_select_topic_returns_none_if_all_stale():
    """If every candidate was sent recently, return None."""
    from incentive_engine import select_topic_for_user
    user_state = {
        "topic_send_history": json.dumps([
            {"topic": "X", "last_sent_at":
             (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()},
            {"topic": "Y", "last_sent_at":
             (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()},
        ]),
        "topic_engagement_history": json.dumps([]),
    }
    assert select_topic_for_user(user_state, ["X", "Y"], "client") is None


def test_select_topic_deterministic_alphabetical_tiebreak():
    """When clicks tied at zero, alphabetical wins (deterministic)."""
    from incentive_engine import select_topic_for_user
    user_state = {
        "topic_send_history": json.dumps([]),
        "topic_engagement_history": json.dumps([]),
    }
    chosen = select_topic_for_user(user_state, ["zebra", "apple", "mango"], "client")
    assert chosen == "apple"


# ── Task 7: AI-draft personal email generator ────────────────────────


def test_generate_personal_email_includes_required_sections(monkeypatch):
    """Generated email must contain: teaching nugget, product link with
    coupon code, share-PS, beta banner (when in beta), unsubscribe link."""
    from incentive_engine import generate_personal_email

    fake_topic_text = "Terrain Restore activates intestinal tight junctions..."
    fake_product = {
        "name": "Terrain Restore",
        "url":  "https://truly.vip/terrain-restore",
        "code": "TR-U42-W17",
    }

    def fake_llm(prompt, max_tokens=500):
        # Subject prompt opens with "Write a short email subject line";
        # teaching prompt opens with "You are writing a short plain-text email".
        if prompt.lower().startswith("write a short email subject line"):
            return "What if your gut runs the show?"
        return ("Today's discovery: tight junctions matter. "
                "Try Terrain Restore 30 minutes before meals.")
    monkeypatch.setattr("incentive_engine._llm_complete", fake_llm)

    email = generate_personal_email(
        user={"id": 42, "name": "Test", "email": "test@example.com"},
        topic="leaky-gut",
        topic_source_text=fake_topic_text,
        product=fake_product,
        is_beta=True,
        audience="client",
    )

    assert "tight junctions" in email["body"].lower()
    assert "TR-U42-W17" in email["body"]
    assert "https://truly.vip/terrain-restore" in email["body"]
    assert "P.S." in email["body"]
    assert "Beta" in email["body"]      # banner for beta cohort
    assert "unsubscribe" in email["body"].lower()
    assert email["subject"]              # non-empty subject


# ── Task 8: engagement-gated send decision ───────────────────────────


def test_should_send_today_unlocks_after_engagement():
    """Should send if user opened/clicked yesterday's send."""
    from incentive_engine import should_send_today

    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    state = {
        "last_send_at": yesterday,
        "last_open_at": yesterday,
        "consecutive_no_engagement_days": 0,
    }
    assert should_send_today(state, paused=False) is True


def test_should_send_today_pauses_after_no_engagement():
    """Should NOT send if user didn't engage with yesterday's email."""
    from incentive_engine import should_send_today
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    state = {
        "last_send_at": yesterday,
        "last_open_at": None,
        "last_click_at": None,
        "consecutive_no_engagement_days": 1,
    }
    assert should_send_today(state, paused=False) is False


def test_should_send_today_first_send_for_new_user():
    """Brand-new opt-in (no last_send_at) should send immediately."""
    from incentive_engine import should_send_today
    state = {"last_send_at": None}
    assert should_send_today(state, paused=False) is True


def test_should_send_today_paused_admin_override():
    """Paused flag overrides everything."""
    from incentive_engine import should_send_today
    state = {"last_send_at": None}  # would normally send
    assert should_send_today(state, paused=True) is False


def test_should_send_today_no_double_send_same_day():
    """If we already sent today, don't send again today."""
    from incentive_engine import should_send_today
    today = datetime.now(timezone.utc).isoformat()
    state = {"last_send_at": today, "last_open_at": today}
    assert should_send_today(state, paused=False) is False


def test_should_send_today_click_counts_as_engagement():
    """A click is sufficient engagement to unlock today's send."""
    from incentive_engine import should_send_today
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    state = {
        "last_send_at": yesterday,
        "last_open_at": None,
        "last_click_at": yesterday,  # clicked but didn't open (rare but real)
    }
    assert should_send_today(state, paused=False) is True


# ── Task 9: reply ingestion + AI categorization ──────────────────────


def test_process_reply_extracts_topics_and_categorizes(monkeypatch):
    from incentive_engine import process_reply

    def fake_llm(prompt, max_tokens=500):
        return json.dumps({
            "summary": "User asks about Terrain Restore for their leaky gut.",
            "category": "topic-request",
            "topics": ["leaky-gut", "terrain-restore"],
            "products": ["Terrain Restore"],
            "conditions": ["leaky gut"],
        })
    monkeypatch.setattr("incentive_engine._llm_complete", fake_llm)

    result = process_reply(
        user_id=42,
        original_send_id=100,
        raw_text="I'd love more info on Terrain Restore for my leaky gut.",
    )
    assert result["ai_category"] == "topic-request"
    assert "leaky-gut" in result["extracted_topics"]
    assert result["routed_to"] == "glen-review"


def test_process_reply_routes_correction_to_pinecone(monkeypatch):
    from incentive_engine import process_reply
    def fake_llm(prompt, max_tokens=500):
        return json.dumps({
            "summary": "User flags incorrect dosage for AngiogenX.",
            "category": "correction",
            "topics": [],
            "products": ["AngiogenX"],
            "conditions": [],
        })
    monkeypatch.setattr("incentive_engine._llm_complete", fake_llm)

    result = process_reply(
        user_id=1, original_send_id=None,
        raw_text="The dose you mentioned for AngiogenX is wrong.",
    )
    assert result["routed_to"] == "pinecone-correction"


def test_process_reply_handles_malformed_llm_output(monkeypatch):
    """If LLM returns non-JSON, fall back to a sensible default."""
    from incentive_engine import process_reply
    monkeypatch.setattr("incentive_engine._llm_complete",
                       lambda p, max_tokens=500: "not valid json at all")
    result = process_reply(user_id=1, original_send_id=None,
                            raw_text="hi")
    assert result["ai_category"] == "question"  # safe default
    assert result["extracted_topics"] == []


# ── Task 10: reply-as-personalization update loop ────────────────────


def test_update_personalization_boosts_clicked_topics(tmp_db, monkeypatch):
    """When a reply mentions topics, those topics get +REPLY_BOOST_WEIGHT
    in topic_engagement_history; new topics are added at REPLY_BOOST_WEIGHT."""
    monkeypatch.setattr("incentive_engine.LOG_DB", tmp_db)

    with sqlite3.connect(tmp_db) as cx:
        cx.executescript("""
            CREATE TABLE personal_email_state (
              user_id                  INTEGER PRIMARY KEY,
              topic_engagement_history TEXT,
              product_affinity         TEXT
            );
            INSERT INTO personal_email_state
              (user_id, topic_engagement_history, product_affinity)
            VALUES (42, '[{"topic":"leaky-gut","click_count":1}]', '{}');
        """)
        cx.commit()

    from incentive_engine import update_personalization_from_reply
    update_personalization_from_reply(
        user_id=42,
        extracted_topics=["leaky-gut", "wet-AMD"],
        extracted_products=["Terrain Restore"],
    )

    with sqlite3.connect(tmp_db) as cx:
        row = cx.execute(
            "SELECT topic_engagement_history, product_affinity "
            "FROM personal_email_state WHERE user_id=42"
        ).fetchone()
    history = json.loads(row[0])
    affinity = json.loads(row[1])
    by_topic = {h["topic"]: h["click_count"] for h in history}
    assert by_topic["leaky-gut"] >= 3   # 1 baseline + 2 reply boost
    assert by_topic["wet-AMD"] >= 2     # new topic, +2 reply boost
    assert affinity.get("Terrain Restore", 0) >= 2


# ── Task 11: newsletter HTML renderer ────────────────────────────────


def test_generate_newsletter_renders_personal_note():
    from incentive_engine import generate_newsletter
    out = generate_newsletter(
        kind="weekly",
        user={"id": 1, "email": "u@example.com"},
        title="This week: tight junctions",
        body_html="<p>Body text here</p>",
        offer={"product_url": "https://truly.vip/x",
               "code": "WK10", "pct": 10, "cta_text": "Shop",
               "deadline": "Friday"},
        is_beta=True,
        personal_note="Saw you've been curious about leaky gut — next week's drop ties in.",
    )
    assert "Saw you've been curious" in out["body"]
    assert "WK10" in out["body"]
    assert "Beta" in out["body"]


def test_generate_newsletter_monthly_includes_closeouts():
    from incentive_engine import generate_newsletter
    out = generate_newsletter(
        kind="monthly",
        user={"id": 1, "email": "u@example.com"},
        title="April Edition",
        body_html="<p>Headline body</p>",
        offer={"product_url": "https://truly.vip/y", "code": "APR15",
               "pct": 15, "cta_text": "Get it", "headline": "Spring stack",
               "window_days": 5, "month_label": "April 2026"},
        closeouts=[
            {"name": "Old Formula A", "url": "https://truly.vip/a",
             "reason": "Reformulated"},
        ],
    )
    assert "Old Formula A" in out["body"]
    assert "APR15" in out["body"]
    assert "Monthly Edition" in out["body"]
    assert "Beta" not in out["body"]


# ── Task 12: per-subscriber personal closing note ────────────────────


def test_personal_note_uses_data_when_available(monkeypatch):
    from incentive_engine import build_personal_note_for_user

    def fake_llm(prompt, max_tokens=500):
        return "Saw you've been curious about leaky gut — next week's drop ties in."
    monkeypatch.setattr("incentive_engine._llm_complete", fake_llm)

    state = {"topic_engagement_history":
             '[{"topic":"leaky-gut","click_count":5}]'}
    note = build_personal_note_for_user(state)
    assert "leaky gut" in note.lower()


def test_personal_note_falls_back_when_no_data():
    from incentive_engine import build_personal_note_for_user
    state = {"topic_engagement_history": "[]"}
    note = build_personal_note_for_user(state)
    assert "reply" in note.lower()
    assert "personal email" in note.lower()


# ── Task 13: beta send orchestrator ──────────────────────────────────


def test_run_daily_send_iterates_beta_cohort_and_gates(tmp_db, monkeypatch, tmp_path):
    """Daily-send orchestrator iterates beta cohort users, applies
    should_send_today, and only sends to gated-pass users."""
    monkeypatch.setattr("incentive_engine.LOG_DB", tmp_db)

    # Point config-loader at a tmp config file with both users in cohort
    cfg_path = tmp_path / "incentive-config.json"
    cfg_path.write_text(json.dumps({
        "beta_cohort_emails": ["a@x.com", "b@x.com"],
        "beta_shared_code": "BETA5",
    }))
    monkeypatch.setattr("incentive_engine._load_incentive_config",
                        lambda: json.loads(cfg_path.read_text()))

    sent_emails = []

    def fake_send(user, subject, body):
        sent_emails.append({"user_id": user["id"], "subject": subject})
    monkeypatch.setattr("incentive_engine._send_email", fake_send)

    monkeypatch.setattr("incentive_engine._llm_complete",
                        lambda p, max_tokens=500: "Stub teaching.")

    from incentive_engine import _init_test_state
    _init_test_state(tmp_db, [
        {"user_id": 1, "name": "Alice", "email": "a@x.com",
         "last_send_at": "2026-04-26T00:00:00+00:00",
         "last_open_at":  "2026-04-26T08:00:00+00:00"},   # engaged → send
        {"user_id": 2, "name": "Bob",   "email": "b@x.com",
         "last_send_at": "2026-04-26T00:00:00+00:00",
         "last_open_at":  None},                           # silent → skip
    ])

    from incentive_engine import run_daily_send_for_beta_cohort
    n = run_daily_send_for_beta_cohort()
    assert n == 1
    assert sent_emails[0]["user_id"] == 1
