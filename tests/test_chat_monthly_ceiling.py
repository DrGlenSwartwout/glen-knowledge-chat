"""Tests for the monthly full-answer word ceiling (Task 6).

Strategy: deterministic data-layer + decision assertions rather than driving the
full RAG pipeline (which requires live Pinecone/Claude).  We:

1. Seed query_log rows via app.log_query (writes through the real DB path) for a
   capped email, then assert monthly_full_words returns >= 10 000.
2. Assert LIMITS["registered"]["monthly_full_words"] == 10 000 so the threshold
   matches what log_query seeded.
3. Assert that used >= cap would trigger mode="brief" (decision layer).
4. Seed a different email under the cap and assert its usage is < 10 000.
5. Assert member tier: monthly_full_words is None (no hard wall) while
   flag_full_words is 100 000.

No live LLM calls are made.  The test skips if app is not importable (missing env
secrets in CI) — same pattern as test_chat_velocity.py.
"""
import importlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _load_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        mod = importlib.import_module("app")
        importlib.reload(mod)  # pick up the DATA_DIR env var
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")
    try:
        mod._init_log_db()
    except Exception as e:
        pytest.skip(f"_init_log_db failed: {e}")
    return mod


# ── 1. Capped email: usage at/above 10 000 ───────────────────────────────────

def test_monthly_full_words_at_cap(tmp_path, monkeypatch):
    """Seeding 10 000 full-mode words makes monthly_full_words return >= 10 000."""
    app = _load_app(tmp_path, monkeypatch)
    from dashboard.chat_limits import monthly_full_words, LIMITS

    cap_email = "cap@test.invalid"
    cap = LIMITS["registered"]["monthly_full_words"]
    assert cap == 10_000, f"Expected registered cap=10000, got {cap}"

    # Seed one row with exactly cap words (answer of 10000 space-separated 'x')
    big_answer = " ".join(["x"] * cap)
    app.log_query(
        query="q", level="self-healing", answer=big_answer,
        email=cap_email, mode="full",
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(app.LOG_DB) as cx:
        used = monthly_full_words(cx, cap_email, now_iso)

    assert used >= cap, f"Expected used >= {cap}, got {used}"


def test_ceiling_decision_triggers_downgrade(tmp_path, monkeypatch):
    """When used >= cap, the downgrade to mode='brief' and _ceiling_hit=True fires."""
    app = _load_app(tmp_path, monkeypatch)
    from dashboard.chat_limits import monthly_full_words, LIMITS

    cap_email = "cap2@test.invalid"
    cap = LIMITS["registered"]["monthly_full_words"]

    big_answer = " ".join(["x"] * cap)
    app.log_query(
        query="q", level="self-healing", answer=big_answer,
        email=cap_email, mode="full",
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(app.LOG_DB) as cx:
        used = monthly_full_words(cx, cap_email, now_iso)

    # Simulate the decision block in chat()
    mode = "full"
    ceiling_hit = False
    pol = LIMITS["registered"]
    _cap = pol.get("monthly_full_words")
    if _cap is not None and used >= _cap:
        mode = "brief"
        ceiling_hit = True

    assert mode == "brief", "mode should be downgraded to 'brief' when cap hit"
    assert ceiling_hit is True, "_ceiling_hit should be True when cap hit"


# ── 2. Under-cap email: usage stays < 10 000 ─────────────────────────────────

def test_monthly_full_words_under_cap(tmp_path, monkeypatch):
    """An email with fewer than cap words is not downgraded."""
    app = _load_app(tmp_path, monkeypatch)
    from dashboard.chat_limits import monthly_full_words, LIMITS

    under_email = "under@test.invalid"
    cap = LIMITS["registered"]["monthly_full_words"]

    # Seed 500 words — well under 10 000
    small_answer = " ".join(["x"] * 500)
    app.log_query(
        query="q", level="self-healing", answer=small_answer,
        email=under_email, mode="full",
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(app.LOG_DB) as cx:
        used = monthly_full_words(cx, under_email, now_iso)

    assert used < cap, f"Expected used < {cap} for under-cap email, got {used}"

    # Decision: no downgrade
    mode = "full"
    ceiling_hit = False
    pol = LIMITS["registered"]
    _cap = pol.get("monthly_full_words")
    if _cap is not None and used >= _cap:
        mode = "brief"
        ceiling_hit = True

    assert mode == "full", "mode should stay 'full' when under cap"
    assert ceiling_hit is False


# ── 3. LIMITS structure for member tier ──────────────────────────────────────

def test_member_limits_no_hard_cap_but_has_flag():
    """Member tier has no monthly_full_words cap but has a flag_full_words threshold."""
    from dashboard.chat_limits import LIMITS
    assert LIMITS["member"]["monthly_full_words"] is None, (
        "member should have no hard cap (None)"
    )
    assert LIMITS["member"]["flag_full_words"] == 100_000, (
        "member flag threshold should be 100 000"
    )


# ── 4. monthly_full_words ignores brief-mode rows and out-of-window rows ──────

def test_monthly_full_words_only_counts_full_in_window(tmp_path, monkeypatch):
    """brief-mode rows and rows > 30 days old are excluded from the sum."""
    app = _load_app(tmp_path, monkeypatch)
    from dashboard.chat_limits import monthly_full_words
    from datetime import timedelta

    mixed_email = "mixed@test.invalid"
    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(days=40)).isoformat()

    # Insert directly: one recent full, one recent brief, one old full
    with sqlite3.connect(app.LOG_DB) as cx:
        cx.execute(
            "INSERT INTO query_log (ts, query, level, answer, session_id, email, "
            "name, ghl_contact_id, mode, user_agent, referer, "
            "extracted_image_data, image_count, word_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now.isoformat(), "q", "self-healing", "a", "", mixed_email,
             "", "", "full", "", "", "", 0, 300),
        )
        cx.execute(
            "INSERT INTO query_log (ts, query, level, answer, session_id, email, "
            "name, ghl_contact_id, mode, user_agent, referer, "
            "extracted_image_data, image_count, word_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now.isoformat(), "q", "self-healing", "a", "", mixed_email,
             "", "", "brief", "", "", "", 0, 9999),
        )
        cx.execute(
            "INSERT INTO query_log (ts, query, level, answer, session_id, email, "
            "name, ghl_contact_id, mode, user_agent, referer, "
            "extracted_image_data, image_count, word_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (old_ts, "q", "self-healing", "a", "", mixed_email,
             "", "", "full", "", "", "", 0, 9000),
        )
        cx.commit()

    used = monthly_full_words(cx, mixed_email, now.isoformat())
    assert used == 300, f"Expected only the 300 recent-full words, got {used}"
