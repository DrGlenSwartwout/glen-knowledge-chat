import sqlite3
from dashboard import pay_it_forward as pif
from dashboard import points
from dashboard import referrals


def _cx():
    cx = sqlite3.connect(":memory:")
    points.init_points_table(cx)
    return cx


def test_award_milestone_credits_points():
    cx = _cx()
    pif.award_milestone(cx, "Member@X.com", milestone_key="program_complete_1")
    assert points.balance(cx, "member@x.com") == pif.MILESTONE_REWARD_CENTS


def test_award_milestone_idempotent_per_key():
    cx = _cx()
    pif.award_milestone(cx, "m@x.com", milestone_key="scan_improved_q1")
    pif.award_milestone(cx, "m@x.com", milestone_key="scan_improved_q1")
    assert points.balance(cx, "m@x.com") == pif.MILESTONE_REWARD_CENTS


def test_award_milestone_distinct_keys_stack():
    cx = _cx()
    pif.award_milestone(cx, "m@x.com", milestone_key="a")
    pif.award_milestone(cx, "m@x.com", milestone_key="b")
    assert points.balance(cx, "m@x.com") == pif.MILESTONE_REWARD_CENTS * 2


def test_award_milestone_ignores_blank():
    cx = _cx()
    pif.award_milestone(cx, "", milestone_key="x")
    pif.award_milestone(cx, "m@x.com", milestone_key="")
    assert points.balance(cx, "m@x.com") == 0


def _seed_redemption(cx, owner, referee):
    referrals.init_tables(cx)
    referrals.record_redemption(cx, "CODE", owner, referee, order_ref=f"o:{referee}")


def test_chain_summary_counts_two_levels():
    cx = _cx()
    # A gifted B and C (L1); B gifted D (L2)
    _seed_redemption(cx, "a@x.com", "b@x.com")
    _seed_redemption(cx, "a@x.com", "c@x.com")
    _seed_redemption(cx, "b@x.com", "d@x.com")
    s = pif.chain_summary(cx, "A@X.com")
    assert s["l1"] == 2
    assert s["l2"] == 1
    assert s["reached"] == 3
    assert s["levels"] == [2, 1]


def test_chain_summary_empty_for_unknown():
    cx = _cx()
    referrals.init_tables(cx)
    s = pif.chain_summary(cx, "nobody@x.com")
    assert s == {"reached": 0, "l1": 0, "l2": 0, "levels": []}


def test_chain_summary_excludes_self_and_dedupes():
    cx = _cx()
    _seed_redemption(cx, "a@x.com", "b@x.com")
    _seed_redemption(cx, "b@x.com", "a@x.com")  # cycle back to seed: must not recount A
    s = pif.chain_summary(cx, "a@x.com")
    assert s["reached"] == 1
    assert s["l1"] == 1
    assert s["l2"] == 0
    assert s["levels"] == [1, 0]


def test_healer_level_thresholds():
    assert pif.healer_level(0) == 1
    assert pif.healer_level(2) == 1
    assert pif.healer_level(3) == 2
    assert pif.healer_level(9) == 2
    assert pif.healer_level(10) == 3


def test_chain_recipients_masked_name_and_date():
    cx = _cx()
    referrals.init_tables(cx)
    cx.execute("CREATE TABLE people (email TEXT UNIQUE, first_name TEXT, last_name TEXT, name TEXT)")
    cx.execute("INSERT INTO people (email, first_name, last_name, name) "
               "VALUES ('b@x.com','Sarah','Hill','Sarah Hill')")
    cx.commit()
    referrals.record_redemption(cx, "GIFT1", "a@x.com", "b@x.com", "o1")
    out = pif.chain_recipients(cx, "A@x.com")  # case-insensitive owner
    assert len(out) == 1
    assert out[0]["name"] == "Sarah H."
    assert out[0]["redeemed_at"]  # non-empty ISO timestamp


def test_chain_recipients_excludes_l2():
    cx = _cx()
    referrals.init_tables(cx)
    cx.execute("CREATE TABLE people (email TEXT UNIQUE, first_name TEXT, last_name TEXT, name TEXT)")
    cx.execute("INSERT INTO people (email, first_name, last_name, name) "
               "VALUES ('b@x.com','Bob','Brown','Bob Brown')")
    cx.commit()
    referrals.record_redemption(cx, "C1", "a@x.com", "b@x.com", "o1")  # A->B (L1)
    referrals.record_redemption(cx, "C2", "b@x.com", "c@x.com", "o2")  # B->C (L2)
    out = pif.chain_recipients(cx, "a@x.com")
    assert len(out) == 1
    assert out[0]["name"] == "Bob B."


def test_chain_recipients_a_friend_fallback():
    cx = _cx()
    referrals.init_tables(cx)  # no people table created
    referrals.record_redemption(cx, "C1", "a@x.com", "nameless@x.com", "o1")
    out = pif.chain_recipients(cx, "a@x.com")
    assert out[0]["name"] == "A friend"
    assert "@" not in out[0]["name"]


def test_chain_recipients_product_from_code():
    cx = _cx()
    referrals.init_tables(cx)
    cx.execute("CREATE TABLE coupons (code TEXT PRIMARY KEY, product_slug TEXT)")
    cx.execute("INSERT INTO coupons (code, product_slug) VALUES ('GIFT9','neuro-magnesium')")
    cx.commit()
    referrals.record_redemption(cx, "GIFT9", "a@x.com", "b@x.com", "o1")
    out = pif.chain_recipients(cx, "a@x.com")
    assert out[0]["product"] == "neuro-magnesium"


def test_chain_recipients_missing_coupon_blank_product():
    cx = _cx()
    referrals.init_tables(cx)  # no coupons table
    referrals.record_redemption(cx, "NOPE", "a@x.com", "b@x.com", "o1")
    out = pif.chain_recipients(cx, "a@x.com")
    assert out[0]["product"] == ""


def test_chain_recipients_newest_first_and_limit():
    cx = _cx()
    referrals.init_tables(cx)
    for i in range(3):
        cx.execute(
            "INSERT INTO referral_redemptions (referee_email, code, owner_email, order_ref, created_at) "
            "VALUES (?,?,?,?,?)",
            (f"r{i}@x.com", f"C{i}", "a@x.com", f"o{i}", f"2026-06-0{i+1}T00:00:00"))
    cx.commit()
    out = pif.chain_recipients(cx, "a@x.com", limit=2)
    assert len(out) == 2
    assert out[0]["redeemed_at"] == "2026-06-03T00:00:00"  # newest first
    assert out[1]["redeemed_at"] == "2026-06-02T00:00:00"


def test_chain_recipients_no_cross_owner_leak():
    cx = _cx()
    referrals.init_tables(cx)
    referrals.record_redemption(cx, "C1", "other@x.com", "b@x.com", "o1")
    assert pif.chain_recipients(cx, "a@x.com") == []


def test_masked_name_first_only():
    cx = _cx()
    cx.execute("CREATE TABLE people (email TEXT UNIQUE, first_name TEXT, last_name TEXT, name TEXT)")
    cx.execute("INSERT INTO people (email, first_name, last_name, name) VALUES ('s@x.com','Sam','','')")
    cx.commit()
    assert pif._masked_name(cx, "s@x.com") == "Sam"


def test_masked_name_from_name_column():
    cx = _cx()
    cx.execute("CREATE TABLE people (email TEXT UNIQUE, first_name TEXT, last_name TEXT, name TEXT)")
    cx.execute("INSERT INTO people (email, first_name, last_name, name) VALUES ('s@x.com','','','Sam Smith')")
    cx.commit()
    assert pif._masked_name(cx, "s@x.com") == "Sam S."


def test_masked_name_never_leaks_email_in_name_column():
    cx = _cx()
    cx.execute("CREATE TABLE people (email TEXT UNIQUE, first_name TEXT, last_name TEXT, name TEXT)")
    cx.execute("INSERT INTO people (email, first_name, last_name, name) VALUES ('s@x.com','','','bob@example.com')")
    cx.commit()
    assert pif._masked_name(cx, "s@x.com") == "A friend"


def _gift_note(cx, *, owner, referee, body="helps my sleep", consent=1, compliance=8, status="pending"):
    from dashboard import product_reviews as pr
    rid = pr.upsert_review(cx, "neuro-magnesium", referee, "Bob", 0, body=body,
                           kind="gift", consent_public=consent, source_tag="gift",
                           gift_owner_email=owner)
    pr.set_scores(cx, rid, compliance=compliance)
    if status != "pending":
        pr.set_status(cx, rid, status)
    return rid


def test_giver_note_returned_when_all_gates_pass():
    cx = _cx()
    _gift_note(cx, owner="a@x.com", referee="b@x.com")
    assert pif._giver_note(cx, "A@x.com", "B@x.com") == "helps my sleep"


def test_giver_note_blank_without_consent():
    cx = _cx()
    _gift_note(cx, owner="a@x.com", referee="b@x.com", consent=0)
    assert pif._giver_note(cx, "a@x.com", "b@x.com") == ""


def test_giver_note_blank_below_compliance_threshold():
    cx = _cx()
    _gift_note(cx, owner="a@x.com", referee="b@x.com", compliance=6)  # < 7
    assert pif._giver_note(cx, "a@x.com", "b@x.com") == ""


def test_giver_note_blank_when_rejected():
    cx = _cx()
    _gift_note(cx, owner="a@x.com", referee="b@x.com", status="rejected")
    assert pif._giver_note(cx, "a@x.com", "b@x.com") == ""


def test_giver_note_blank_for_other_owner():
    cx = _cx()
    _gift_note(cx, owner="someoneelse@x.com", referee="b@x.com")
    assert pif._giver_note(cx, "a@x.com", "b@x.com") == ""  # not this giver's note


def test_giver_note_blank_when_no_row():
    cx = _cx()
    from dashboard import product_reviews as pr
    pr.init_table(cx)  # table exists but empty
    assert pif._giver_note(cx, "a@x.com", "b@x.com") == ""


def test_chain_recipients_includes_note():
    cx = _cx()
    referrals.init_tables(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS people (email TEXT UNIQUE, first_name TEXT, last_name TEXT, name TEXT)")
    cx.execute("INSERT INTO people (email, first_name, last_name, name) VALUES ('b@x.com','Barbara','','Barbara')")
    cx.commit()
    referrals.record_redemption(cx, "C1", "a@x.com", "b@x.com", "o1")
    _gift_note(cx, owner="a@x.com", referee="b@x.com", body="changed my mornings")
    out = pif.chain_recipients(cx, "a@x.com")
    assert out[0]["note"] == "changed my mornings"
    # a recipient with no qualifying note -> note is ""
    referrals.record_redemption(cx, "C2", "a@x.com", "c@x.com", "o2")
    out2 = pif.chain_recipients(cx, "a@x.com")
    notes = {e["name"]: e["note"] for e in out2}  # names masked; just assert one "" present
    assert "" in notes.values()
