import sqlite3
from dashboard import product_reviews as pr


def test_upsert_records_gift_owner_email():
    cx = sqlite3.connect(":memory:")
    rid = pr.upsert_review(cx, "neuro-magnesium", "b@x.com", "Bob", 0, body="helped my sleep",
                           kind="gift", consent_public=1, source_tag="gift",
                           gift_owner_email="A@x.com")
    row = pr.get_review(cx, rid)
    assert row["kind"] == "gift"
    assert row["gift_owner_email"] == "a@x.com"   # normalized lower
    assert row["product_slug"] == "neuro-magnesium"
    assert row["consent_public"] == 1


def test_upsert_default_gift_owner_blank_for_normal_reviews():
    cx = sqlite3.connect(":memory:")
    rid = pr.upsert_review(cx, "_results", "c@x.com", "Cy", 5, body="great")
    row = pr.get_review(cx, rid)
    assert (row["gift_owner_email"] or "") == ""


def test_aggregate_excludes_gift_notes():
    cx = sqlite3.connect(":memory:")
    # one approved 5-star rated review
    rid1 = pr.upsert_review(cx, "neuro-magnesium", "rater@x.com", "Ray", 5, body="great")
    pr.set_status(cx, rid1, "approved")
    # one approved gift note (rating 0)
    rid2 = pr.upsert_review(cx, "neuro-magnesium", "gift@x.com", "Gigi", 0, body="helped",
                            kind="gift", consent_public=1, gift_owner_email="a@x.com")
    pr.set_status(cx, rid2, "approved")
    agg = pr.aggregate(cx, "neuro-magnesium")
    assert agg["count"] == 1      # gift note not counted
    assert agg["avg"] == 5.0      # gift rating 0 not averaged in
