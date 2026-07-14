import sqlite3
import pytest
from dashboard import data_sharing as ds

def _cx():
    cx = sqlite3.connect(":memory:")
    ds.init_data_sharing_tables(cx)
    return cx

def test_expand_toggles_and_tier():
    # Toggle 1 only -> chat/improve_ai -> tier 1
    grants, attr = ds.expand_toggles({"improve_ai_chat": True})
    assert ("chat", "improve_ai") in grants
    assert attr == "anonymized"
    assert ds.derive_tier(grants, attr) == 1

    # Toggle 2 -> scans+results research/improve, anonymized -> tier 2
    grants, attr = ds.expand_toggles({"research_results": True})
    assert ("scans", "research") in grants and ("results", "improve_ai") in grants
    assert ds.derive_tier(grants, attr) == 2

    # Toggle 3 -> scans/results marketing attributed -> tier 3
    grants, attr = ds.expand_toggles({"share_story": True})
    assert attr == "attributed"
    assert ("scans", "marketing") in grants
    assert ds.derive_tier(grants, attr) == 3

    # Toggle 4 -> video attributed -> tier 4
    grants, attr = ds.expand_toggles({"video_testimonial": True})
    assert ("video", "marketing") in grants
    assert ds.derive_tier(grants, attr) == 4

    # Empty -> tier 0
    grants, attr = ds.expand_toggles({})
    assert ds.derive_tier(grants, attr) == 0

def test_set_and_revoke_consent_persists():
    cx = _cx()
    st = ds.set_consent(cx, "A@Ex.com", {"research_results": True})
    assert st["tier"] == 2
    assert ds.get_consent(cx, "a@ex.com")["tier"] == 2
    # Turning it off revokes (prospective): tier drops, grant row keeps history
    st2 = ds.set_consent(cx, "a@ex.com", {})
    assert st2["tier"] == 0
    assert ds.get_consent(cx, "a@ex.com")["grants"] == []
    revoked = cx.execute("SELECT COUNT(*) FROM member_data_sharing_grants "
                         "WHERE email='a@ex.com' AND revoked_at IS NOT NULL").fetchone()[0]
    assert revoked >= 1  # history preserved, not deleted
