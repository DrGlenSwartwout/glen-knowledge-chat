"""Operational CRM/marketing tags must not surface as biofield stresses.

Tags like type:client, consent:opted-in, source:*, reengagement:*, pb:cert,
"email engage 0-7 days", "concierge", "begin" describe pipeline/lifecycle state,
not health status. They are filtered out of both profile mining and the stress
panel. Clinical tags (pb:migraine, Inflammation) are kept.
"""
import sqlite3

from dashboard.biofield_profile import mine_profile_stresses, is_operational_tag
from dashboard import biofield_stress as st


DROP = ["type:client", "type:practitioner", "consent:opted-in", "state:has-scanned",
        "source:e4l-portal-import-2026-05-22", "pract:eaf8ccf5", "reengagement:practitioner",
        "e4l:interested", "e4l account", "nes client", "begin", "concierge",
        "email engage 0-7 days", "ash certification course", "certification masterclass replay",
        "pb:cert", "pb:ash-certification-1"]
KEEP = ["pb:migraine", "pb:fatty-liver", "pb:wet-amd", "Inflammation", "Heavy metals",
        "Hashimoto's", "Acidic"]


def test_is_operational_tag_classifies():
    for t in DROP:
        assert is_operational_tag(t), f"{t!r} should be filtered"
    for t in KEEP:
        assert not is_operational_tag(t), f"{t!r} should be kept"


def test_mine_drops_operational_tags_keeps_clinical():
    profile = {"tags": ["type:client", "consent:opted-in", "begin", "pb:migraine", "Inflammation"],
               "conditions": "Hashimoto's"}
    out = mine_profile_stresses(profile, lambda t: [])
    assert set(out) == {"pb:migraine", "Inflammation", "Hashimoto's"}


def test_list_stresses_hides_operational_tag_rows():
    cx = sqlite3.connect(":memory:")
    st.init_stress_tables(cx)
    # a real health stress from scan + two operational CRM tags mined into the panel
    st.add_stress(cx, "9", "Membrane", source="scan", balance="required")
    st.add_stress(cx, "9", "type:client", source="tag", balance="required")
    st.add_stress(cx, "9", "consent:opted-in", source="tag", balance="required")
    st.add_stress(cx, "9", "pb:migraine", source="tag", balance="required")
    data = st.list_stresses(cx, "9", [])
    labels = {s["label"] for s in data["active"] + data["balanced"]}
    assert "Membrane" in labels
    assert "pb:migraine" in labels               # clinical tag kept
    assert "type:client" not in labels           # operational tags gone
    assert "consent:opted-in" not in labels
