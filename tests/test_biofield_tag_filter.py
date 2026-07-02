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


# The real people.tags CRM vocabulary a bare namespace-check missed.
_BROAD_CRM = ["aerai", "aweber email list opted in", "beta-personal-email", "budget <$300/mo",
              "chatbot", "chatbot-lead", "close_leads", "e4l-membership-lapsed-2026-06",
              "e4l.db import", "email bounced", "email paramedic - email list growth", "fireside",
              "household:gonsai", "portal-invite", "practitioner-portal",
              "topic-32-my-health-focus", "pb:od", "pb:rn", "pb:zyto", "pb:cert"]
# free-text health terms that must NOT be dropped (no collision with CRM keywords/prefixes)
_HEALTH_FREETEXT = ["Inflammation", "Heavy metals", "Hashimoto's", "Acidic", "Liver",
                    "lead toxicity", "mercury toxicity", "beta-carotene", "beta-glucan",
                    "chronic fatigue", "pb:migraine", "pb:wet-amd", "pb:fatty-liver"]


def test_broad_crm_vocabulary_filtered_health_kept():
    for t in _BROAD_CRM:
        assert is_operational_tag(t), f"{t!r} (CRM) should be filtered"
    for t in _HEALTH_FREETEXT:
        assert not is_operational_tag(t), f"{t!r} (health) must be kept"


def test_prettify_condition_keeps_medical_acronyms():
    from dashboard.biofield_profile import prettify_condition as p, clean_health_tag as c
    assert p("pb:wet-amd") == "Wet AMD"
    assert p("psc-cataract") == "PSC Cataract"
    assert p("pb:ebv") == "EBV"
    assert p("fatty-liver") == "Fatty Liver"
    assert p("dry-eye") == "Dry Eye"
    assert c("pb:dry-amd") == "Dry AMD"          # reveals use the same helper


def test_mine_keeps_freetext_health_drops_broad_crm():
    profile = {"tags": ["chatbot-lead", "aweber email list opted in", "budget <$300/mo",
                        "pb:migraine", "Inflammation"],
               "conditions": "Hashimoto's; Eczema", "body_systems": ["Liver"]}
    out = mine_profile_stresses(profile, lambda t: [])
    assert set(out) == {"pb:migraine", "Inflammation", "Hashimoto's", "Eczema", "Liver"}


def test_mine_drops_operational_tags_keeps_clinical():
    profile = {"tags": ["type:client", "consent:opted-in", "begin", "pb:migraine", "Inflammation"],
               "conditions": "Hashimoto's"}
    out = mine_profile_stresses(profile, lambda t: [])
    assert set(out) == {"pb:migraine", "Inflammation", "Hashimoto's"}


def test_tag_labels_cleaned_for_display():
    cx = sqlite3.connect(":memory:")
    st.init_stress_tables(cx)
    # profile tag-mining stored these with JSON-quote artifacts + pb: namespace
    for code, label in [("pb:migraine", '"pb:migraine"'),
                        ("pb:fatty-liver", '"pb:fatty-liver"'),
                        ("pb:wet-amd", '"pb:wet-amd"')]:
        cx.execute("INSERT INTO biofield_auth_stress(test_id,code,label,source,balance,"
                   "manual_balanced,created_at,updated_at) VALUES(9,?,?,'tag','required',0,'','')",
                   (code, label))
    cx.commit()
    labels = {s["label"] for s in st.list_stresses(cx, "9", [])["active"]}
    assert labels == {"Migraine", "Fatty Liver", "Wet AMD"}   # acronym kept uppercase
    assert not any('"' in l or l.lower().startswith("pb:") for l in labels)


def test_scan_and_voice_labels_left_untouched():
    cx = sqlite3.connect(":memory:")
    st.init_stress_tables(cx)
    st.add_stress(cx, "9", "Membrane instability", source="scan", balance="required")
    st.add_stress(cx, "9", "always feels wired at night", source="voice", balance="required")
    labels = {s["label"] for s in st.list_stresses(cx, "9", [])["active"]}
    assert labels == {"Membrane instability", "always feels wired at night"}


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
    assert "Migraine" in labels                  # clinical tag kept (pb: cleaned off)
    assert "type:client" not in labels           # operational tags gone
    assert "consent:opted-in" not in labels
