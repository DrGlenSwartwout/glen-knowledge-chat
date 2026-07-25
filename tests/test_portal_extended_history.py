import sqlite3

from dashboard import portal_extended_history as eh


def test_extended_history_roundtrip_preserves_category_keys():
    cx = sqlite3.connect(":memory:")
    eh.save(cx, "Client@Example.com", {
        "surgeries_yes": True,
        "surgeries_text": "Appendectomy; reason: appendicitis; age 12",
        "toxins_yes": True,
        "toxins_text": "Workplace solvent exposure; 2018-2020",
        "family_history_yes": True,
        "family_history_text": "Maternal grandmother; diabetes; onset about 60",
        "sleep_yes": False,
        "sleep_text": "must not persist",
    })
    got = eh.get(cx, "client@example.com")
    answers = got["answers"]
    assert answers["surgeries_text"].startswith("Appendectomy")
    assert answers["toxins_text"].startswith("Workplace")
    assert answers["family_history_text"].startswith("Maternal grandmother")
    assert answers["sleep_yes"] is False
    assert answers["sleep_text"] == ""
    assert eh.has(cx, "CLIENT@example.com") is True
