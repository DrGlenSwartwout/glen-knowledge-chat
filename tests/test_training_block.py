"""Training tab block — the 12-module ASH curriculum with per-practitioner progress."""
from dashboard import practitioner_portal as pp


def test_training_block_lists_twelve_modules_with_progress():
    b = pp.training_block(3)
    assert b["modules_completed"] == 3
    assert b["modules_total"] == 12
    assert len(b["modules"]) == 12
    assert b["modules"][0] == {"n": 1, "title": "Body", "subtitle": "5 States of Matter", "complete": True}
    assert b["modules"][2]["complete"] is True    # module 3 done
    assert b["modules"][3]["complete"] is False   # module 4 not yet
    assert b["modules"][6]["subtitle"] == "5 Embryological Tissue Layers"  # module 7
    assert b["modules"][11]["title"] == "Prognosis"


def test_training_block_clamps_and_handles_none():
    assert pp.training_block(0)["modules_completed"] == 0
    assert all(not m["complete"] for m in pp.training_block(0)["modules"])
    assert pp.training_block(99)["modules_completed"] == 12
    assert all(m["complete"] for m in pp.training_block(99)["modules"])
    assert pp.training_block(None)["modules_completed"] == 0
