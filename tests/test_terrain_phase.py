"""Consumer-facing names for the 5 terrain phases (the BSI 'phase P' reading)."""
from dashboard.terrain_phase import PHASE_NAMES, phase_num, phase_name, phase_display


def test_phase_names_cover_1_to_5():
    assert PHASE_NAMES == {
        1: "Terrain Revive",
        2: "Terrain Repair",
        3: "Terrain Renew",
        4: "Terrain Refresh",
        5: "Terrain Relief",
    }


def test_phase_num_coerces_and_bounds():
    assert phase_num(4) == 4
    assert phase_num("4") == 4
    assert phase_num(" 2 ") == 2
    assert phase_num(0) is None
    assert phase_num(6) is None
    assert phase_num(None) is None
    assert phase_num("") is None
    assert phase_num("phase 3") is None  # not a bare int -> caller must pass the number


def test_phase_name_and_display():
    assert phase_name(4) == "Terrain Refresh"
    assert phase_name("1") == "Terrain Revive"
    assert phase_name(None) == ""
    assert phase_name(9) == ""
    assert phase_display(4) == "Terrain Refresh (Phase 4)"
    assert phase_display(None) == ""
    assert phase_display(0) == ""
