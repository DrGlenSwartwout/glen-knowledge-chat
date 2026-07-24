"""Test condition recommendation source registration, the glaucoma triage
decision table, and the triage -> seed writer (replace-on-retriage)."""
import sqlite3

from dashboard import recommendation_sources as rs
from dashboard import condition_triage as ct
from dashboard import condition_programs as cp
from dashboard import recommendation_events as re_events


def test_condition_source_registered():
    """Condition source must be registered with clinical kind."""
    assert rs.known_source("condition")
    s = rs.RECOMMENDATION_SOURCES["condition"]
    assert s["kind"] == "clinical" and "history" in s["label"].lower()


N, E = "glaucoma-normal-iop", "glaucoma-elevated-iop"


def rp(**a):
    return ct.resolve_programs("glaucoma", a)


def test_decision_table():
    assert rp(iop_od=17, iop_os=18) == [N]                          # <20 normal
    assert rp(iop_od=24, iop_os=19) == [E]                          # >=22 elevated (higher eye)
    assert rp(iop_od=20, iop_os=21, field_loss=False) == [E, N]     # borderline -> both
    assert rp(iop_od=20, iop_os=21, field_loss=True) == [E]         # borderline + field loss -> elevated
    assert rp(iop_od=15, iop_os=16, on_meds=True) == [E, N]         # on meds -> both (lead E)
    assert rp(category="normal") == [N]
    assert rp(category="elevated") == [E]
    assert rp(category="not sure", field_loss=True) == [E]
    assert ct.resolve_programs("cataract", {"iop_od": 30}) == []    # pilot: glaucoma only


def test_single_program_conditions_resolve_with_no_triage_questions():
    assert ct.resolve_programs("dry-eye", {}) == ["dry-eye"]
    assert ct.resolve_programs("vision-improvement", {}) == ["vision-improvement"]
    # answers are irrelevant for a single-program condition
    assert ct.resolve_programs("dry-eye", {"iop_od": 30}) == ["dry-eye"]


def test_multi_subtype_conditions_still_return_empty():
    assert ct.resolve_programs("cataract", {}) == []
    assert ct.resolve_programs("macular", {}) == []


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cp.init_table(cx)
    re_events.init_recommendation_events(cx)
    ct.init_table(cx)
    return cx


def _seed_fake_programs(cx):
    cp.upsert(cx, "glaucoma-elevated-iop", "Glaucoma - Elevated IOP", False,
              [{"slug": "neuroprotect", "name": "Neuroprotect"},
               {"slug": "iop-syntropy", "name": "IOP Syntropy"}])
    cp.upsert(cx, "glaucoma-normal-iop", "Glaucoma - Normal IOP", False,
              [{"slug": "neuroprotect", "name": "Neuroprotect"},
               {"slug": "ocuflow-daytime", "name": "OcuFlow Daytime"}])


def test_seed_from_triage_writes_and_replaces_on_retriage():
    cx = _cx()
    _seed_fake_programs(cx)

    result = ct.seed_from_triage(cx, "a@x.com", "glaucoma", {"iop_od": 25})
    assert result["programs"] == [E]
    assert set(result["seeded"]) == {"neuroprotect", "iop-syntropy"}

    prods = {p["product_key"] for p in re_events.product_sources(cx, "a@x.com")}
    assert "iop-syntropy" in prods
    assert "ocuflow-daytime" not in prods

    # Re-triage with a normal-range answer REPLACES the elevated seed.
    result2 = ct.seed_from_triage(cx, "a@x.com", "glaucoma", {"iop_od": 17})
    assert result2["programs"] == [N]
    assert set(result2["seeded"]) == {"neuroprotect", "ocuflow-daytime"}

    prods2 = {p["product_key"] for p in re_events.product_sources(cx, "a@x.com")}
    assert "iop-syntropy" not in prods2          # prior condition seed cleared
    assert "ocuflow-daytime" in prods2

    stored = ct.get_triage(cx, "a@x.com", "glaucoma")
    assert stored["resolved_programs"] == [N]
    assert stored["iop_od"] == "17"


def test_seed_from_triage_skips_do_not_recommend_slugs():
    """Defensive filter: even if a program's items_json somehow contains a
    do-not-recommend slug, seed_from_triage must never record it."""
    from dashboard.related_products import DO_NOT_RECOMMEND
    dnr_slug = next(iter(DO_NOT_RECOMMEND))
    cx = _cx()
    cp.upsert(cx, "glaucoma-normal-iop", "Glaucoma - Normal IOP", False,
              [{"slug": dnr_slug, "name": "DNR item"},
               {"slug": "neuroprotect", "name": "Neuroprotect"}])

    result = ct.seed_from_triage(cx, "b@x.com", "glaucoma", {"category": "normal"})

    assert dnr_slug not in result["seeded"]
    assert "neuroprotect" in result["seeded"]
    prods = {p["product_key"] for p in re_events.product_sources(cx, "b@x.com")}
    assert dnr_slug not in prods
    assert "neuroprotect" in prods


def _seed_dry_eye_program(cx):
    cp.upsert(cx, "dry-eye", "Dry Eye", False,
              [{"slug": "aces-eye-drops", "name": "ACES Eye Drops"},
               {"slug": "wholomega", "name": "WholOmega", "dose": "4 capsules/day"}],
              modifiers=[
                  {"when": "aqueous_deficiency", "action": "add",
                   "source": "client-reported", "client_default": True,
                   "items": [{"slug": "moisturize", "name": "Moisturize"}]},
                  {"when": "severe", "action": "add",
                   "source": "client-reported", "client_default": False,
                   "items": [{"slug": "moisture-eyes-night-oil",
                              "name": "Moisture Eyes Night Oil"}]},
              ])


def test_dry_eye_triage_seeds_moisturize_when_sjogrens_reported():
    cx = _cx()
    _seed_dry_eye_program(cx)
    result = ct.seed_from_triage(cx, "de-a@x.com", "dry-eye", {"sjogrens": True})
    assert result["programs"] == ["dry-eye"]
    assert "moisturize" in result["seeded"]
    assert "moisture-eyes-night-oil" not in result["seeded"]


def test_dry_eye_triage_seeds_moisturize_by_default_when_unanswered():
    cx = _cx()
    _seed_dry_eye_program(cx)
    result = ct.seed_from_triage(cx, "de-b@x.com", "dry-eye", {})
    assert "moisturize" in result["seeded"]


def test_dry_eye_triage_explicit_no_aqueous_answer_skips_moisturize():
    cx = _cx()
    _seed_dry_eye_program(cx)
    result = ct.seed_from_triage(cx, "de-c@x.com", "dry-eye",
                                  {"not_enough_tears": False})
    assert "moisturize" not in result["seeded"]


def test_dry_eye_triage_severe_seeds_night_oil():
    cx = _cx()
    _seed_dry_eye_program(cx)
    result = ct.seed_from_triage(cx, "de-d@x.com", "dry-eye", {"severe": True})
    assert "moisture-eyes-night-oil" in result["seeded"]
    assert "moisturize" in result["seeded"]  # default true, unanswered


def test_dry_eye_triage_without_severe_skips_night_oil():
    cx = _cx()
    _seed_dry_eye_program(cx)
    result = ct.seed_from_triage(cx, "de-e@x.com", "dry-eye", {})
    assert "moisture-eyes-night-oil" not in result["seeded"]


def test_non_numeric_med_count_does_not_raise_and_stores_zero():
    """int(med_count) must be guarded -- a non-numeric string (e.g. free-text
    entry) must not raise ValueError; it should be stored as 0."""
    cx = _cx()
    _seed_fake_programs(cx)
    result = ct.seed_from_triage(cx, "c@x.com", "glaucoma",
                                  {"iop_od": 25, "med_count": "two"})
    assert result["programs"] == [E]
    stored = ct.get_triage(cx, "c@x.com", "glaucoma")
    assert stored["med_count"] == 0
