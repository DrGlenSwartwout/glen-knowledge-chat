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


def test_single_program_conditions_resolve_with_no_triage_questions():
    assert ct.resolve_programs("dry-eye", {}) == ["dry-eye"]
    assert ct.resolve_programs("retinitis-pigmentosa", {}) == ["retinitis-pigmentosa"]
    assert ct.resolve_programs("diabetic-retinopathy", {}) == ["diabetic-retinopathy"]
    assert ct.resolve_programs("vision-improvement", {}) == ["vision-improvement"]
    assert ct.resolve_programs("other", {"other_condition": "Uveitis"}) == []


def test_other_condition_free_text_roundtrips():
    cx = _cx()
    result = ct.seed_from_triage(
        cx, "other@x.com", "other", {"other_condition": "Uveitis"})
    assert result["programs"] == []
    stored = ct.get_triage(cx, "other@x.com", "other")
    assert stored["other_condition"] == "Uveitis"
    # answers are irrelevant for a single-program condition
    assert ct.resolve_programs("dry-eye", {"iop_od": 30}) == ["dry-eye"]


# ---------------------------------------------------------------------------
# Cataract sub-type triage (Dr. Glen's confirmed decision table)
# ---------------------------------------------------------------------------

def rc(**a):
    return ct.resolve_programs("cataract", a)


def test_cataract_told_psc_over_50_returns_both():
    assert rc(cataract_type="psc", age=51) == ["psc-cataract", "senile-cataract"]


def test_cataract_told_psc_at_boundary_50_returns_psc_only():
    # "age > 50" is strict -- exactly 50 stays on the psc-only side of rule 1.
    assert rc(cataract_type="psc", age=50) == ["psc-cataract"]


def test_cataract_told_psc_under_50_returns_psc_only():
    assert rc(cataract_type="psc", age=49) == ["psc-cataract"]


def test_cataract_told_psc_under_50_with_risk_flag_still_psc_only():
    # Rule 1 does not change routing on risk flags -- only rule 3 (not sure) does.
    assert rc(cataract_type="psc", age=40, steroids=True) == ["psc-cataract"]


def test_cataract_told_senile_returns_senile_only():
    assert rc(cataract_type="senile", age=70) == ["senile-cataract"]
    assert rc(cataract_type="senile", age=30) == ["senile-cataract"]  # type told wins over age


def test_cataract_not_sure_under_50_returns_psc():
    assert rc(age=49) == ["psc-cataract"]
    assert rc(cataract_type="not_sure", age=49) == ["psc-cataract"]


def test_cataract_not_sure_boundary_50_no_risk_returns_senile():
    assert rc(age=50) == ["senile-cataract"]


def test_cataract_not_sure_over_50_no_risk_returns_senile():
    assert rc(age=60) == ["senile-cataract"]


def test_cataract_not_sure_unknown_age_no_risk_returns_senile():
    assert rc() == ["senile-cataract"]


def test_cataract_not_sure_over_50_with_risk_flag_returns_both():
    # Coordinator correction: 50+/unknown age WITH any risk flag -> both.
    assert rc(age=60, steroids=True) == ["psc-cataract", "senile-cataract"]
    assert rc(age=60, diabetes=True) == ["psc-cataract", "senile-cataract"]
    assert rc(age=60, inflammation=True) == ["psc-cataract", "senile-cataract"]
    assert rc(age=60, radiation=True) == ["psc-cataract", "senile-cataract"]
    assert rc(age=60, atopy=True) == ["psc-cataract", "senile-cataract"]


def test_cataract_not_sure_over_50_no_risk_flags_returns_senile_only():
    assert rc(age=60) == ["senile-cataract"]


def test_cataract_not_sure_unknown_age_with_risk_flag_returns_both():
    assert rc(steroids=True) == ["psc-cataract", "senile-cataract"]


def test_cataract_not_sure_under_50_with_risk_flag_still_psc_only():
    # Under-50 branch (rule 3) is unaffected by risk flags -- unchanged.
    assert rc(age=40, steroids=True) == ["psc-cataract"]


# ---------------------------------------------------------------------------
# Macular sub-type triage (Dr. Glen's confirmed decision table)
# ---------------------------------------------------------------------------

def rm(**a):
    return ct.resolve_programs("macular", a)


def test_macular_wet_type_returns_wet_amd():
    assert rm(amd_type="wet") == ["wet-amd"]


def test_macular_injections_returns_wet_amd_even_if_dry_type_absent():
    assert rm(injections=True) == ["wet-amd"]


def test_macular_dry_type_returns_dry_amd():
    assert rm(amd_type="dry") == ["dry-amd"]


def test_macular_not_sure_with_distortion_returns_wet_amd():
    assert rm(distortion=True) == ["wet-amd"]


def test_macular_not_sure_without_distortion_returns_dry_amd():
    assert rm() == ["dry-amd"]
    assert rm(distortion=False) == ["dry-amd"]


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


# ---------------------------------------------------------------------------
# Brunescent modifier: threaded from the "yellow_vision" triage answer into
# senile-cataract's client-reported "brunescent" modifier, default off.
# ---------------------------------------------------------------------------

def _seed_senile_with_brunescent_modifier(cx):
    cp.upsert(cx, "senile-cataract", "Senile (Age-Related) Cataract", False,
              [{"slug": "golden-book", "name": "Golden Book"},
               {"slug": "clarity", "name": "Clarity"}],
              [{"when": "brunescent", "action": "add", "source": "client-reported",
                "client_default": False,
                "items": [{"slug": "lens-zyme", "name": "Lens-Zyme Brunescence Buster"}]}])


def test_brunescent_default_off_lens_zyme_not_seeded():
    cx = _cx()
    _seed_senile_with_brunescent_modifier(cx)
    result = ct.seed_from_triage(cx, "d@x.com", "cataract",
                                  {"cataract_type": "senile", "age": 70})
    assert "lens-zyme" not in result["seeded"]
    assert "golden-book" in result["seeded"]


def test_brunescent_on_via_yellow_vision_answer_seeds_lens_zyme():
    cx = _cx()
    _seed_senile_with_brunescent_modifier(cx)
    result = ct.seed_from_triage(cx, "e@x.com", "cataract",
                                  {"cataract_type": "senile", "age": 70,
                                   "yellow_vision": True})
    assert "lens-zyme" in result["seeded"]
    assert "golden-book" in result["seeded"]
    stored = ct.get_triage(cx, "e@x.com", "cataract")
    assert stored["yellow_vision"] is True


def test_brunescent_default_false_when_yellow_vision_absent():
    stored_facts = ct.resolve_client_facts("cataract", {})
    assert stored_facts.get("brunescent") is False


def test_resolve_client_facts_only_applies_to_cataract():
    assert ct.resolve_client_facts("macular", {"yellow_vision": True}) == {}


# ---------------------------------------------------------------------------
# consult_recommended propagation: True when ANY resolved program has the flag.
# ---------------------------------------------------------------------------

def test_seed_from_triage_reports_consult_recommended_true_for_wet_amd():
    cx = _cx()
    cp.upsert(cx, "wet-amd", "Wet AMD", True,
              [{"slug": "angiogenx", "name": "AngiogenX"}])
    result = ct.seed_from_triage(cx, "f@x.com", "macular", {"amd_type": "wet"})
    assert result["programs"] == ["wet-amd"]
    assert result["consult_recommended"] is True


def test_seed_from_triage_reports_consult_recommended_false_otherwise():
    cx = _cx()
    cp.upsert(cx, "dry-amd", "Dry AMD", False,
              [{"slug": "wholomega", "name": "WholOmega"}])
    result = ct.seed_from_triage(cx, "g@x.com", "macular", {"amd_type": "dry"})
    assert result["programs"] == ["dry-amd"]
    assert result["consult_recommended"] is False


def test_seed_from_triage_reports_consult_recommended_false_for_glaucoma():
    cx = _cx()
    _seed_fake_programs(cx)
    result = ct.seed_from_triage(cx, "h@x.com", "glaucoma", {"iop_od": 25})
    assert result["consult_recommended"] is False
