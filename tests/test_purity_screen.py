"""The role-aware excipient screen and its safety guards."""
from dashboard import purity_avoidlist as pa
from dashboard import purity_screen as ps

AL = pa.load_avoidlist()


def test_stearate_alias_matches_red_under_any_wording():
    for wording in ["Magnesium Stearate", "vegetable stearate", "Stearic Acid"]:
        out = ps.screen_label(["Vitamin C"], ["Cellulose", wording], AL)
        assert out["color"] == "red", wording
        assert out["red_hits"], wording


def test_silica_in_other_ingredients_is_yellow():
    out = ps.screen_label(["Vitamin C"], ["Silicon Dioxide"], AL)
    assert out["color"] == "yellow"
    assert out["yellow_hits"] == ["Silicon Dioxide"]


def test_silica_as_an_ACTIVE_is_ignored():
    # Same substance, nutritional role: must not count. This is the whole point
    # of screening only Other Ingredients.
    out = ps.screen_label(["Silica 10 mg", "Vitamin C"], ["Cellulose"], AL)
    assert out["color"] == "green"
    assert out["yellow_hits"] == []


def test_red_beats_yellow():
    out = ps.screen_label(None, ["Silicon Dioxide", "Magnesium Stearate"], AL)
    assert out["color"] == "red"


def test_clean_other_ingredients_are_green():
    out = ps.screen_label(["Vitamin C"], ["Hypromellose", "Microcrystalline Cellulose"], AL)
    assert out["color"] == "green"
    assert out["red_hits"] == [] and out["yellow_hits"] == []


def test_empty_list_is_green_but_none_is_unrated():
    # A product genuinely listing no other ingredients is pristine (green).
    assert ps.screen_label(["Taurine"], [], AL)["color"] == "green"
    # Absence of DATA is never green.
    assert ps.screen_label(["Taurine"], None, AL)["color"] == "unrated"


def test_version_is_echoed():
    out = ps.screen_label(None, ["Cellulose"], AL)
    assert out["avoidlist_version"] == AL["version"]


# --- Word-boundary regression tests (substring false-positive bug) ---------

def test_flakes_words_are_not_flagged_by_the_lake_alias():
    # "lake" (from FD&C "aluminum lake" dyes) must not match inside "flakes".
    for wording in ["Oat Flakes", "Rice Flakes", "Potato Flakes"]:
        out = ps.screen_label(["Vitamin C"], ["Cellulose", wording], AL)
        assert out["color"] == "green", wording
        assert out["red_hits"] == [], wording


def test_calcium_silicate_is_not_flagged_by_the_silica_alias():
    # "silica" must not match inside "silicate" -- calcium silicate is a
    # distinct anticaking agent, not silicon dioxide.
    out = ps.screen_label(["Vitamin C"], ["Calcium Silicate"], AL)
    assert out["color"] == "green"
    assert out["yellow_hits"] == []


def test_aluminum_lake_dye_is_still_flagged_red():
    # Positive control: word-boundary matching must still catch real hits.
    for wording in ["Aluminum Lake", "FD&C Yellow 5 Aluminum Lake"]:
        out = ps.screen_label(["Vitamin C"], ["Cellulose", wording], AL)
        assert out["color"] == "red", wording
        assert out["red_hits"], wording


def test_bare_silica_is_still_flagged_yellow():
    # Positive control: unqualified "Silica" and "Silicon Dioxide" still hit.
    for wording in ["Silica", "Silicon Dioxide"]:
        out = ps.screen_label(["Vitamin C"], [wording], AL)
        assert out["color"] == "yellow", wording
        assert out["yellow_hits"] == [wording], wording


def test_magnesium_stearate_is_still_flagged_red():
    # Positive control: multi-word alias still matches on word boundaries.
    out = ps.screen_label(["Vitamin C"], ["Magnesium Stearate"], AL)
    assert out["color"] == "red"
    assert out["red_hits"] == ["Magnesium Stearate"]


# --- Hyphen + negation regression tests (final-review findings) ------------

def test_hyphenated_stearate_is_still_flagged_red():
    # Bug 1 (false negative, the unsafe direction): a hyphen instead of a
    # space must not let a real red excipient slip through as green.
    out = ps.screen_label(["Vitamin C"], ["Magnesium-Stearate"], AL)
    assert out["color"] == "red"
    assert out["red_hits"] == ["Magnesium-Stearate"]


def test_hyphenated_aluminum_lake_is_still_flagged_red():
    # The hyphen-collapse fix must not break the existing word-boundary
    # protection -- "aluminum-lake" IS an aluminum lake dye and stays red.
    out = ps.screen_label(["Vitamin C"], ["Aluminum-Lake"], AL)
    assert out["color"] == "red"
    assert out["red_hits"] == ["Aluminum-Lake"]


def test_absence_declarations_are_green_not_red():
    # Bug 2 (false positives, stat-inflating): marketing phrasings that
    # DECLARE ABSENCE of an excipient must not be flagged as containing it.
    for wording in ["Gelatin-Free Capsule", "non-gelatin", "Non-Hydrogenated Palm Oil"]:
        out = ps.screen_label(["Vitamin C"], ["Cellulose", wording], AL)
        assert out["color"] == "green", wording
        assert out["red_hits"] == [], wording


def test_negation_stripping_does_not_over_correct_real_hits():
    # Positive controls: negation stripping must be conservative enough that
    # real excipients (no negation language present) still flag correctly.
    for wording, expected_color in [
        ("Magnesium Stearate", "red"),
        ("Gelatin", "red"),
        ("Hydrogenated Soybean Oil", "red"),
        ("Aluminum Lake", "red"),
        ("Oat Flakes", "green"),
        ("Calcium Silicate", "green"),
        ("Silicon Dioxide", "yellow"),
    ]:
        out = ps.screen_label(["Vitamin C"], [wording], AL)
        assert out["color"] == expected_color, wording
