import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dashboard import analysis_autoconfirm as ac

_RES = lambda name: "slug-" + name.lower().replace(" ", "-") if name else None  # fake resolver

def _draft(**over):
    base = {"greeting": "Aloha Pat.",
            "layers": [{"title": "Cellular", "remedy": "Vitality", "dosage": "1 cap", "frequency": "daily"}]}
    base.update(over); return base

def test_quality_pass_clean_draft():
    ok, reasons = ac.evaluate_quality(_draft(), resolve_slug=_RES, red_flag_terms=set())
    assert ok is True and reasons == []

def test_quality_fail_no_layers():
    ok, reasons = ac.evaluate_quality(_draft(layers=[]), resolve_slug=_RES, red_flag_terms=set())
    assert ok is False and any("layer" in r for r in reasons)

def test_quality_fail_unresolvable_remedy():
    ok, reasons = ac.evaluate_quality(
        _draft(layers=[{"title": "X", "remedy": "Nonexistent", "dosage": "1"}]),
        resolve_slug=lambda n: None, red_flag_terms=set())
    assert ok is False and any("remedy" in r for r in reasons)

def test_quality_fail_missing_dosing():
    ok, reasons = ac.evaluate_quality(
        _draft(layers=[{"title": "X", "remedy": "Vitality"}]),
        resolve_slug=_RES, red_flag_terms=set())
    assert ok is False and any("dosing" in r for r in reasons)

def test_quality_fail_red_flag_term():
    ok, reasons = ac.evaluate_quality(
        _draft(greeting="Concerns about your cancer diagnosis."),
        resolve_slug=_RES, red_flag_terms={"cancer"})
    assert ok is False and any("red_flag" in r for r in reasons)

def test_sampler_bounds():
    assert ac.should_sample("a@x.com", "2026-07-01", 0) is False
    assert ac.should_sample("a@x.com", "2026-07-01", 100) is True

def test_sampler_deterministic():
    a = ac.should_sample("a@x.com", "2026-07-01", 25)
    b = ac.should_sample("a@x.com", "2026-07-01", 25)
    assert a == b  # same input → same decision every time

def test_sampler_rate_is_roughly_pct():
    hits = sum(ac.should_sample(f"u{i}@x.com", "2026-07-01", 20) for i in range(1000))
    assert 120 <= hits <= 280  # ~20% of 1000, generous band
