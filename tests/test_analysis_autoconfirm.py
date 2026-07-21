import sys, pathlib, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dashboard import analysis_autoconfirm as ac

_RES = lambda name: "slug-" + name.lower().replace(" ", "-") if name else None  # fake resolver

def _draft(**over):
    base = {"greeting": "Aloha Pat.",
            "layers": [{"title": "Cellular", "remedy": "Vitality", "dosage": "1 cap", "frequency": "daily"}]}
    base.update(over); return base

def _cx():
    cx = sqlite3.connect(":memory:")
    ac.init_autoconfirm_log(cx)
    return cx

def _spy():
    calls = []
    def fn(cx, email, scan_date, content):
        calls.append((email, scan_date))
    fn.calls = calls
    return fn

def _call(cx, content, enabled=True, pct=0, resolver=_RES, confirm=None, red=set()):
    confirm = confirm or _spy()
    out = ac.maybe_auto_confirm(cx, "a@x.com", "2026-07-01", content,
        enabled=enabled, sample_pct=pct, resolve_slug=resolver,
        red_flag_terms=red, confirm_fn=confirm, now="2026-07-21T00:00:00Z")
    return out, confirm

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

def test_quality_fail_untitled_layer_with_content():
    ok, reasons = ac.evaluate_quality(
        _draft(layers=[
            {"title": "Cellular", "remedy": "Vitality", "dosage": "1 cap", "frequency": "daily"},
            {"remedy": "Bogus", "dosage": "1 cap"},
        ]),
        resolve_slug=_RES, red_flag_terms=set())
    assert ok is False and any("no title" in r for r in reasons)

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

def test_disabled_flag_short_circuits():
    cx = _cx()
    out, confirm = _call(cx, _draft(), enabled=False)
    assert out == "disabled" and confirm.calls == []

def test_clean_draft_auto_confirms_and_logs():
    cx = _cx()
    out, confirm = _call(cx, _draft(), pct=0)
    assert out == "confirmed" and confirm.calls == [("a@x.com", "2026-07-01")]
    row = cx.execute("SELECT decision FROM analysis_autoconfirm_log").fetchone()
    assert row[0] == "confirmed"

def test_quality_fail_holds_and_does_not_confirm():
    cx = _cx()
    out, confirm = _call(cx, _draft(layers=[]))
    assert out == "held_quality" and confirm.calls == []

def test_sampled_draft_is_held_even_when_clean():
    cx = _cx()
    out, confirm = _call(cx, _draft(), pct=100)
    assert out == "held_sample" and confirm.calls == []
