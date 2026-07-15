import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_begin_tos_version_bumped():
    txt = (ROOT / "app.py").read_text()
    assert 'BEGIN_TOS_VERSION = "rm-e4l-tc-2026-07-15"' in txt

def test_ethics_clause_in_onboarding_copy():
    for f in ["static/begin-doorway.html", "static/begin-fireside.html", "static/begin-biofield.html"]:
        assert "share it ethically" in (ROOT / f).read_text().lower()

def test_no_em_dash_in_new_clause():
    for f in ["static/begin-doorway.html", "static/begin-fireside.html", "static/begin-biofield.html"]:
        for line in (ROOT / f).read_text().splitlines():
            if "share it ethically" in line.lower():
                assert "—" not in line  # no em dash
