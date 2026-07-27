from dashboard import fullscript_ingredients as fi


class _Resp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


# The Other Ingredients text as it really appears embedded in the page payload
# (unicode-escaped, wrapped in HTML tags) — verified 2026-07-26 against Jarrow
# Magnesium Taurate.
JARROW_RAW = (
    'x\\u003cb\\u003eOther Ingredients:\\u003c/b\\u003e\\u003cbr\\u003e\\n'
    'Capsule (hydroxypropylmethylcellulose), magnesium stearate '
    '(vegetable source) and silicon dioxide.\\u003cbr\\u003e\\u003cp\\u003eKeep out.\\u003c/p\\u003e'
)


def test_fetch_returns_cleaned_text_on_200():
    calls = {}

    def fake_fetch(url, headers):
        calls["url"] = url
        calls["ua"] = headers.get("User-Agent", "")
        return _Resp(200, JARROW_RAW)

    text = fi.fetch_page_text("magnesium-taurate", fetch=fake_fetch)
    assert calls["url"] == "https://fullscript.com/catalog/products/magnesium-taurate"
    assert "Mozilla" in calls["ua"]                       # browser UA sent
    # unicode escapes decoded and tags stripped -> the ingredient line is clean,
    # contiguous, human-readable text
    assert "Other Ingredients:" in text
    assert "magnesium stearate (vegetable source)" in text
    assert "silicon dioxide" in text
    assert "\\u003c" not in text and "<b>" not in text     # cleaned


def test_fetch_returns_none_on_non_200():
    assert fi.fetch_page_text("x", fetch=lambda u, h: _Resp(403, "denied")) is None


def test_fetch_returns_none_on_exception():
    def boom(url, headers):
        raise RuntimeError("network down")
    assert fi.fetch_page_text("x", fetch=boom) is None


def test_fetch_returns_none_on_blank_slug():
    assert fi.fetch_page_text("", fetch=lambda u, h: _Resp(200, "x")) is None
    assert fi.fetch_page_text(None, fetch=lambda u, h: _Resp(200, "x")) is None
