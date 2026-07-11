from dashboard.product_page_sections import filter_sections

_ALL = [{"id": i} for i in
        ("intro", "description", "video", "ingredients", "comparison", "research", "images", "cta")]


def _ids(secs):
    return [s["id"] for s in secs]


def test_formula_keeps_all_sections():
    out = filter_sections(_ALL, has_ingredients=True, has_own_video=False)
    assert _ids(out) == _ids(_ALL)


def test_device_drops_ingredients_comparison_and_empty_video():
    # No ingredient list and no product video of its own: ingredients, comparison
    # (which also carries the Miron rotator + story), and the empty Watch section go.
    out = filter_sections(_ALL, has_ingredients=False, has_own_video=False)
    assert _ids(out) == ["intro", "description", "research", "images", "cta"]


def test_device_with_own_video_keeps_watch():
    out = filter_sections(_ALL, has_ingredients=False, has_own_video=True)
    assert "video" in _ids(out)
    assert "ingredients" not in _ids(out) and "comparison" not in _ids(out)


def test_does_not_mutate_input():
    before = _ids(_ALL)
    filter_sections(_ALL, has_ingredients=False, has_own_video=False)
    assert _ids(_ALL) == before
