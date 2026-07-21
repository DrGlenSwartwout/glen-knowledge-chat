import sqlite3
import app as app_module
from dashboard import recommendation_events as re, wishlist as wl


def test_merge_emits_self_for_merged_slugs():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx); wl.init_wishlist_table(cx)
    # anonymous session added two products
    wl.toggle(cx, "sess:S1", "neuro-magnesium")
    wl.toggle(cx, "sess:S1", "immune-modulation")
    app_module._wishlist_merge_with_self(cx, "S1", "A@B.com")
    # both moved to the email wishlist AND recorded as self events
    assert wl.slugs_for(cx, "email:a@b.com") == {"neuro-magnesium", "immune-modulation"}
    selfev = {e["product_key"] for e in re.list_events(cx, "a@b.com") if e["source_key"] == "self"}
    assert selfev == {"neuro-magnesium", "immune-modulation"}
    # idempotent: a second merge (nothing in session now) adds nothing
    app_module._wishlist_merge_with_self(cx, "S1", "a@b.com")
    assert len([e for e in re.list_events(cx, "a@b.com") if e["source_key"] == "self"]) == 2


def test_merge_helper_never_raises_on_blank():
    cx = sqlite3.connect(":memory:")
    re.init_recommendation_events(cx); wl.init_wishlist_table(cx)
    app_module._wishlist_merge_with_self(cx, "", "a@b.com")   # no session -> no-op
    app_module._wishlist_merge_with_self(cx, "S1", "")        # no email -> no-op
    assert re.list_events(cx, "a@b.com") == []
