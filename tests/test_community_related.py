import sqlite3
import app as appmod
from dashboard import community as _c


def _seed(*, tags_per_item):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _c.init_community_tables(cx); _c.init_feed_tables(cx)
        ids = []
        for i, tags in enumerate(tags_per_item):
            cid = _c.upsert_full(cx, type="coaching_replay", title=f"T{i}", description="",
                                 video_ref=f"https://rumble.com/v-{i}", interest_tags=tags,
                                 transcript=""); _c.publish(cx, cid); ids.append(cid)
        # embed item 0 near [1,0,0], item 1 near [0,1,0]
        _c.set_embedding(cx, ids[0], [1.0, 0.0, 0.0], appmod.COMMUNITY_FEED_MODEL)
        _c.set_embedding(cx, ids[1], [0.0, 1.0, 0.0], appmod.COMMUNITY_FEED_MODEL)
        cx.commit()
    return ids


# Runs first, deliberately, before any other test in this module seeds embedded
# items into the shared appmod.LOG_DB file (there is no per-test DB reset here) —
# otherwise a leftover embedded item from a later-defined test could satisfy this
# query and falsely pass the "unembedded item skipped" assertion below.
def test_related_empty_when_no_embeddings():
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _c.init_community_tables(cx); _c.init_feed_tables(cx)
        cid = _c.upsert_full(cx, type="coaching_replay", title="NE", description="",
                             video_ref="https://rumble.com/v-ne", interest_tags=[],
                             transcript=""); _c.publish(cx, cid); cx.commit()
        out = appmod._community_related(cx, [1.0, 0.0, 0.0], is_paid=True, k=2)
    assert out == []                                   # unembedded item skipped


def test_related_returns_nearest_first_paid():
    ids = _seed(tags_per_item=[["sleep"], ["adrenals"]])
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        out = appmod._community_related(cx, [1.0, 0.05, 0.0], is_paid=True, k=2)
    assert out[0]["id"] == ids[0]                      # nearest to [1,0,0]
    assert out[0]["kind"] == "full"
    assert "video_ref" not in out[0]                   # never leaks the Rumble link


def test_related_free_is_teaser_kind():
    _seed(tags_per_item=[["sleep"], ["adrenals"]])
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        out = appmod._community_related(cx, [1.0, 0.0, 0.0], is_paid=False, k=2)
    assert out and all(it["kind"] == "teaser" for it in out)
    assert all("video_ref" not in it for it in out)


def test_related_excludes_below_min_sim():
    _seed(tags_per_item=[["sleep"], ["adrenals"]])
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        # query orthogonal-ish to both → nothing clears a high bar
        out = appmod._community_related(cx, [0.0, 0.0, 1.0], is_paid=True, k=2, min_sim=0.72)
    assert out == []
