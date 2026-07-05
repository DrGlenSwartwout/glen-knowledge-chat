import sqlite3, json
from dashboard import community as _c


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _c.init_community_tables(cx)
    return cx


def test_create_and_get():
    cx = _cx()
    cid = _c.create_content(cx, type="coaching_replay", title="Week 1",
                            description="d", video_ref="https://rumble.com/v-abc",
                            tier="paid", interest_tags=["sleep", "adrenals"])
    row = _c.get_content(cx, cid)
    assert row["title"] == "Week 1"
    assert row["tier"] == "paid"
    assert json.loads(row["interest_tags"]) == ["sleep", "adrenals"]
    assert row["published"] == 0


def test_list_full_only_published_newest_first():
    cx = _cx()
    a = _c.create_content(cx, type="coaching_replay", title="A", description="",
                          video_ref="r/a", tier="paid", interest_tags=[])
    b = _c.create_content(cx, type="course_session", title="B", description="",
                          video_ref="r/b", tier="paid", interest_tags=[])
    _c.publish(cx, a); _c.publish(cx, b)
    cx.execute("UPDATE community_content SET published_at='2026-01-01' WHERE id=?", (a,))
    cx.execute("UPDATE community_content SET published_at='2026-02-01' WHERE id=?", (b,)); cx.commit()
    titles = [r["title"] for r in _c.list_full(cx)]
    assert titles == ["B", "A"]  # newest first


def test_list_full_excludes_unpublished_and_outtakes():
    cx = _cx()
    f = _c.create_content(cx, type="coaching_replay", title="F", description="",
                          video_ref="r/f", tier="paid", interest_tags=[]); _c.publish(cx, f)
    _c.create_content(cx, type="coaching_replay", title="Draft", description="",
                      video_ref="r/d", tier="paid", interest_tags=[])  # unpublished
    o = _c.add_outtake(cx, parent_id=f, title="clip", video_ref="/asset/x.mp4",
                       interest_tags=[]); _c.publish(cx, o)
    assert [r["title"] for r in _c.list_full(cx)] == ["F"]
    full = _c.list_full(cx)[0]
    assert [ot["title"] for ot in full["outtakes"]] == ["clip"]


def test_upsert_full_is_idempotent_on_video_ref():
    cx = _cx()
    id1 = _c.upsert_full(cx, type="coaching_replay", title="v1", description="",
                         video_ref="r/same", interest_tags=["x"], transcript="t1")
    o1 = _c.add_outtake(cx, parent_id=id1, title="old-clip", video_ref="/a.mp4",
                        interest_tags=[])
    id2 = _c.upsert_full(cx, type="coaching_replay", title="v2", description="",
                         video_ref="r/same", interest_tags=["y"], transcript="t2")
    assert id1 == id2  # same row
    row = _c.get_content(cx, id2)
    assert row["title"] == "v2" and row["transcript"] == "t2"
    # old out-takes cleared by the re-upsert
    assert cx.execute("SELECT COUNT(*) FROM community_content WHERE parent_id=?",
                      (id2,)).fetchone()[0] == 0


def test_add_outtake_is_free_and_linked():
    cx = _cx()
    f = _c.upsert_full(cx, type="course_session", title="F", description="",
                       video_ref="r/f", interest_tags=[], transcript="")
    o = _c.add_outtake(cx, parent_id=f, title="teaser", video_ref="/asset/t.mp4",
                       interest_tags=["thyroid"]); _c.publish(cx, o)
    row = _c.get_content(cx, o)
    assert row["tier"] == "free" and row["type"] == "outtake" and row["parent_id"] == f
    assert [r["title"] for r in _c.list_outtakes(cx, parent_id=f)] == ["teaser"]
