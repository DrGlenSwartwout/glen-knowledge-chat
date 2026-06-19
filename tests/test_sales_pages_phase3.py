import sqlite3
from dashboard import sales_images as si

def _cx(): return sqlite3.connect(":memory:")

def test_queue_enqueue_pending_done():
    cx = _cx()
    si.enqueue(cx, "longevity")
    assert si.list_pending(cx) == ["longevity"]
    assert si.queue_state(cx, "longevity") == "pending"
    si.mark_done(cx, "longevity")
    assert si.list_pending(cx) == []
    assert si.queue_state(cx, "longevity") == "done"

def test_enqueue_idempotent_resets_to_pending():
    cx = _cx()
    si.enqueue(cx, "energy"); si.mark_failed(cx, "energy")
    si.enqueue(cx, "energy")
    assert si.queue_state(cx, "energy") == "pending"

def test_record_and_display_first_ready_per_kind():
    cx = _cx()
    si.record_image(cx, "longevity", "botanical", 1, "botanical-1.png")
    si.record_image(cx, "longevity", "botanical", 2, "botanical-2.png")
    si.record_image(cx, "longevity", "mechanism", 1, "mechanism-1.png")
    disp = si.display_images(cx, "longevity")
    assert disp == {"botanical": "botanical-1.png", "mechanism": "mechanism-1.png"}
    assert len(si.get_images(cx, "longevity")) == 3
