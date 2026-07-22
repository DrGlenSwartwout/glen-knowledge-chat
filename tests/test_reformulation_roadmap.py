"""Reformulation-roadmap tests. Pure sqlite; the LLM client is stubbed so no
network call. Covers the deterministic frequency, the corpus filter, generate
(cache + parse), latest, and the empty-corpus guard (must not call the LLM)."""
import sqlite3

from dashboard import reformulation_roadmap as rr
from dashboard import supplement_reviews as sr


def _cx():
    cx = sqlite3.connect(":memory:")
    sr.init_table(cx)
    rr.init_table(cx)
    return cx


class _FakeMsg:
    def __init__(self, text):
        self.content = [type("B", (), {"type": "text", "text": text})()]


class _FakeMessages:
    def __init__(self, text):
        self._t = text

    def create(self, **k):
        return _FakeMsg(self._t)


class _FakeClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


def test_frequency_ranks_by_count():
    cx = _cx()
    sr.create_request(cx, "a@x.com", "Mag Oxide", "BrandA")
    sr.create_request(cx, "b@x.com", "Mag Oxide", "BrandA")   # same product, another submitter
    sr.create_request(cx, "c@x.com", "Fish Oil", "OmegaCo")
    freq = rr.frequency(cx)
    assert freq[0]["product_name"] == "Mag Oxide" and freq[0]["count"] == 2
    assert any(f["product_name"] == "Fish Oil" and f["count"] == 1 for f in freq)


def test_corpus_only_reviewed_rows():
    cx = _cx()
    sr.create_request(cx, "a@x.com", "P1", "B")               # requested, no critique yet
    b = sr.create_request(cx, "b@x.com", "P2", "B"); sr.set_draft(cx, b["id"], "weak: uses oxide")
    items = rr.corpus(cx)
    assert len(items) == 1 and items[0]["product_name"] == "P2"


def test_generate_caches_and_latest_reads():
    cx = _cx()
    b = sr.create_request(cx, "b@x.com", "Mag Oxide", "BrandA")
    sr.set_draft(cx, b["id"], "uses poorly-absorbed magnesium oxide")
    fake = _FakeClient('here is the plan {"roadmap":[{"category":"Magnesium","submission_count":1,'
                       '"common_weaknesses":["oxide form"],"reformulation_opportunity":"offer glycinate",'
                       '"priority":5}]} done')
    out = rr.generate(cx, fake)
    assert out["n_reviews"] == 1 and out["roadmap"][0]["category"] == "Magnesium"
    lt = rr.latest(cx)
    assert lt["roadmap"][0]["reformulation_opportunity"] == "offer glycinate" and lt["n_reviews"] == 1


def test_generate_empty_corpus_skips_llm():
    cx = _cx()

    class Boom:
        @property
        def messages(self):
            raise AssertionError("LLM must not be called when there are no reviewed submissions")

    out = rr.generate(cx, Boom())
    assert out["roadmap"] == [] and out["n_reviews"] == 0


def test_generate_bad_json_degrades_to_empty():
    cx = _cx()
    b = sr.create_request(cx, "b@x.com", "P", "B"); sr.set_draft(cx, b["id"], "some critique")
    out = rr.generate(cx, _FakeClient("not json at all"))
    assert out["roadmap"] == [] and out["n_reviews"] == 1  # counted, but no roadmap


def test_latest_empty_when_never_generated():
    cx = _cx()
    assert rr.latest(cx) == {"roadmap": [], "n_reviews": 0, "generated_at": None}
