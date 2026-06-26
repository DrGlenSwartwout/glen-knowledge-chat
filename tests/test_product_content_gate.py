"""Post-generation compliance gate: deny-term detection + retry/degrade logic,
tested deterministically with a fake Anthropic client (no network)."""
import dashboard.product_content as pc


def test_deny_hits_detects_disease_and_verbs():
    assert "floaters" in pc._deny_hits("These floaters are annoying")
    assert "detachment" in pc._deny_hits("risk of retinal detachment")
    assert set(pc._deny_hits("It treats and prevents and cures")) >= {"treats", "prevents", "cures"}


def test_deny_hits_clean_structure_function_passes():
    assert pc._deny_hits("Supports the vitreous body and connective-tissue resilience") == []
    # anatomy/biochem terms are allowed (not disease claims)
    assert pc._deny_hits("supports the retina; antioxidants counter oxidation and glycation") == []


class _Msg:
    def __init__(self, text):
        self.content = [type("C", (), {"text": text})()]


class _Messages:
    def __init__(self, seq):
        self.seq = list(seq)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return _Msg(self.seq.pop(0))


class _Client:
    def __init__(self, seq):
        self.messages = _Messages(seq)


def test_gen_compliant_clean_first_try_no_retry():
    cl = _Client(["Supports the vitreous body."])
    raw, ok = pc._gen_compliant(cl, "sys", "user", 100, lambda r: r)
    assert ok is True and "vitreous" in raw
    assert len(cl.messages.calls) == 1


def test_gen_compliant_retries_then_succeeds():
    cl = _Client(["Helps with floaters.", "Supports clear, resilient vitreous tissue."])
    raw, ok = pc._gen_compliant(cl, "sys", "user", 100, lambda r: r)
    assert ok is True and "floater" not in raw.lower()
    assert len(cl.messages.calls) == 2
    # the retry prompt names the violated word
    assert "floaters" in cl.messages.calls[1]["messages"][0]["content"].lower()


def test_gen_compliant_degrades_when_still_violating():
    cl = _Client(["floaters everywhere", "still about floaters"])
    raw, ok = pc._gen_compliant(cl, "sys", "user", 100, lambda r: r)
    assert ok is False
    assert len(cl.messages.calls) == 2
