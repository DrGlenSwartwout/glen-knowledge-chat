"""`covers` and `uncovered` must come out in chain order, not hash order.

`_covers_for` used to iterate a set intersection (`codes`) and a set (`active_tokens`),
so the label order depended on the process hash seed. The same seeded fixture produced
['Membrane', 'Lymph'] or ['Lymph', 'Membrane'] run to run, and
test_biofield_suggest_remedies failed on origin/main under PYTHONHASHSEED=0 while
passing on 1..9. A flaky test corrupts every regression diff that follows it.

`token_label` is a dict filled while walking the active stresses, so its insertion
order is chain order. Iterating it fixes both lists.

Eight tokens here, not two: a set that happened to iterate correctly would be a 1-in-8!
coincidence rather than a 1-in-2 coin flip, so this test actually catches a regression.
"""
from dashboard.biofield_stress import _covers_for


CHAIN_ORDER = ["m1", "l2", "k3", "h4", "s5", "g6", "n7", "c8"]
LABELS = {"m1": "Membrane", "l2": "Lymph", "k3": "Kidney", "h4": "Heart",
          "s5": "Spleen", "g6": "Gut", "n7": "Nerve", "c8": "Cortex"}


def _token_label():
    """Insertion-ordered exactly as the stresses appear on the chain."""
    return {t: LABELS[t] for t in CHAIN_ORDER}


def test_covers_follows_chain_order_not_set_order():
    active = set(CHAIN_ORDER)
    coverage = {"one remedy": set(CHAIN_ORDER)}     # a set: iteration order is arbitrary
    picks, uncovered = _covers_for(["one remedy"], active, _token_label(), coverage)
    assert picks[0]["covers"] == [LABELS[t] for t in CHAIN_ORDER]
    assert uncovered == []


def test_uncovered_follows_chain_order_not_set_order():
    active = set(CHAIN_ORDER)
    coverage = {"partial": {"k3", "h4"}}
    picks, uncovered = _covers_for(["partial"], active, _token_label(), coverage)
    assert picks[0]["covers"] == ["Kidney", "Heart"]      # chain order, not {"h4","k3"}
    assert uncovered == ["Membrane", "Lymph", "Spleen", "Gut", "Nerve", "Cortex"]


def test_result_is_identical_across_repeated_calls():
    active = set(CHAIN_ORDER)
    coverage = {"r": {"c8", "m1", "s5"}}
    runs = {tuple(_covers_for(["r"], active, _token_label(), coverage)[0][0]["covers"])
            for _ in range(50)}
    assert len(runs) == 1, f"nondeterministic within a process: {runs}"


def test_a_remedy_covering_nothing_active_reports_no_labels():
    picks, uncovered = _covers_for(["stranger"], set(CHAIN_ORDER), _token_label(),
                                   {"stranger": {"zz9"}})
    assert picks[0]["covers"] == []
    assert uncovered == [LABELS[t] for t in CHAIN_ORDER]
