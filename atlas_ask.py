"""Atlas chat spine: answer a question and say which concepts to highlight."""
import re


def _tokens(text):
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def match_concepts(question, concepts, k=5):
    """Rank concepts by token overlap of label+aliases+summary with the question."""
    q = _tokens(question)
    scored = []
    for c in concepts:
        hay = _tokens(" ".join([c.get("label", ""), " ".join(c.get("aliases", [])),
                                c.get("summary", "")]))
        score = len(q & hay)
        if score:
            scored.append((score, c["id"]))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [cid for _, cid in scored[:k]]


def atlas_ask(question, concepts, answer_fn=None):
    """Return {answer, concept_ids, highlight}. answer_fn(question, concept_ids)->str
    lets the route inject the live chat backend; tests pass a stub."""
    ids = match_concepts(question, concepts, k=5)
    if answer_fn is not None:
        answer = answer_fn(question, ids)
    else:
        labels = [c["label"] for c in concepts if c["id"] in ids]
        answer = ("Related concepts: " + ", ".join(labels)) if labels else \
                 "I couldn't find a matching concept yet."
    return {"answer": answer, "concept_ids": ids, "highlight": ids}
