"""Authoritative, hand-authored seed content for mentor pages.

Some mentor pages ship from vetted source material (Dr. Glen's research dossiers)
rather than being generated, so the pilot pages are accurate about real people
from day one. dashboard/mentor_copy.build_page() uses a seed when the slug has one
and falls back to grounded generation otherwise.

Voice follows Glen's copy rules: no em dashes, no ALL CAPS, no invented facts.
Each seed is written from the Becker+Burr field-of-life research dossier.
"""

SEEDS = {
    "harold-saxton-burr": {
        "name": "Harold Saxton Burr, PhD",
        "field": "Bioelectrodynamics, the electrodynamic field of life (L-fields)",
        "lifespan": "1889-1973",
        "vital_status": "deceased",
        "lineage": ["Burr", "Becker", "Cyril Smith", "Nordenstrom", "Levin"],
        "sources": [
            "H. S. Burr, Blueprint for Immortality: The Electric Patterns of Life (1972)",
            "Burr & Northrop, The Electro-Dynamic Theory of Life, Quarterly Review of Biology (1935)",
            "Langman & Burr, Science 105 (1947): 209 (cervical-cancer L-field screening)",
            "M. Levin, bioelectric morphogenesis, Cell (2021), which cites Burr",
        ],
        "seo": {
            "title": "Harold Saxton Burr, L-fields and the electric patterns of life",
            "meta_description": ("Yale anatomist Harold Saxton Burr showed that living "
                                 "bodies are organized by measurable electrodynamic fields. "
                                 "His L-field research anchors the modern bioelectric lineage."),
        },
        "content": {
            "life_and_work": (
                "Harold Saxton Burr was a Yale anatomist, not a figure on the fringe. He earned "
                "his BA at Yale in 1908 and his PhD there in 1915, then spent his whole career at "
                "the Yale School of Medicine, holding the E. K. Hunt Professorship of Anatomy and "
                "chairing the department for several terms.\n\n"
                "For four decades he taught medical students by day and ran a parallel research "
                "program into the electrical organization of living things, publishing more than "
                "ninety peer-reviewed papers between 1932 and 1972, most of them in the Yale "
                "Journal of Biology and Medicine."
            ),
            "key_contribution": (
                "Burr's central finding was that living organisms are shaped by measurable "
                "electrodynamic fields, which he called L-fields, or life-fields. Using "
                "high-impedance voltmeters that drew almost no current from the tissue, he "
                "measured steady voltage patterns on and around the body and found that they "
                "come before the physical changes they seem to guide.\n\n"
                "He documented voltage axes in salamander eggs that predicted the future nervous "
                "system, voltage shifts that predicted ovulation a day or two ahead, gradients "
                "that tracked wound healing, and, with the gynecologist Louis Langman, abnormal "
                "field patterns that flagged cervical malignancy in a screening of more than a "
                "thousand women. His 1972 synthesis, Blueprint for Immortality, remains the most "
                "cited account of this work."
            ),
            "lineage": (
                "Burr sits at the head of the strongest credentialed lineage behind the idea that "
                "the body is a bioelectric system. His work passed to Robert Becker, the "
                "orthopedic surgeon who regrew salamander limbs with tiny direct currents and "
                "wrote The Body Electric, and on through Cyril Smith and Bjorn Nordenstrom.\n\n"
                "It arrives in the present through Michael Levin at Tufts, whose 2021 work on "
                "bioelectric morphogenesis names Burr as a predecessor. What was once treated as "
                "energy medicine is now, in its core claim, mainstream physiology."
            ),
            "why_it_matters": (
                "Much of what Burr measured in the 1930s and 1940s has since been confirmed. The "
                "wound currents he described were rediscovered as the endogenous electric fields "
                "that guide healing cells, and his claim that voltage patterns organize the embryo "
                "is now an active field of research.\n\n"
                "For Dr. Glen Swartwout's clinical approach, Burr is the historical anchor for a "
                "simple, testable idea, that fields organize matter in the body, and that reading "
                "and supporting those fields is a legitimate part of care."
            ),
        },
    },
}


def get_seed(slug):
    return SEEDS.get((slug or "").strip().lower())
