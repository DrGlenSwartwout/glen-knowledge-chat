"""Portal-facing Eye & Vision report derived from canonical E4L mappings.

Phase 1 is descriptive only: structure/function relationships and reported
history context. It intentionally contains no diagnosis, treatment, product,
remedy, protocol, or dosing suggestions.
"""
from __future__ import annotations

import hashlib
import json

from dashboard import biofield_e4l, intake, portal_extended_history


SECTION_KEY = "eye_vision_report"
OCULAR_STRUCTURES = {
    "eye", "eyes", "retina", "retinal", "optic nerve",
    "vision", "visual processing", "visual pathway",
}
EYE_HISTORY_TERMS = (
    "eye", "eyes", "vision", "visual", "ocular", "glaucoma", "cataract",
    "macular", "retina", "retinal", "optic nerve", "dry eye", "myopia",
    "hyperopia", "astigmatism", "floaters", "uveitis", "keratoconus",
    "amblyopia", "strabismus", "diplopia",
)
LIMITATIONS = [
    "This report describes E4L Infoceutical pattern-to-structure relationships.",
    "It does not diagnose an eye condition or establish damage, severity, prognosis, or causation.",
    "It does not replace an eye examination or other appropriate clinical assessment.",
    "Remedies, products, protocols, dosing, and support suggestions are outside this phase.",
]
CLINICAL_CONTEXT_USAGE = (
    "Related symptoms and conditions organize reported history only. "
    "They are not scan findings or diagnostic conclusions."
)
RELATIONSHIP_MAP = {
    "eye": {
        "label": "Eye and visual system",
        "structure_function": [
            "Integrates the optical surface, anterior and posterior segments, retina, "
            "optic pathway, and ocular-motor system",
            "Supports image formation, light regulation, visual signal transmission, "
            "alignment, fixation, and tracking",
        ],
        "innervation_regulation": [
            "CN II for visual input; CN III, IV, and VI for eye movement",
            "CN V/VII and autonomic regulation for surface protection, pupil, "
            "accommodation, tears, and circulation",
        ],
        "eav_points": [
            {"code": "OR-1–OR-13", "name": "Segment-specific orbital EAV points"},
            {"code": "NRD-1a / NRD-3a",
             "name": "Autonomic and parasympathetic regulatory points"},
        ],
        "energetic_relationships": [
            {"type": "classical meridians",
             "names": ["Liver", "Gall Bladder", "Bladder", "Heart",
                       "Small Intestine", "Lung", "Large Intestine",
                       "Stomach", "Spleen/Pancreas", "Kidney"]},
            {"type": "EAV vessels",
             "names": ["Endocrine/Triple Warmer", "Circulation", "Nerve",
                       "Connective Tissue", "Lymph"]},
        ],
        "information_relationships": [
            "Optical input through the cornea, pupil, lens, vitreous, and retina",
            "Retinal and optic-pathway signaling into thalamic and cortical processing",
            "Motor feedback through fixation, saccade, pursuit, vergence, vestibular, "
            "and proprioceptive integration",
        ],
        "symptom_context": [
            "blur or fluctuating vision", "glare or light sensitivity", "eye strain",
            "double vision or tracking difficulty", "dryness, redness, or discomfort",
            "flashes, floaters, distortion, or field change",
        ],
        "condition_context": [
            "ocular-surface and tear-film disorders", "corneal or lens disorders",
            "retinal, macular, or optic-nerve disorders", "uveal or vascular disorders",
            "ocular-motor, binocular-vision, or visual-processing disorders",
        ],
    },
    "retina": {
        "label": "Retina",
        "structure_function": ["Phototransduction",
                               "Central and peripheral visual input"],
        "innervation_regulation": ["CN II"],
        "eav_points": [{"code": "OR-5", "name": "Retina"}],
        "energetic_relationships": [
            {"type": "classical meridian", "names": ["Small Intestine"]},
            {"type": "EAV vessel", "names": ["Endocrine/Triple Warmer"]},
        ],
        "information_relationships": [
            "Light absorption and phototransduction",
            "Retinal signal transmission into the optic pathway",
            "Photoenergetic and circadian signaling into autonomic/endocrine regulation",
        ],
        "symptom_context": [
            "flashes", "floaters", "field loss", "night-vision difficulty",
            "distorted or reduced vision",
        ],
        "condition_context": [
            "retinal tear or detachment", "retinopathy", "retinal vascular disorders",
            "inherited retinal degeneration",
        ],
    },
    "optic nerve": {
        "label": "Optic nerve",
        "structure_function": [
            "Transmission of retinal signals into the central visual pathway"],
        "innervation_regulation": ["CN II"],
        "eav_points": [{"code": "Temporal reflex point", "name": "Optic nerve"}],
        "energetic_relationships": [
            {"type": "classical meridians", "names": ["Liver", "Kidney"]}],
        "information_relationships": [
            "Carries retinal information toward the optic chiasm, thalamus, and cortex",
            "Participates in the afferent pupil-light pathway",
        ],
        "symptom_context": [
            "reduced acuity", "color desaturation", "contrast loss",
            "relative afferent pupil change", "visual-field loss",
        ],
        "condition_context": [
            "glaucoma", "optic neuritis",
            "ischemic, compressive, toxic, or nutritional optic neuropathy",
        ],
    },
    "vision": {
        "label": "Visual pathway and processing",
        "structure_function": [
            "Transforms retinal input into visual perception, spatial orientation, "
            "learning, and visually guided action",
            "Coordinates attention with fixation, saccades, pursuit, and vergence",
        ],
        "innervation_regulation": [
            "CN II; thalamic, cortical, brain-stem, cerebellar, autonomic, and "
            "ocular-motor regulation"],
        "eav_points": [
            {"code": "Composite",
             "name": "Retina, optic pathway, autonomic, and ocular-motor points"}],
        "energetic_relationships": [
            {"type": "EAV vessels", "names": ["Endocrine/Triple Warmer", "Nerve"]},
            {"type": "classical meridians",
             "names": ["Bladder", "Gall Bladder", "Liver"]},
        ],
        "information_relationships": [
            "Retina to optic nerve, chiasm, thalamus, visual and association cortex",
            "Bidirectional visual, eye-movement, vestibular, and motor feedback",
            "Light input linked with hypothalamic, circadian, autonomic, and "
            "endocrine regulation",
        ],
        "symptom_context": [
            "visual-processing fatigue", "poor tracking or reading fluency",
            "spatial-orientation or visual-perceptual difficulty", "light sensitivity",
            "unstable gaze or visually triggered dizziness",
        ],
        "condition_context": [
            "visual-processing or visual-perceptual disorders",
            "ocular-motor and binocular-vision disorders",
            "migraine or visual-vestibular contexts",
            "central neurologic visual-pathway disorders",
        ],
    },
}
RELATIONSHIP_ALIASES = {
    "eyes": "eye", "retinal": "retina",
    "visual pathway": "vision", "visual processing": "vision",
}
URGENT_TERMS = (
    "curtain", "sudden vision loss", "acute vision loss", "painful red eye",
    "acute double vision", "sudden double vision", "new ptosis",
    "dilated pupil", "severe eye trauma", "sudden neurologic",
)
PROGRAM_MATCHES = {
    "glaucoma": ["glaucoma-elevated-iop", "glaucoma-normal-iop"],
    "elevated eye pressure": ["glaucoma-elevated-iop"],
    "normal tension glaucoma": ["glaucoma-normal-iop"],
    "macular degeneration": ["dry-amd", "wet-amd"],
    "dry amd": ["dry-amd"], "wet amd": ["wet-amd"],
    "cataract": ["senile-cataract", "psc-cataract"],
    "dry eye": ["dry-eye"],
    "retinitis pigmentosa": ["retinitis-pigmentosa"],
    "diabetic retinopathy": ["diabetic-retinopathy"],
}


def _norm(value):
    return " ".join(str(value or "").strip().lower().replace("-", " ").split())


def _is_ocular(value):
    return _norm(value) in OCULAR_STRUCTURES


def _relationship_layers(structures):
    out, seen = [], set()
    for structure in structures:
        key = RELATIONSHIP_ALIASES.get(_norm(structure), _norm(structure))
        if key in seen or key not in RELATIONSHIP_MAP:
            continue
        seen.add(key)
        out.append({"mapped_from": structure, **RELATIONSHIP_MAP[key]})
    return out


def _pattern_label(category):
    return ("informational stress/rejuvenator pattern"
            if _norm(category) in {"er", "mr"} else "Infoceutical pattern")


def _flatten_strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_flatten_strings(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    return []


def _find_practitioner_satisfaction(value):
    """Return an explicitly stored satisfaction answer; never infer it."""
    keys = {
        "current_practitioner_satisfactory",
        "happy_with_current_practitioners",
        "happy_with_current_practitioner",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if _norm(key).replace(" ", "_") in keys and isinstance(item, bool):
                return item
        for item in value.values():
            found = _find_practitioner_satisfaction(item)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_practitioner_satisfaction(item)
            if found is not None:
                return found
    return None


def reported_eye_history(cx, email):
    """Return matching client-reported/canonical eye-history labels."""
    labels = []
    try:
        response = intake.get_response(cx, email) or {}
        labels.extend(_flatten_strings(response.get("answers") or {}))
    except Exception:
        pass
    try:
        extended = portal_extended_history.get(cx, email) or {}
        labels.extend(_flatten_strings(extended.get("answers") or {}))
    except Exception:
        pass
    try:
        from dashboard import canonical_tags
        labels.extend(str(v) for v in
                      (canonical_tags.get_person(cx, email).get("conditions") or []))
    except Exception:
        pass
    try:
        row = cx.execute(
            "SELECT conditions, tags FROM people WHERE lower(email)=lower(?)",
            ((email or "").strip(),)).fetchone()
        if row:
            for cell in (row[0], row[1]):
                try:
                    labels.extend(_flatten_strings(json.loads(cell or "[]")))
                except Exception:
                    pass
    except Exception:
        pass
    matches = []
    for label in labels:
        text = _norm(label)
        if text and any(term in text for term in EYE_HISTORY_TERMS):
            clean = str(label).strip()
            if clean and clean not in matches:
                matches.append(clean)
    return matches


def _mappings_for(codes, db_path=None):
    mappings = {}
    cx = biofield_e4l._connect_ro(biofield_e4l._db_path(db_path))
    if cx is None or not codes:
        return mappings
    try:
        placeholders = ",".join("?" for _ in codes)
        rows = cx.execute(
            f"SELECT code, structure, stype, is_primary, source_phrase "
            f"FROM e4l_pattern_structures WHERE code IN ({placeholders}) "
            f"ORDER BY is_primary DESC, structure ASC", tuple(codes)).fetchall()
        for row in rows:
            mappings.setdefault(row["code"], []).append(dict(row))
    except Exception:
        return {}
    finally:
        cx.close()
    return mappings


def _finding(pattern, mappings, history):
    ocular = [m for m in mappings if _is_ocular(m.get("structure"))]
    if not ocular:
        return None
    structures = list(dict.fromkeys(
        str(m.get("structure") or "").strip() for m in ocular
        if str(m.get("structure") or "").strip()))
    direct = any(bool(m.get("is_primary")) for m in ocular)
    code = str(pattern.get("code") or "").strip()
    name = str(pattern.get("name") or code or "E4L pattern").strip()
    pattern_label = _pattern_label(pattern.get("category"))
    seed = code + "|" + "|".join(sorted(_norm(v) for v in structures))
    if direct:
        body = (f"The scan prioritized the {name} {pattern_label}, which "
                f"the E4L catalog maps directly to {', '.join(structures)}.")
        classification = "direct_ocular"
    else:
        body = (f"The broader {name} {pattern_label} includes an E4L "
                f"catalog relationship with {', '.join(structures)}.")
        classification = "associated_ocular"
    history_context = None
    if history:
        history_context = (
            f"Your reported history includes {', '.join(history)}. It is shown "
            "alongside this relationship for context; the scan does not diagnose "
            "the reported condition or establish a causal relationship.")
    return {
        "finding_id": hashlib.sha256(seed.encode()).hexdigest()[:16],
        "code": code or None,
        "category": pattern.get("category"),
        "name": name,
        "priority_rank": pattern.get("rank"),
        "classification": classification,
        "ocular_structures": structures,
        "client_text": body,
        "history_context": history_context,
        "integrated_relationships": _relationship_layers(structures),
        "clinical_context_usage": CLINICAL_CONTEXT_USAGE,
    }


def _practitioner_history(cx, email):
    labels = reported_eye_history(cx, email)
    all_text = list(labels)
    practitioner_satisfactory = None
    try:
        intake_answers = (intake.get_response(cx, email) or {}).get("answers") or {}
        practitioner_satisfactory = _find_practitioner_satisfaction(intake_answers)
    except Exception:
        pass
    try:
        extended = portal_extended_history.get(cx, email) or {}
        extended_answers = extended.get("answers") or {}
        all_text.extend(_flatten_strings(extended_answers))
        if practitioner_satisfactory is None:
            practitioner_satisfactory = _find_practitioner_satisfaction(
                extended_answers)
    except Exception:
        pass
    current_products = []
    try:
        from dashboard import portal_health_history
        health = portal_health_history.get(cx, email) or {}
        for kind in portal_health_history.KINDS:
            if health.get(kind + "_yes") and health.get(kind + "_text"):
                current_products.append({
                    "kind": kind, "names": health[kind + "_text"]})
    except Exception:
        pass
    return labels, all_text, current_products, practitioner_satisfactory


def _suggestion_contract(findings, history, all_text, current_products,
                         practitioner_satisfactory=None):
    safety_text = " ".join(str(v) for v in all_text).lower()
    urgent_matches = [term for term in URGENT_TERMS if term in safety_text]
    candidates = []
    if urgent_matches:
        seed = "urgent|" + "|".join(sorted(urgent_matches))
        candidates.append({
            "suggestion_id": hashlib.sha256(seed.encode()).hexdigest()[:16],
            "category": "clinical_referral",
            "name": "Urgent conventional eye or medical assessment",
            "status": "pending_practitioner_approval",
            "client_visible": False,
            "priority": "urgent",
            "trigger": {"reported_terms": urgent_matches},
            "rationale": (
                "Reported language matches an urgent eye/neurologic safety rule. "
                "Remedy and product drafts are suppressed until appropriate assessment."
            ),
            "confidence": "rule_match",
            "provenance": "eye_vision_safety_triage",
            "referral_route": _urgent_referral_route(practitioner_satisfactory),
        })
    else:
        for finding in findings:
            if _norm(finding.get("category")) in {"er", "mr"}:
                continue
            seed = "infoceutical|" + finding["finding_id"]
            candidates.append({
                "suggestion_id": hashlib.sha256(seed.encode()).hexdigest()[:16],
                "category": "infoceutical",
                "name": finding["name"],
                "code": finding.get("code"),
                "status": "pending_practitioner_approval",
                "client_visible": False,
                "priority": finding.get("priority_rank"),
                "trigger": {"finding_id": finding["finding_id"],
                            "classification": finding["classification"]},
                "rationale": (
                    "The E4L scan prioritized this pattern. Its ocular mapping is "
                    "context for review, not a claim that it treats an eye condition."
                ),
                "confidence": "scan_prioritized_canonical_mapping",
                "provenance": "e4l_scan_and_canonical_structure_map",
            })
    program_keys = []
    for label in history:
        text = _norm(label)
        for phrase, keys in PROGRAM_MATCHES.items():
            if phrase in text:
                program_keys.extend(keys)
    for key in dict.fromkeys(program_keys):
        candidates.append({
            "suggestion_id": hashlib.sha256(
                ("program|" + key).encode()).hexdigest()[:16],
            "category": "authored_support_program_review",
            "name": key,
            "status": "pending_practitioner_approval",
            "client_visible": False,
            "priority": None,
            "trigger": {"reported_history": history},
            "rationale": (
                "Reported history matches an existing authored eye-support program. "
                "Review its stored items, modifiers, and consult flag."
            ),
            "confidence": "history_term_match",
            "provenance": "authored_condition_program_reference",
        })
    structures = {
        _norm(structure)
        for finding in findings
        for structure in finding.get("ocular_structures") or []
    }
    return {
        "phase": 2, "audience": "practitioner_only",
        "approval_required": True, "client_payload_includes_drafts": False,
        "safety_status": (
            "suppressed_urgent_review" if urgent_matches else "needs_review"),
        "urgent_matches": urgent_matches,
        "current_products_for_interaction_review": current_products,
        "interaction_review_status": (
            "required" if current_products else "history_not_supplied"),
        "candidates": candidates,
        "program_extension_opportunities": _program_extension_opportunities(
            safety_text, structures),
        "publication_rule": (
            "Each candidate requires reviewer identity, timestamp, client-safe wording, "
            "and contraindication/interaction review before client publication."
        ),
    }


def _urgent_referral_route(practitioner_satisfactory):
    if practitioner_satisfactory is True:
        return {
            "status": "current_practitioner",
            "instruction": (
                "Contact the current satisfactory practitioner promptly. "
                "Do not delay emergency or urgent conventional care."
            ),
            "practitioner_finder_url": None,
        }
    if practitioner_satisfactory is False:
        return {
            "status": "practitioner_finder",
            "instruction": (
                "Use the practitioner finder for additional support while also seeking "
                "the urgent conventional assessment indicated by the symptom rule."
            ),
            "practitioner_finder_url": "/practitioner-finder",
        }
    return {
        "status": "ask_first",
        "question": (
            "Are you happy with your current practitioners and able to contact one "
            "promptly about this concern?"
        ),
        "if_yes": "Contact the current practitioner promptly.",
        "if_no": "Offer the practitioner finder.",
        "practitioner_finder_url": "/practitioner-finder",
        "urgent_care_guardrail": (
            "This routing question must not delay emergency or urgent conventional care."
        ),
    }


def _program_extension_opportunities(history_text, structures):
    opportunities = []

    def add(key, label, basis, scope, safeguards):
        opportunities.append({
            "program_key": key, "label": label,
            "status": "program_design_review", "client_visible": False,
            "basis": basis, "proposed_scope": scope,
            "required_safeguards": safeguards,
            "publication_ready": False,
        })

    if any(term in history_text for term in
           ("diabetes", "diabetic", "blood sugar", "glucose", "insulin")):
        add(
            "glucose-metabolic-regulation", "Glucose and metabolic regulation",
            "Reported metabolic history; not inferred from the scan.",
            "Adjunctive support relevant to retinal, lens, vascular, and healing context.",
            ["Medication and hypoglycemia review", "Glucose-monitoring plan",
             "Do not replace diabetes care"])
    if structures & {"retina", "optic nerve", "vision", "visual processing",
                     "visual pathway"}:
        add(
            "cellular-energy-metabolism", "Cellular energy and metabolism",
            "Mapped neural/retinal structure-function context.",
            "A cross-condition program for mitochondrial, membrane, oxidative, and "
            "cellular-energy support after an authored evidence review.",
            ["Define eligible contexts", "Review interactions and duplication",
             "Do not claim restoration of damaged tissue"])
    if any(term in history_text for term in
           ("toxin", "heavy metal", "mercury", "lead", "arsenic", "cadmium")):
        add(
            "toxin-exposure-support", "Toxin and heavy-metal exposure support",
            "Reported toxin/exposure history; never inferred from an E4L pattern.",
            "Assessment-led exposure reduction and medically appropriate support.",
            ["Confirm exposure and source", "Require qualified testing/oversight",
             "Avoid unsupervised chelation or detox claims",
             "Review kidney, liver, pregnancy, and medication risks"])
    if "retina" in structures or any(
            term in history_text for term in ("retinopathy", "vascular", "circulation")):
        add(
            "ocular-circulation-support", "Ocular circulation and vascular support",
            "Retinal mapping or reported vascular history.",
            "Adjunctive cardiovascular/metabolic risk and ocular-perfusion support.",
            ["Coordinate with conventional evaluation", "Review bleeding risk",
             "Do not imply treatment of vascular occlusion"])
    if structures & {"vision", "visual processing", "visual pathway"}:
        add(
            "neurovisual-processing-support", "Neurovisual processing support",
            "Mapped visual-processing or pathway relationship.",
            "Visual-motor, vestibular, circadian, and sensory-integration support.",
            ["Screen for neurologic red flags", "Define referral boundaries",
             "Separate training/support from disease treatment"])
    return opportunities


def build_practitioner_suggestions(cx, email, today, db_path=None):
    """Build draft suggestions. Never call this from a client portal payload."""
    history, all_text, current_products, practitioner_satisfactory = (
        _practitioner_history(cx, email))
    scan = biofield_e4l.scan_context(email, today, db_path=db_path, limit=100)
    if not scan.get("found"):
        return None
    patterns = scan.get("findings") or []
    mappings = _mappings_for(
        [p.get("code") for p in patterns if p.get("code")], db_path=db_path)
    findings = [
        finding for pattern in patterns
        if (finding := _finding(
            pattern, mappings.get(pattern.get("code"), []), history))
    ]
    return _suggestion_contract(
        findings, history, all_text, current_products,
        practitioner_satisfactory=practitioner_satisfactory)


def build_portal_block(cx, email, today, saved_collapsed=None, db_path=None):
    """Build the safe client view and resolve its initial/persisted open state."""
    history = reported_eye_history(cx, email)
    scan = biofield_e4l.scan_context(email, today, db_path=db_path, limit=100)
    if not scan.get("found"):
        return None
    patterns = scan.get("findings") or []
    mappings = _mappings_for(
        [p.get("code") for p in patterns if p.get("code")], db_path=db_path)
    findings = []
    for pattern in patterns:
        finding = _finding(pattern, mappings.get(pattern.get("code"), []), history)
        if finding:
            findings.append(finding)
    findings.sort(key=lambda row: (
        0 if row["classification"] == "direct_ocular" else 1,
        row["priority_rank"] if isinstance(row["priority_rank"], (int, float)) else 10**9,
        row["code"] or "",
    ))
    default_open = bool(history)
    is_open = (not saved_collapsed) if saved_collapsed is not None else default_open
    return {
        "section_key": SECTION_KEY,
        "title": "Your Eye & Vision Pattern Report",
        "intro": ("This focused view gathers the eye- and vision-related "
                  "structure/function relationships present in your E4L scan."),
        "status": "ready" if findings else "no_eye_patterns",
        "scan_date": scan.get("scan_date"),
        "findings": findings,
        "empty_state": ("This scan did not contain a canonical eye- or "
                        "vision-mapped Infoceutical pattern.") if not findings else None,
        "limitations": LIMITATIONS,
        "open": is_open,
        "default_open": default_open,
        "preference_saved": saved_collapsed is not None,
        "history_eye_issue": bool(history),
    }
