IGNORE_INDEX = -100

VERACITY_LABELS = {"TRUE", "FALSE"}
MODALITY_LABELS = {"visual", "audio", "both", "none"}
TASK_MODES = {
    "veracity_only",
    "no_evidence",
    "full",
    "inconsistency",
    "inconsistency_no_grounding",
}

INCONSISTENCY_LABELS = {
    "Negation/Polarity Flip",
    "Comparative/Scalar Flip",
    "Numeric Distortion",
    "Entity Substitution",
    "Predicate/Role Substitution",
    "Temporal/Spatial Reframing",
    "Context Omission / Quote Mining",
    "Causal Overreach",
    "Affective/Framing Distortions",
    "Exaggeration / Minimisation",
    "No Inconsistency",
}

EXCLUDED_LABELS = {
    "Quantifier and Scope Distortions",
    "Epistemic Inflation",
    "Entity Disambiguation / Homonyms",
}
