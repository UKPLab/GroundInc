"""Label sets and task modes shared by prompt rendering and target building."""

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

PROMPT_NAME_TO_INFERENCE_STYLE = {
    "veracity": "temporal_veracity_only",
    "no_evidence": "inconsistency_veracity_only",
    "full": "temporal_inconsistency_veracity",
    "inconsistency": "temporal_inconsistency_veracity",
    "inconsistency_no_grounding": "inconsistency_veracity_only",
}
