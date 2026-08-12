"""
Review-packet generator -- the bounded "agentic" layer.

SCOPE, stated explicitly (see report for full Responsible-AI discussion):
  - Detection (classifier probability, SHAP attributions, typology cluster,
    provider drift) is 100% deterministic, reproducible, and computed
    upstream of this module.
  - This module's ONLY job is to assemble those already-computed,
    structured outputs into a short, readable packet for a human care
    coordinator -- selecting which SHAP factors to surface in plain
    language, naming the typology, and attaching provider-drift context
    IF the episode's providers show it. It adds zero new detection logic.
  - Template-based in this offline POC (no external API call needed to run
    the demo reproducibly). Swappable to an LLM API call over this exact
    structured input -- see README for the production note.
  - Never outputs a diagnosis. Never recommends contacting the patient
    directly. Always ends with a human-review instruction.
"""

import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "n_visits": "number of visits",
    "span_days": "total episode duration (days)",
    "n_providers": "number of distinct providers seen",
    "n_specialties": "number of distinct specialties seen",
    "specialty_fanout_ratio": "specialist fan-out relative to visit count",
    "diag_to_therap_ratio": "diagnostic-to-therapeutic claim ratio",
    "diagnosis_churn_rate": "diagnosis-code churn (how often the working diagnosis changed)",
    "interval_cv": "irregularity of visit spacing",
    "interval_trend": "trend in visit spacing (lengthening vs. tightening)",
    "days_to_first_therapy": "days elapsed before any therapeutic/procedural claim",
    "has_reached_therapy": "whether a therapeutic/procedural claim has been reached",
    "pingpong_rate": "rate of bouncing back to a prior specialty without escalation",
    "top_provider_share": "share of visits held by the single most-seen provider",
    "total_allowed_amount": "total allowed cost of the episode",
    "avg_claim_cost": "average claim cost",
}


def top_shap_factors(row_idx, feature_cols, shap_values, X_row, k=3):
    """Return the top-k features by |SHAP value| for one episode, signed."""
    vals = shap_values[row_idx]
    order = np.argsort(-np.abs(vals))[:k]
    out = []
    for i in order:
        col = feature_cols[i]
        direction = "increases" if vals[i] > 0 else "decreases"
        out.append({
            "feature": col,
            "label": FEATURE_LABELS.get(col, col),
            "value": X_row[col],
            "shap_value": float(vals[i]),
            "direction": direction,
        })
    return out


def compose_review_packet(episode_row, shap_factors, provider_drift_context=None) -> dict:
    proba = episode_row["fragmentation_proba"]
    typology = episode_row.get("typology_name")
    typology_desc = episode_row.get("typology_description")

    factor_sentences = []
    for f in shap_factors:
        val = f["value"]
        val_str = f"{val:.2f}" if isinstance(val, (float, np.floating)) else str(val)
        factor_sentences.append(f"{f['label']} ({val_str}) {f['direction']} the model's fragmentation estimate")

    lines = [
        f"Episode {episode_row['episode_id']} (patient {episode_row['patient_id']}, "
        f"presenting complaint: {episode_row['presenting_complaint']}) — "
        f"model-estimated fragmentation probability: {proba:.0%}.",
    ]
    if pd.notna(typology):
        lines.append(f"Pattern type: {typology} — {typology_desc}")
    lines.append("Top contributing factors (SHAP): " + "; ".join(factor_sentences) + ".")

    if provider_drift_context:
        lines.append(
            f"Provider context: {provider_drift_context['provider_id']} "
            f"({provider_drift_context['provider_specialty']}) shows a coding-pattern drift flag "
            f"(diagnostic-claim share moved from {provider_drift_context['baseline_diag_share']:.0%} "
            f"baseline to {provider_drift_context['latest_diag_share']:.0%} recently) — this is "
            "supporting context, not an accusation, and should be reviewed independently."
        )

    action = (
        "HIGH PRIORITY — recommend care-navigator review and single point-of-contact assignment."
        if proba >= 0.6 else
        "MODERATE PRIORITY — recommend adding to the care-manager review queue."
        if proba >= 0.3 else
        "No action recommended at this time."
    )
    lines.append(f"Recommended next step: {action}")
    lines.append("This packet is a decision-support draft for human care-coordinator review only. "
                  "It does not diagnose, and no outreach should occur without human sign-off.")

    return {
        "episode_id": episode_row["episode_id"],
        "narrative": " ".join(lines),
        "recommended_action": action,
        "shap_factors": shap_factors,
    }
