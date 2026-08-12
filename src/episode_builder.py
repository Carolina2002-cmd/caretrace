"""
Episode-level feature engineering. Builds ONE row per episode_id from the
raw claims stream. Deliberately contains no label / ground-truth logic --
that lives only in claims_generator.py's audit process and is joined in by
train.py, never computed here.
"""

import numpy as np
import pandas as pd

DIAGNOSTIC_TYPES = {"Diagnostic", "Imaging", "Lab"}
THERAPEUTIC_TYPES = {"Therapeutic", "Procedure"}


def build_episode_features(claims: pd.DataFrame) -> pd.DataFrame:
    df = claims.copy()
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    rows = []

    for episode_id, g in df.sort_values("claim_date").groupby("episode_id"):
        g = g.reset_index(drop=True)
        n_visits = len(g)
        span_days = (g["claim_date"].max() - g["claim_date"].min()).days
        n_specialties = g["provider_specialty"].nunique()
        n_providers = g["provider_id"].nunique()
        n_diag_codes = g["diagnosis_code"].nunique()

        specialty_fanout_ratio = n_specialties / max(n_visits, 1)

        n_diag_claims = g["claim_type"].isin(DIAGNOSTIC_TYPES).sum()
        n_therap_claims = g["claim_type"].isin(THERAPEUTIC_TYPES).sum()
        diag_to_therap_ratio = (n_diag_claims + 1) / (n_therap_claims + 1)

        diagnosis_churn_rate = n_diag_codes / max(n_visits, 1)

        if n_visits > 1:
            intervals = g["claim_date"].diff().dt.days.dropna().values
            interval_cv = float(np.std(intervals) / (np.mean(intervals) + 1e-6))
            interval_trend = float(np.corrcoef(np.arange(len(intervals)), intervals)[0, 1]) if len(intervals) > 1 else 0.0
        else:
            interval_cv, interval_trend = 0.0, 0.0

        therap_idx = g.index[g["claim_type"].isin(THERAPEUTIC_TYPES)]
        has_reached_therapy = len(therap_idx) > 0
        days_to_first_therapy = (
            (g.loc[therap_idx[0], "claim_date"] - g["claim_date"].min()).days if has_reached_therapy else span_days
        )

        specialties_seq = g["provider_specialty"].tolist()
        pingpong = sum(
            1 for i in range(2, len(specialties_seq))
            if specialties_seq[i] == specialties_seq[i - 2] and specialties_seq[i] != specialties_seq[i - 1]
        )
        pingpong_rate = pingpong / max(n_visits - 2, 1)

        top_provider_share = g["provider_id"].value_counts(normalize=True).iloc[0]
        total_allowed = g["allowed_amount"].sum()
        avg_claim_cost = g["allowed_amount"].mean()

        rows.append({
            "episode_id": episode_id,
            "patient_id": g["patient_id"].iloc[0],
            "presenting_complaint": g["presenting_complaint"].iloc[0],
            "n_visits": n_visits,
            "span_days": span_days,
            "n_specialties": n_specialties,
            "n_providers": n_providers,
            "specialty_fanout_ratio": specialty_fanout_ratio,
            "diag_to_therap_ratio": diag_to_therap_ratio,
            "diagnosis_churn_rate": diagnosis_churn_rate,
            "interval_cv": interval_cv,
            "interval_trend": interval_trend,
            "days_to_first_therapy": days_to_first_therapy,
            "has_reached_therapy": has_reached_therapy,
            "pingpong_rate": pingpong_rate,
            "top_provider_share": top_provider_share,
            "total_allowed_amount": total_allowed,
            "avg_claim_cost": avg_claim_cost,
        })

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "n_visits", "span_days", "n_specialties", "n_providers",
    "specialty_fanout_ratio", "diag_to_therap_ratio", "diagnosis_churn_rate",
    "interval_cv", "interval_trend", "days_to_first_therapy",
    "has_reached_therapy", "pingpong_rate", "top_provider_share",
    "total_allowed_amount", "avg_claim_cost",
]
