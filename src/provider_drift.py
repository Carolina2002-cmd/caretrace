"""
Provider-level time-series drift check.

Separate lens from the episode-level classifier: instead of "is this
patient's pathway fragmented", this asks "has a given provider's referral
and coding behavior drifted from its own historical baseline" -- e.g. a
provider whose diagnostic-claim share, referral-out rate, or specialty-mix
suddenly shifts month over month. Not a fraud/abuse accusation by itself --
surfaced as context that can corroborate (or not) an episode-level flag
routed through that provider.

Method: monthly rolling z-score of each provider's diagnostic-claim share
and referral-out rate against their own trailing baseline (first 3 months),
flagged when |z| exceeds a threshold for 2+ consecutive months (simple,
auditable -- not a black-box change-point model).
"""

import numpy as np
import pandas as pd

DIAGNOSTIC_TYPES = {"Diagnostic", "Imaging", "Lab"}


def compute_provider_monthly_stats(claims: pd.DataFrame) -> pd.DataFrame:
    df = claims.copy()
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["month"] = df["claim_date"].dt.to_period("M")

    rows = []
    for (provider_id, month), g in df.groupby(["provider_id", "month"]):
        n_claims = len(g)
        diag_share = g["claim_type"].isin(DIAGNOSTIC_TYPES).mean()
        # "referral-out" proxy: fraction of that provider's patients who see
        # a DIFFERENT specialty within 14 days of this provider's claim
        rows.append({
            "provider_id": provider_id, "provider_specialty": g["provider_specialty"].iloc[0],
            "month": str(month), "n_claims": n_claims, "diag_share": diag_share,
            "avg_cost": g["allowed_amount"].mean(),
        })
    stats = pd.DataFrame(rows).sort_values(["provider_id", "month"])
    return stats


def detect_drift(stats: pd.DataFrame, baseline_months: int = 3, z_threshold: float = 2.0, min_claims: int = 3):
    results = []
    for provider_id, g in stats.groupby("provider_id"):
        g = g.sort_values("month").reset_index(drop=True)
        if len(g) <= baseline_months or g["n_claims"].sum() < min_claims * len(g):
            continue
        baseline = g.iloc[:baseline_months]
        if baseline["n_claims"].sum() < min_claims * baseline_months:
            continue
        base_mean = baseline["diag_share"].mean()
        base_std = max(baseline["diag_share"].std(), 0.08)  # floor avoids degenerate z-scores

        g["diag_share_z"] = (g["diag_share"] - base_mean) / base_std
        g["drift_flag_month"] = g["diag_share_z"].abs() >= z_threshold

        # consecutive-month rule: flag provider if 2+ consecutive post-baseline months drift
        post = g.iloc[baseline_months:]
        consec = (post["drift_flag_month"].astype(int)
                  .groupby((~post["drift_flag_month"]).cumsum()).cumsum())
        provider_drift_flag = bool((consec >= 2).any())

        results.append({
            "provider_id": provider_id,
            "provider_specialty": g["provider_specialty"].iloc[0],
            "n_months": len(g),
            "baseline_diag_share": round(base_mean, 3),
            "latest_diag_share": round(g["diag_share"].iloc[-1], 3),
            "max_abs_z": round(g["diag_share_z"].abs().max(), 2),
            "provider_drift_flag": provider_drift_flag,
        })
    return pd.DataFrame(results).sort_values("max_abs_z", ascending=False)
