"""
Typology clustering: among episodes the classifier considers at-risk,
cluster them into a small number of interpretable "fragmentation
typologies" -- distinct enough that a care coordinator can triage
differently by type, rather than treating every flag identically.

Naming is done post-hoc from cluster feature-center profiles, not fixed in
advance, so the labels genuinely reflect what the data groups together.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

TYPOLOGY_FEATURES = [
    "specialty_fanout_ratio", "diag_to_therap_ratio", "pingpong_rate",
    "interval_cv", "top_provider_share", "diagnosis_churn_rate",
]

# Human-readable typology naming rules, applied to standardized cluster
# centers (highest-magnitude distinguishing feature(s) drive the name).
NAME_RULES = [
    (("pingpong_rate", "top_provider_share"), "Referral Ping-Pong", "No single provider is coordinating care; visits bounce between the same specialties without escalation."),
    (("diag_to_therap_ratio", "diagnosis_churn_rate"), "Diagnostic Loop", "Heavy, recurring diagnostic/imaging/lab activity with a diagnosis that keeps shifting and never stabilizes."),
    (("specialty_fanout_ratio",), "Specialist Fan-Out", "Rapid escalation across many specialties with little continuity of care."),
]


def fit_typologies(scored_df: pd.DataFrame, n_clusters: int = 3, risk_col: str = "fragmentation_proba", risk_threshold: float = 0.3):
    at_risk = scored_df[scored_df[risk_col] >= risk_threshold].copy()
    if len(at_risk) < n_clusters * 3:
        # not enough at-risk episodes for stable clusters; relax threshold
        at_risk = scored_df.sort_values(risk_col, ascending=False).head(max(n_clusters * 10, 30)).copy()

    X = at_risk[TYPOLOGY_FEATURES].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    at_risk["typology_cluster"] = km.fit_predict(Xs)

    # name each cluster from its standardized center
    centers = pd.DataFrame(km.cluster_centers_, columns=TYPOLOGY_FEATURES)
    typology_names = {}
    used_names = set()
    for cluster_id, row in centers.iterrows():
        best_rule, best_score = None, -np.inf
        for feats, name, desc in NAME_RULES:
            score = row[list(feats)].mean()
            if score > best_score and name not in used_names:
                best_rule, best_score = (name, desc), score
        if best_rule is None:  # all names used, fall back
            best_rule = (f"Fragmentation Pattern {cluster_id}", "Elevated fragmentation signals combine without one dominant driver.")
        typology_names[cluster_id] = best_rule
        used_names.add(best_rule[0])

    at_risk["typology_name"] = at_risk["typology_cluster"].map(lambda c: typology_names[c][0])
    at_risk["typology_description"] = at_risk["typology_cluster"].map(lambda c: typology_names[c][1])

    result = scored_df.merge(
        at_risk[["episode_id", "typology_cluster", "typology_name", "typology_description"]],
        on="episode_id", how="left",
    )
    return result, km, scaler, typology_names
