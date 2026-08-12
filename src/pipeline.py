"""
End-to-end v2 pipeline: claims -> episode features -> classifier + SHAP ->
typology clustering -> provider drift -> review packets -> saved artifacts
for the dashboard.

Run: python -m src.pipeline
"""

import os
import joblib
import numpy as np
import pandas as pd

from .claims_generator import generate as generate_claims
from .train import train as train_classifier
from .typology import fit_typologies
from .provider_drift import compute_provider_monthly_stats, detect_drift
from .review_packet import top_shap_factors, compose_review_packet

DATA_DIR, MODELS_DIR = "data", "models"


def run():
    if not os.path.exists(f"{DATA_DIR}/claims.csv"):
        generate_claims()

    scored, model, explainer, shap_values = train_classifier()
    scored["has_reached_therapy"] = scored["has_reached_therapy"].astype(int)

    scored, km, scaler, typology_names = fit_typologies(scored)

    claims = pd.read_csv(f"{DATA_DIR}/claims.csv")
    provider_stats = compute_provider_monthly_stats(claims)
    provider_drift = detect_drift(provider_stats)
    drift_map = provider_drift[provider_drift["provider_drift_flag"]].set_index("provider_id").to_dict("index")

    art = joblib.load(f"{MODELS_DIR}/classifier.pkl")
    feature_cols = art["feature_cols"]

    episode_to_providers = claims.groupby("episode_id")["provider_id"].unique().to_dict()

    packets = {}
    for row_idx, row in scored.iterrows():
        X_row = row[feature_cols]
        factors = top_shap_factors(row_idx, feature_cols, shap_values, X_row)
        drift_ctx = None
        for pid in episode_to_providers.get(row["episode_id"], []):
            if pid in drift_map:
                d = dict(drift_map[pid]); d["provider_id"] = pid
                drift_ctx = d
                break
        packets[row["episode_id"]] = compose_review_packet(row, factors, drift_ctx)

    scored["review_narrative"] = scored["episode_id"].map(lambda e: packets[e]["narrative"])
    scored["recommended_action"] = scored["episode_id"].map(lambda e: packets[e]["recommended_action"])

    scored.to_csv(f"{MODELS_DIR}/episodes_final.csv", index=False)
    provider_drift.to_csv(f"{MODELS_DIR}/provider_drift.csv", index=False)
    joblib.dump(packets, f"{MODELS_DIR}/review_packets.pkl")

    print(f"Pipeline complete. {len(scored)} episodes scored, "
          f"{int((scored['fragmentation_proba'] >= 0.5).sum())} above 0.5 probability, "
          f"{int(provider_drift['provider_drift_flag'].sum())} providers flagged for drift.")
    return scored


if __name__ == "__main__":
    run()
