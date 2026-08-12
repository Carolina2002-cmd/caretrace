"""
Synthetic claims generator, v2.

Key design change from v1 (make_demo_data.py): ground truth no longer comes
from a deterministic function of the same features the model will see.

Real-world analogy: a payer doesn't get a "this patient is fragmented" label
for free -- it would come from a SEPARATE process (e.g. a case manager's
chart review, a retrospective audit) that imperfectly reflects an underlying
latent condition, using information the claims-derived feature table doesn't
fully capture (chart notes, patient-reported experience, clinical judgment).

So this generator:
  1. Draws a hidden latent `care_complexity` value per patient (not a binary
     odyssey/not-odyssey flag) that stochastically influences the visible
     claims pattern (specialty fan-out, diagnostic mix, visit spacing).
  2. Independently simulates a chart-audit process: a reviewer looks at a
     sample of episodes and records a `confirmed_fragmentation` outcome that
     depends on the latent complexity PLUS independent audit noise PLUS a
     small amount of information the claims features don't contain (a
     synthetic "chart_context_score" that is NOT joined into the feature
     table). This keeps the label statistically related to, but not a
     deterministic function of, the visible feature set.

Two outputs:
  - data/claims.csv                  (visible to the model / feature builder)
  - data/chart_audit_outcomes.csv    (episode_id, confirmed_fragmentation,
                                       audited_by, audit_notes -- joined onto
                                       features ONLY at training time, via
                                       episode_id, never used as a feature)
"""

import random
import numpy as np
import pandas as pd
from datetime import timedelta

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

SPECIALTIES = [
    "Primary Care", "Cardiology", "Neurology", "Gastroenterology",
    "Rheumatology", "Endocrinology", "Orthopedics", "Pulmonology",
    "Dermatology", "General Surgery", "Physical Therapy", "Psychiatry",
]
PRESENTING_COMPLAINTS = [
    "Chronic Fatigue", "Joint Pain", "Abdominal Pain", "Chest Pain",
    "Headache", "Back Pain", "Shortness of Breath", "Skin Rash",
]
DIAGNOSTIC_TYPES = {"Diagnostic", "Imaging", "Lab"}
THERAPEUTIC_TYPES = {"Therapeutic", "Procedure"}

N_PROVIDERS_PER_SPECIALTY = 25


def _make_provider_pool():
    """Fixed provider roster so provider_drift.py can track the same
    provider IDs across the whole simulated period."""
    pool = {}
    for spec in SPECIALTIES:
        pool[spec] = [f"P-{spec[:3].upper()}-{i:02d}" for i in range(1, N_PROVIDERS_PER_SPECIALTY + 1)]
    return pool


def _gen_episode(patient_id, episode_id, complaint, complexity, start_date, provider_pool):
    """complexity in [0,1]: latent driver of visible fragmentation-like
    claims patterns. NOT the label itself."""
    events = []
    n_visits = int(np.clip(np.random.normal(3 + complexity * 13, 1.6), 2, 22))
    n_specialties = int(np.clip(np.random.normal(1 + complexity * 6, 0.8), 1, 8))
    specialties_for_episode = random.sample(SPECIALTIES, k=min(n_specialties, len(SPECIALTIES)))

    date = start_date
    diag_code_base = f"DX-{random.randint(100, 999)}"
    diagnosis_locked = False

    for i in range(n_visits):
        gap_mean = 10 + complexity * 20
        gap = int(np.random.exponential(scale=gap_mean)) + np.random.choice([0, 0, int(10 + complexity * 20)])
        date = date + timedelta(days=int(max(gap, 1)))
        specialty = random.choice(specialties_for_episode)
        provider_id = random.choice(provider_pool[specialty])

        diag_prob = min(0.85, 0.25 + complexity * 0.55)
        therap_prob_base = max(0.05, 0.5 - complexity * 0.4)
        if i < 2:
            claim_type = np.random.choice(["E&M", "Diagnostic", "Lab"], p=[0.5, 0.3, 0.2])
        else:
            claim_type = np.random.choice(
                ["Diagnostic", "Imaging", "Lab", "E&M", "Therapeutic", "Procedure"],
                p=[
                    diag_prob * 0.4, diag_prob * 0.3, diag_prob * 0.3,
                    (1 - diag_prob) * 0.3,
                    (1 - diag_prob) * 0.4 * (1 - complexity * 0.5),
                    (1 - diag_prob) * 0.3 * (1 - complexity * 0.5),
                ] / np.sum([
                    diag_prob * 0.4, diag_prob * 0.3, diag_prob * 0.3,
                    (1 - diag_prob) * 0.3,
                    (1 - diag_prob) * 0.4 * (1 - complexity * 0.5),
                    (1 - diag_prob) * 0.3 * (1 - complexity * 0.5),
                ]),
            )

        diagnosis_change = (not diagnosis_locked) and (np.random.rand() < (0.25 + complexity * 0.4))
        if diagnosis_change:
            diag_code = f"DX-{random.randint(100, 999)}"
        else:
            diag_code = diag_code_base
        if claim_type in THERAPEUTIC_TYPES and np.random.rand() < 0.6:
            diagnosis_locked = True

        allowed_amount = round(np.random.gamma(shape=2.0, scale=90 if claim_type in DIAGNOSTIC_TYPES else 140), 2)

        events.append({
            "patient_id": patient_id, "episode_id": episode_id,
            "claim_date": date.strftime("%Y-%m-%d"),
            "provider_id": provider_id, "provider_specialty": specialty,
            "presenting_complaint": complaint, "diagnosis_code": diag_code,
            "claim_type": claim_type, "allowed_amount": allowed_amount,
        })
    return events, date


def _simulate_chart_audit(episode_id, complexity, n_visits, n_specialties, span_days):
    """Independent-ish audit process. Uses the SAME latent complexity plus
    genuinely independent noise and a synthetic 'chart context' signal that
    is never exposed to the feature builder -- this is what keeps the label
    from being a deterministic function of the visible claims features."""
    chart_context_score = np.random.beta(2, 2)  # unobserved-to-model signal
    audit_noise = np.random.normal(0, 0.10)
    latent_signal = 0.72 * complexity + 0.15 * chart_context_score + audit_noise
    confirmed = latent_signal > 0.50
    audited_by = random.choice(["CM-Reviewer-A", "CM-Reviewer-B", "CM-Reviewer-C"])
    notes = (
        "Chart review supports care-coordination gap." if confirmed
        else "Chart review does not support a coordination gap at this time."
    )
    return {
        "episode_id": episode_id,
        "confirmed_fragmentation": bool(confirmed),
        "audited_by": audited_by,
        "audit_notes": notes,
    }


def generate(n_patients=900, out_dir="data"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    provider_pool = _make_provider_pool()

    all_events, all_audits = [], []
    start_base = pd.Timestamp("2024-01-01")
    episode_counter = 0

    for pid in range(1, n_patients + 1):
        patient_id = f"BENE-{pid:05d}"
        n_episodes = np.random.choice([1, 1, 1, 2], p=[0.55, 0.2, 0.15, 0.1])
        cursor = start_base + timedelta(days=int(np.random.randint(0, 60)))
        for _ in range(n_episodes):
            episode_counter += 1
            episode_id = f"EP-{episode_counter:06d}"
            complaint = random.choice(PRESENTING_COMPLAINTS)
            # complexity latent: mostly low, right-tailed (most episodes are simple)
            complexity = float(np.clip(np.random.beta(1.6, 4.5), 0, 1))
            events, last_date = _gen_episode(patient_id, episode_id, complaint, complexity, cursor, provider_pool)
            all_events.extend(events)

            ep_df = pd.DataFrame(events)
            audit = _simulate_chart_audit(
                episode_id, complexity, len(events),
                ep_df["provider_specialty"].nunique(),
                (pd.to_datetime(ep_df["claim_date"]).max() - pd.to_datetime(ep_df["claim_date"]).min()).days,
            )
            all_audits.append(audit)
            cursor = last_date + timedelta(days=int(np.random.randint(30, 120)))

    claims_df = pd.DataFrame(all_events).sort_values(["patient_id", "episode_id", "claim_date"]).reset_index(drop=True)
    audits_df = pd.DataFrame(all_audits)

    claims_df.to_csv(f"{out_dir}/claims.csv", index=False)
    audits_df.to_csv(f"{out_dir}/chart_audit_outcomes.csv", index=False)

    rate = audits_df["confirmed_fragmentation"].mean()
    print(f"Generated {len(claims_df)} claims across {claims_df['episode_id'].nunique()} episodes "
          f"({claims_df['patient_id'].nunique()} patients). Chart-audit-confirmed fragmentation rate: {rate:.1%}")
    return claims_df, audits_df


if __name__ == "__main__":
    generate()
