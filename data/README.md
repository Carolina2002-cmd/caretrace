# Data — transparency note

**All data in this repository is SYNTHETIC.** `demo_claims.csv` is generated
by `src/make_demo_data.py` and contains no real patient, provider, or payer
information.

- **Schema** loosely mirrors publicly documented CMS claims structures
  (beneficiary ID, claim date, provider ID/specialty, diagnosis code, claim
  type, allowed amount) — similar in spirit to CMS's public **Synthetic
  Public Use Files (SynPUF)**, https://www.cms.gov/data-research/statistics-trends-reports/medicare-claims-synthetic-public-use-files
- **Values are generated**, not sampled from SynPUF or any real dataset —
  this sandbox environment could not reach external data-hosting domains, so
  a transparent synthetic generator was used instead, matching this
  project's Responsible-AI commitment to never fabricate a claim about real
  data provenance.
- A `_ground_truth_odyssey` label is embedded **only** to validate the
  detection pipeline in this POC (Part 8 requirement: prove the model finds
  what it's supposed to find). It is dropped before the unsupervised/graph
  pipeline runs and would not exist in a real production deployment, where
  "diagnostic odyssey" is not a labeled outcome anyone has coded for.
- **To adapt to real data:** point `src/pipeline.py`'s `DATA_PATH` at a
  de-identified claims extract with equivalent columns
  (`patient_id, claim_date, provider_id, provider_specialty,
  diagnosis_code, claim_type, allowed_amount`). No other code changes
  required — this is the intended real-data on-ramp.
