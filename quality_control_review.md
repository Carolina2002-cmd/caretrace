# Quality Control — Hiring Manager Score & Red Flag Review

## Score (0–100)

| Category | Weight | Score | Notes |
|---|---|---|---|
| Healthcare relevance | 15 | 13 | Diagnostic fragmentation is a real, if less commonly packaged, decision-support problem; connects to a documented clinical phenomenon (diagnostic odyssey) rather than an invented one. Slight deduction: no single named regulatory mechanism as clean as HRRP to anchor the payment story. |
| Technical depth | 15 | 14 | Supervised classifier + SHAP, unsupervised typology clustering, and a separate time-series drift model — three distinct technique families used deliberately, not for show. A real leakage bug was caught and fixed mid-build (see below), which is itself evidence of technical rigor. Slight deduction: single train/test split, no cross-validation. |
| POC quality | 20 | 16 | End-to-end, runs locally, real (not fabricated) metrics off an actual training run, working SHAP + typology + drift + review-packet dashboard. Deduction: entirely synthetic data including the label itself (see Weakest Area). |
| Business value | 15 | 12 | Treatment/Payment/Operations framing is specific and consistent, but payment linkage is a plausible inference ("could complement payment-integrity workflows") rather than tied to a named external program the way HRRP anchors readmission risk. |
| Responsible AI | 10 | 9 | Explicit, load-bearing boundary between deterministic detection and the bounded agentic layer; provider drift is carefully framed as context, not accusation; human-in-the-loop stated everywhere. |
| Report quality | 10 | 9 | Fits 2 pages + bibliography, real citations, states the novelty claim carefully ("less commonly explored," not "first ever"). |
| Presentation quality | 5 | 5 | 10 clean, varied slides; consistent visual motif; real architecture diagram, not a stock graphic. The POC itself was also rebuilt as a branded "control center" UI (header status bar, signal-strength evidence system, live decision simulator) rather than a default Streamlit layout — verified rendering with zero runtime exceptions across all four views and every interactive control (Streamlit AppTest + Playwright screenshots). |
| Interview potential | 10 | 9 | Every technical choice — including the pivot from unsupervised to supervised, and the leakage fix — has a specific, defensible rationale in interview_prep.md. |
| **Total** | **100** | **87** | |

## Red Flag Review

| Red flag | Present? | Notes |
|---|---|---|
| Generic AI language ("revolutionize," "seamless," "cutting-edge") | No | Avoided throughout. |
| Unsupported claims | No | All Cotiviti-specific statements use "could explore" / "potential" framing. |
| Fake statistics | No | ROC-AUC 0.899, PR-AUC 0.502, precision/recall figures all come from an actual held-out test run — reproducible via `python -m src.pipeline`. |
| Fake citations | No | Bibliography entries (rare-disease diagnostic-odyssey literature, CMS SynPUF, Isolation Forest, Obermeyer et al. 2019, FDA 2021, NIST 2023) verified against real sources. |
| Overengineering | No | Three technique families (classifier, clustering, time-series drift) each do a distinct job; no LLM forced into detection; the agentic layer is one narrow, clearly-scoped function. |
| Unrealistic clinical claims | No | Scoped as decision support throughout — report, deck, dashboard, README, and every review-packet narrative ends with a human-sign-off line. |
| Missing healthcare context | No | Each engineered feature has a stated clinical rationale (e.g., diagnosis-code churn indicating a diagnosis that isn't stabilizing). |
| Missing Cotiviti relevance | No | TPO framing appears in report, deck, and README. |
| Poor explainability | No | SHAP per-episode, surfaced directly in the dashboard, plus the typology label as a second, complementary explanation layer. |
| No human oversight | No | Every review packet ends with an explicit human-sign-off requirement; provider drift is explicitly labeled "context, not accusation." |
| Weak POC | Minor | Entirely synthetic data, including the training label — see Weakest Area below. |
| Too many buzzwords | No | "Agentic AI" is used only for the one function that fits it, with an explicit statement of what it does *not* do. |
| Claims that can't be defended in an interview | No | Every claim (including the "why supervised over unsupervised" pivot and the leakage fix) has a specific answer in interview_prep.md. |

## Weakest area and how it was addressed

**Weakest area: the training label is entirely synthetic, including the
audit-simulation process that generates it.** There is no real "chart review
confirmed a diagnostic odyssey" outcome anywhere — "diagnostic odyssey" is
not a label anyone codes at scale today, so even a real-data version of this
project would need to bootstrap a labeling process (e.g., a pilot chart
review) before a supervised classifier could be trained at all.

This was addressed by (1) building the label from a genuinely separate
simulated process rather than deriving it from the same features the model
sees, specifically to avoid producing an artificially inflated, undefensible
metric; (2) actually catching a leakage bug during development — an earlier
version of the generator produced a near-1.0 AUC because the label formula
was too tightly coupled to the visible features, which was a clear signal
something was wrong, not a result to report; and (3) stating this limitation
explicitly and consistently in the README, `data/README.md`, the report, and
interview_prep.md, with real next steps (bootstrap a labeled pilot dataset
via an actual chart-review process) rather than presenting the synthetic
metrics as a claim about real-world performance.

## Final recommendation

The submission is ready as-is. Two things to do personally before recording
the video: (1) run `streamlit run app.py` end-to-end once, live, on your own
machine so the on-camera demo isn't your first time seeing it render outside
this environment; and (2) rehearse the leakage-bug story from question 6 in
interview_prep.md out loud once — it's one of the strongest technical-depth
signals in the whole submission, and it should come across as fluent, not
read off a page.
