# Interview Preparation — 20 Likely Questions & Answers

**1. Why this problem?**
Because it's a real decision-support gap that conventional healthcare
analytics structurally can't see: per-claim utilization review and
single-score risk models evaluate encounters independently, so a patient
bouncing between specialists for months with no converging diagnosis never
trips any individual rule. It's also explicitly outside the common
internship-project categories (readmission, sepsis, chatbot, image
classification) while staying technically substantive and payer-relevant.

**2. Why this dataset?**
I generated synthetic claims data with a schema loosely modeled on public
CMS claims structures (patient/episode ID, claim date, provider ID/specialty,
diagnosis code, claim type, allowed amount) — similar in spirit to CMS's
public Synthetic Public Use Files. My sandbox couldn't reach external
data-hosting domains, so I built a transparent synthetic generator instead
of pretending to use real data, and I was explicit about that everywhere the
numbers appear.

**3. Why an XGBoost classifier here, and not the unsupervised approach you
mentioned earlier?**
I actually built both. My first pass used unsupervised anomaly detection
(Isolation Forest + peer-cohort z-scores) because I didn't want to assume a
labeled outcome exists in the real world. But once I formalized what a real
label *would* look like — a case manager's chart-review confirmation — a
supervised classifier became the stronger choice for this POC: it's more
directly evaluable (ROC-AUC, precision/recall against a held-out set) and
SHAP explainability on a tree model is cleaner than decomposing an ensemble
anomaly score. The unsupervised approach is still the right call if no
labeled outcome exists yet in a real deployment — I'd frame this as
"phase 2, once enough audited cases accumulate to train a supervised model."

**4. Why these metrics — why not just accuracy?**
The chart-audit-confirmed fragmentation rate is about 9%, so accuracy is
misleading — a model that never flags anyone would score ~91% accuracy and
be useless. I used ROC-AUC and PR-AUC for overall discrimination (PR-AUC
specifically because it's more informative than ROC-AUC under class
imbalance), and precision/recall at the operating threshold for the
practical review-queue tradeoff.

**5. Precision vs. recall — which matters more here, and why?**
Recall matters more. A false negative — missing a genuinely fragmented care
episode — means that patient stays stuck in an unproductive, costly
diagnostic loop. A false positive means a care coordinator spends a few
minutes reviewing a case that turns out fine. That asymmetry is why I'd
tune the threshold below a naive 0.5 in a real deployment, based on the
care-management team's actual review capacity.

**6. How do you know the label isn't leaking into the features?**
I built the audit label from a process deliberately separate from the
feature builder: `claims_generator.py` draws a latent "care complexity"
value that stochastically shapes the visible claims pattern, and
*independently* simulates a chart-audit outcome using that same latent plus
audit noise plus an unobserved "chart context" signal that's never joined
into the feature table. I actually caught a leakage bug in an earlier
version — my first label formula was a near-deterministic function of two
features I was also feeding the model, which produced a suspicious 1.0 AUC.
I rebuilt the generator once I saw that, and the corrected version trains to
a realistic 0.899.

**7. How do you explain individual predictions?**
SHAP (TreeExplainer on the XGBoost model). For any single episode, the
dashboard shows the top contributing features, signed by direction — e.g.,
"diagnostic-to-therapeutic claim ratio: +0.61" — so a reviewer sees why a
score was generated, not just the number.

**8. What are the fragmentation "typologies," and why cluster at all instead
of just using the classifier score?**
A single probability tells a coordinator *how much* to worry, not *what
kind* of problem they're looking at. I cluster at-risk episodes (K-Means
over a subset of the interpretable features) and name each cluster
post-hoc from its feature-center profile — I ended up with three distinct,
sensibly named types: Specialist Fan-Out, Referral Ping-Pong, and Diagnostic
Loop. That lets a coordinator triage differently — a ping-pong case might
need a single point-of-contact assignment, while a diagnostic-loop case
might need a fresh clinical review.

**9. What's the provider-drift piece, and is it a fraud-detection feature?**
No — I was careful to scope it as context, not an accusation. It's a
separate time-series check: each provider's monthly diagnostic-claim share
compared to their own first-three-month baseline, flagged when it drifts by
more than ~2 standard deviations for two consecutive months. If a flagged
episode touches a provider with a drift flag, the review packet surfaces
that as corroborating context — explicitly labeled as something to review
independently, never as a finding on its own.

**10. Why an "agentic" layer at all — why not just show the SHAP output
directly?**
Because a list of feature names and SHAP values isn't what a busy care
coordinator wants to read first. The review-packet generator's only job is
synthesis: it takes the already-computed classifier probability, SHAP
factors, typology, and drift context and assembles them into a short
narrative. It adds zero new detection logic — everything it says traces
back to a structured value computed upstream. That's a deliberate boundary,
not a shortcut.

**11. Why didn't you call a real LLM API for that layer?**
For this offline POC, I used a deterministic template so the demo is fully
reproducible without needing an API key or network access — you get the
same narrative every run. The interface (`compose_review_packet`) is
written so swapping in a real LLM call over the same structured input would
require no other code changes. I'd make that swap for a production version,
specifically because narrative composition is a task generative AI is
genuinely good at, unlike the detection itself.

**12. Why not use an LLM for the detection itself?**
Because the input is structured, tabular, sequential claims data, and a
gradient-boosted tree model is both more accurate and more directly
auditable for that data type. Detection needs to be reproducible and
explainable at the feature level — an LLM's reasoning path over the same
data would be harder to audit and wouldn't improve discrimination on a
problem this well-suited to tree-based models.

**13. What's model drift here, and how would you monitor it?**
Two kinds. Classifier drift: the relationship between claims features and
confirmed fragmentation could shift as coding practices or referral norms
change — monitored via calibration tracking on newly audited cases.
Provider-baseline drift: a provider's "normal" pattern can legitimately
shift over time (e.g., a new sub-specialty focus), so baselines would need
periodic re-anchoring, not a permanently fixed reference period.

**14. How would you check for bias?**
Subgroup performance analysis — sensitivity, specificity, and calibration
across age, geography, payer type, and presenting-complaint frequency —
before any pilot. I'd also specifically check whether patients with less
common presenting complaints are systematically over-flagged simply because
there's less peer data to build a stable typology cluster around them.

**15. What clinical/operational validation would this need before real
use?**
Real, de-identified claims data; real case-manager-confirmed chart-review
outcomes to retrain and validate against (not the simulated audit); a
subgroup fairness audit; and a pilot where flagged episodes are reviewed but
not acted on automatically, comparing model flags against care-coordinator
judgment before any wider rollout.

**16. Why human-in-the-loop, not full automation?**
A fragmentation flag is a pattern signal, not a diagnosis — it doesn't know
about context a care coordinator has, like an active care plan already
underway or a patient's specific access barriers. The review packet ends
every narrative with an explicit instruction that it requires human sign-off
before any outreach.

**17. How would you scale this?**
The pipeline is already batchable per episode. For production, I'd serve
scoring via an API triggered on episode updates rather than a Streamlit
demo, write flagged episodes to a care-management worklist, and log every
prediction with its SHAP explanation and typology for auditability.

**18. How does this support Treatment, Payment, and Operations?**
Treatment: prioritizes care-navigation outreach toward patients most likely
stuck in an unproductive diagnostic loop. Payment: surfaces a pattern tied
to low-value repeat diagnostic spend, complementing payment-integrity
workflows. Operations: gives utilization management and case management a
shared, explainable, typology-differentiated prioritization queue instead of
ad hoc case review.

**19. What would you change with access to real healthcare data?**
Retrain against real claims and real chart-audit outcomes instead of the
simulated label; validate whether the three typologies still separate
cleanly on real data or need re-clustering; run a genuine subgroup fairness
audit; and recalibrate provider-drift baselines against real coding-pattern
variance instead of synthetic volumes.

**20. What was the biggest technical limitation in this build, and how did
you handle it?**
Not having a real, labeled outcome to train against — "diagnostic odyssey"
isn't a coded outcome anyone has labeled at scale. I addressed it by
explicitly simulating the label as a separate process from the features
(rather than deriving it from them), catching and fixing a leakage bug when
my first attempt at that process was too tightly coupled to the visible
features, and documenting the whole design decision in `data/README.md` and
`quality_control_review.md` rather than presenting the metrics as if they
came from a real audited dataset.
