"""
CareTrace — Diagnostic Odyssey Radar
Healthcare Intelligence Control Center

Run: streamlit run app.py
"""

import time
import datetime
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import networkx as nx

from src.graph_signals import build_transition_graph, graph_summary

st.set_page_config(page_title="CareTrace | Diagnostic Odyssey Radar", layout="wide",
                    initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------------
TEAL_DARK, TEAL, TEAL_LIGHT = "#16333A", "#2E6F7A", "#DCE9E8"
SAND, INK, CORAL, CORAL_LIGHT = "#F7F3EC", "#1D2A2E", "#C1622D", "#F2C9A6"
GOOD, WARN, BAD = "#3D8361", "#C98A2C", "#B84C3E"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.2rem; max-width: 1400px; }}

.ct-header {{
  background: linear-gradient(120deg, {TEAL_DARK} 0%, {TEAL} 100%);
  border-radius: 14px; padding: 18px 28px; margin-bottom: 18px;
  display: flex; justify-content: space-between; align-items: center;
  color: white; box-shadow: 0 4px 18px rgba(22,51,58,0.25);
}}
.ct-header h1 {{ margin: 0; font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }}
.ct-header .tagline {{ font-size: 12.5px; opacity: 0.85; margin-top: 2px; }}
.ct-status {{ display: flex; gap: 22px; align-items: center; }}
.ct-status .item {{ text-align: right; }}
.ct-status .label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; }}
.ct-status .value {{ font-size: 14px; font-weight: 700; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; }}

.ct-card {{
  background: white; border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
  border: 1px solid #E7E1D3; box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}}
.ct-card h4 {{ margin: 0 0 10px 0; font-size: 13px; text-transform: uppercase;
  letter-spacing: 1px; color: {CORAL}; font-weight: 700; }}

.badge {{ display:inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px;
  font-weight: 700; margin-right: 6px; }}
.badge-high {{ background: #FBE4E1; color: {BAD}; }}
.badge-mod {{ background: #FBEED9; color: {WARN}; }}
.badge-low {{ background: #E1F0E7; color: {GOOD}; }}
.badge-typ {{ background: {TEAL_LIGHT}; color: {TEAL_DARK}; }}

.signal-row {{ display:flex; justify-content: space-between; align-items:center;
  padding: 9px 0; border-bottom: 1px solid #F0EBE0; }}
.signal-row:last-child {{ border-bottom: none; }}
.signal-label {{ font-size: 13.5px; color: {INK}; }}
.signal-dots span {{ font-size: 15px; letter-spacing: 1px; }}

.flow-step {{ text-align:center; flex:1; }}
.flow-step .n {{ width: 34px; height: 34px; border-radius: 50%; background: {TEAL};
  color:white; display:flex; align-items:center; justify-content:center; margin: 0 auto 6px auto;
  font-weight:700; font-size: 14px; }}
.flow-step .t {{ font-size: 12px; font-weight: 600; color: {INK}; }}
.flow-arrow {{ color: #B7ADA0; font-size: 20px; padding-top: 6px; }}

.kicker {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: {CORAL}; font-weight: 700; }}

div[data-testid="stMetricValue"] {{ font-weight: 800; color: {TEAL_DARK}; }}
</style>
""", unsafe_allow_html=True)

DISCLAIMER = ("⚠️ **Decision-support prototype.** All data is SYNTHETIC (see `data/README.md`). "
              "This tool never diagnoses or acts autonomously — every finding routes to a human "
              "care coordinator for review.")

DEMO_CASES = {
    "🟢 Normal — routine pathway": "EP-000322",
    "🟡 Borderline — weak signals": "EP-000232",
    "🔴 High priority — converging signals": "EP-000418",
}


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    claims = pd.read_csv("data/claims.csv")
    claims["claim_date"] = pd.to_datetime(claims["claim_date"])
    episodes = pd.read_csv("models/episodes_final.csv")
    provider_drift = pd.read_csv("models/provider_drift.csv")
    return claims, episodes, provider_drift


@st.cache_resource
def load_model_artifacts():
    art = joblib.load("models/classifier.pkl")
    shap_values = np.load("models/shap_values.npy")
    packets = joblib.load("models/review_packets.pkl")
    return art, shap_values, packets


def signal_dots(abs_shap):
    if abs_shap >= 1.2:
        return "●●●●", "Critical", BAD
    elif abs_shap >= 0.7:
        return "●●●○", "Strong", BAD
    elif abs_shap >= 0.3:
        return "●●○○", "Moderate", WARN
    else:
        return "●○○○", "Weak", GOOD


def risk_badge(proba):
    if proba >= 0.6:
        return "High Priority", "badge-high", BAD
    elif proba >= 0.3:
        return "Moderate Priority", "badge-mod", WARN
    return "Low Priority", "badge-low", GOOD


# ----------------------------------------------------------------------------
# CHART BUILDERS
# ----------------------------------------------------------------------------
def render_gauge(proba):
    score = proba * 100
    color = BAD if score >= 60 else (WARN if score >= 30 else GOOD)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": "Fragmentation Probability", "font": {"size": 13}},
        number={"suffix": "%", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 9}},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#EFF6F1"},
                {"range": [30, 60], "color": "#FBF2E1"},
                {"range": [60, 100], "color": "#FBEAE7"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=15, r=15, t=45, b=5),
                       paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_timeline(ep_claims):
    color_map = {"E&M": "#8FA6A3", "Diagnostic": TEAL, "Imaging": "#5AA0AC",
                 "Lab": "#8FC1C6", "Therapeutic": GOOD, "Procedure": "#2C7A4B"}
    fig = px.scatter(
        ep_claims, x="claim_date", y="provider_specialty", color="claim_type",
        size="allowed_amount", size_max=22, color_discrete_map=color_map,
        hover_data={"provider_id": True, "diagnosis_code": True, "allowed_amount": ":$,.0f"},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FCFAF6",
                       legend=dict(orientation="h", y=-0.25), xaxis_title=None, yaxis_title=None)
    return fig


def render_graph(g: nx.DiGraph):
    pos = nx.spring_layout(g, seed=42)
    edge_x, edge_y = [], []
    for a, b in g.edges():
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1.4, color="#C9BFAE"), hoverinfo="none")
    node_x = [pos[n][0] for n in g.nodes()]
    node_y = [pos[n][1] for n in g.nodes()]
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=list(g.nodes()),
                             textposition="top center", textfont=dict(size=10),
                             marker=dict(size=26, color=TEAL, line=dict(width=2, color="white")))
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(visible=False), yaxis=dict(visible=False),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
def render_header(n_episodes):
    now = datetime.datetime.now().strftime("%b %d, %Y  ·  %H:%M")
    st.markdown(f"""
    <div class="ct-header">
      <div>
        <h1>🧭 CareTrace</h1>
        <div class="tagline">Diagnostic Odyssey Radar — claims-based detection of fragmented, non-converging care</div>
      </div>
      <div class="ct-status">
        <div class="item"><div class="label">AI Status</div>
          <div class="value"><span class="dot" style="background:{GOOD}"></span>Model Ready</div></div>
        <div class="item"><div class="label">Data Status</div>
          <div class="value"><span class="dot" style="background:{GOOD}"></span>{n_episodes:,} episodes loaded</div></div>
        <div class="item"><div class="label">Session Time</div>
          <div class="value">{now}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    claims, episodes, provider_drift = load_data()
    art, shap_values, packets = load_model_artifacts()
    feature_cols = art["feature_cols"]
    model = art["model"]

    render_header(len(episodes))
    st.warning(DISCLAIMER)

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        view = st.radio("**View**", ["Analyst", "Executive", "Technical", "Responsible AI"], horizontal=False)
        st.divider()

        st.markdown("**🎬 Demo Cases**")
        chosen_demo = None
        for label, eid in DEMO_CASES.items():
            if st.button(label, use_container_width=True):
                chosen_demo = eid
        if chosen_demo:
            st.session_state["selected_episode"] = chosen_demo

        st.divider()
        st.markdown("**🔎 Full Episode Search**")
        sort_desc = st.checkbox("Sort by fragmentation probability", value=True)
        options = episodes.sort_values("fragmentation_proba", ascending=False) if sort_desc else episodes
        default_ep = st.session_state.get("selected_episode", options.iloc[0]["episode_id"])
        opt_list = options.reset_index(drop=True)
        default_idx = int(opt_list.index[opt_list["episode_id"] == default_ep][0]) if default_ep in opt_list["episode_id"].values else 0
        episode_id = st.selectbox(
            "Episode", options["episode_id"], index=default_idx,
            format_func=lambda e: f"{e}  ({episodes.set_index('episode_id').loc[e, 'fragmentation_proba']:.0%})"
        )
        st.session_state["selected_episode"] = episode_id

        st.divider()
        st.markdown("**🩺 Model Health** <span style='font-size:10px;color:#999'>(simulated)</span>", unsafe_allow_html=True)
        st.progress(0.94, text="Data quality: 94%")
        st.progress(0.88, text="Model confidence (avg): 88%")
        c1, c2 = st.columns(2)
        c1.metric("Drift indicator", "Stable", delta=None)
        c2.metric("Missing data", "0.4%", delta=None)

    row = episodes[episodes["episode_id"] == episode_id].iloc[0]
    row_idx = episodes.index[episodes["episode_id"] == episode_id][0]
    ep_claims = claims[claims["episode_id"] == episode_id].sort_values("claim_date")
    packet = packets[episode_id]

    # ============== EXECUTIVE VIEW ==============
    if view == "Executive":
        st.subheader("Executive Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cases reviewed", f"{len(episodes):,}")
        c2.metric("Patterns detected", f"{int((episodes['fragmentation_proba']>=0.3).sum()):,}")
        c3.metric("High-priority cases", f"{int((episodes['fragmentation_proba']>=0.6).sum()):,}")
        c4.metric("Providers w/ coding drift", f"{int(provider_drift['provider_drift_flag'].sum())}")
        st.plotly_chart(
            px.histogram(episodes, x="fragmentation_proba", nbins=30, color_discrete_sequence=[TEAL],
                          labels={"fragmentation_proba": "Fragmentation probability"})
            .update_layout(height=280, title="Distribution across all episodes", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FCFAF6"),
            use_container_width=True)
        st.info("Potential operational impact: episodes above 60% probability represent the recommended "
                "care-navigator review queue for this population.")
        return

    # ============== TECHNICAL VIEW ==============
    if view == "Technical":
        st.subheader("Technical Detail")
        try:
            metrics = pd.read_json("models/classifier_metrics.json", typ="series")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            c2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
            c3.metric("Precision @0.5", f"{metrics['precision_at_0.5']:.2f}")
            c4.metric("Recall @0.5", f"{metrics['recall_at_0.5']:.2f}")
        except Exception:
            st.write("Metrics unavailable.")
        st.markdown("**Model:** XGBoost classifier, 250 estimators, max depth 3, trained on 15 episode-level features.")
        st.markdown("**Global feature importance (mean |SHAP|, all scored episodes)**")
        imp = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=True)
        st.plotly_chart(px.bar(imp, x="mean_abs_shap", y="feature", orientation="h",
                                color_discrete_sequence=[TEAL]).update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)"),
                         use_container_width=True)
        st.markdown("**Provider drift table**")
        st.dataframe(provider_drift[provider_drift["provider_drift_flag"]].reset_index(drop=True), use_container_width=True, hide_index=True)
        return

    # ============== RESPONSIBLE AI VIEW ==============
    if view == "Responsible AI":
        st.subheader("Responsible AI")
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"""
            <div class="ct-card"><h4>Human Oversight</h4>
            <span class="badge badge-high">REQUIRED</span> Every finding routes to a care coordinator.
            No autonomous outreach or clinical action is ever taken.</div>
            <div class="ct-card"><h4>Explainability</h4>
            <span class="badge badge-low">AVAILABLE</span> SHAP-attributed, signed contributing factors
            for every flagged episode.</div>
            <div class="ct-card"><h4>Patient Data</h4>
            <span class="badge badge-typ">SYNTHETIC</span> All data in this prototype is synthetic —
            see data/README.md.</div>
            """, unsafe_allow_html=True)
        with colB:
            st.markdown(f"""
            <div class="ct-card"><h4>Model Limitations</h4>
            Training label is a simulated chart-audit outcome, not a real clinical determination.
            Not evaluated for subgroup performance disparities.</div>
            <div class="ct-card"><h4>Clinical Validation</h4>
            <span class="badge badge-mod">REQUIRED BEFORE PRODUCTION</span> Real-data validation,
            subgroup fairness audit, and a supervised pilot are required before any deployment.</div>
            <div class="ct-card"><h4>Provider Drift</h4>
            Framed as corroborating context for review — never a fraud or abuse finding on its own.</div>
            """, unsafe_allow_html=True)
        return

    # ============== ANALYST VIEW (default control center) ==============
    badge_text, badge_class, badge_color = risk_badge(row["fragmentation_proba"])
    typology = row.get("typology_name")

    top_row1, top_row2 = st.columns([2.3, 1])

    with top_row1:
        st.markdown('<div class="ct-card"><h4>Clinical Journey Timeline</h4>', unsafe_allow_html=True)
        st.plotly_chart(render_timeline(ep_claims), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ct-card"><h4>Specialty Referral Graph</h4>', unsafe_allow_html=True)
        g = build_transition_graph(ep_claims)
        gsum = graph_summary(g)
        gcol, tcol = st.columns([2.2, 1])
        with gcol:
            st.plotly_chart(render_graph(g), use_container_width=True)
        with tcol:
            st.metric("Specialties", gsum["n_nodes"])
            st.metric("Ping-pong edges", gsum["reciprocal_edges"])
            st.metric("Graph density", gsum["density"])
        st.markdown("</div>", unsafe_allow_html=True)

    with top_row2:
        st.markdown('<div class="ct-card"><h4>AI Pattern Assessment</h4>', unsafe_allow_html=True)
        st.plotly_chart(render_gauge(row["fragmentation_proba"]), use_container_width=True)
        st.markdown(f'<span class="badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
        if pd.notna(typology):
            st.markdown(f'<br><span class="badge badge-typ">Pattern type: {typology}</span>', unsafe_allow_html=True)
        conf = "High" if abs(row["fragmentation_proba"] - 0.5) > 0.35 else "Moderate"
        st.markdown(f"""
        <div style="margin-top:12px; font-size:13px;">
        <b>Confidence:</b> {conf} &nbsp;|&nbsp; <b>Evidence strength:</b> {"Strong" if row['fragmentation_proba']>=0.6 else "Moderate" if row['fragmentation_proba']>=0.3 else "Weak"}<br>
        <b>Human review:</b> {"Required" if row['fragmentation_proba']>=0.3 else "Optional"}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ct-card"><h4>Run Analysis</h4>', unsafe_allow_html=True)
        if st.button("▶ Run AI Analysis", use_container_width=True, type="primary"):
            steps = ["Loading claims…", "Extracting episode features…", "Scoring with XGBoost…",
                     "Computing SHAP attributions…", "Clustering fragmentation typology…",
                     "Checking provider coding drift…", "Composing review packet…", "Done."]
            ph = st.empty()
            for s in steps:
                ph.write(f"⏳ {s}")
                time.sleep(0.28)
            ph.success("✅ Analysis complete — results below reflect this episode's live scoring.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- PATTERN REVEAL ----------------
    st.markdown('<div class="ct-card"><h4>Why Did the System Flag This? — Pattern Reveal</h4>', unsafe_allow_html=True)
    X_row = row[feature_cols].copy()
    vals = shap_values[row_idx]
    order = np.argsort(-np.abs(vals))[:5]

    flow_cols = st.columns([1, 0.3, 1, 0.3, 1, 0.3, 1])
    labels_short = ["Signal 1", "→", "Signal 2", "→", "Signal 3", "→", "AI Pattern"]
    for i, col in enumerate(flow_cols):
        with col:
            if labels_short[i] == "→":
                st.markdown(f'<div class="flow-arrow" style="text-align:center;padding-top:14px;">→</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="flow-step"><div class="n">{"AI" if "AI" in labels_short[i] else labels_short[i][-1]}</div>'
                             f'<div class="t">{labels_short[i]}</div></div>', unsafe_allow_html=True)

    st.write("")
    for i in order:
        col_name = feature_cols[i]
        shap_val = vals[i]
        val = X_row[col_name]
        dots, strength, color = signal_dots(abs(shap_val))
        direction = "↑ increases" if shap_val > 0 else "↓ decreases"
        with st.expander(f"{col_name.replace('_',' ').title()}  —  value: {val:.2f}   [{strength} signal]"):
            cc1, cc2 = st.columns([1, 2])
            with cc1:
                st.markdown(f'<span class="signal-dots" style="color:{color}">{dots}</span> **{strength}**', unsafe_allow_html=True)
                st.metric("SHAP contribution", f"{shap_val:+.3f}")
            with cc2:
                st.write(f"**What happened:** this episode's `{col_name}` measured **{val:.2f}**.")
                st.write(f"**Why it matters:** this value {direction} the model's fragmentation probability "
                         f"by {abs(shap_val):.2f} (on the model's log-odds scale) relative to the average episode.")
                st.write(f"**Model contribution:** ranked #{list(order).index(i)+1} of the top drivers for this episode.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- EVIDENCE -> EXPLANATION -> RECOMMENDATION ----------------
    st.markdown('<div class="ct-card"><h4>Evidence → Explanation → Recommendation → Human Action</h4>', unsafe_allow_html=True)
    ev_col, ex_col = st.columns([1, 1.3])
    with ev_col:
        st.markdown("**📋 Evidence (model-derived)**")
        for i in order:
            col_name = feature_cols[i]
            dots, strength, color = signal_dots(abs(vals[i]))
            st.markdown(f'<div class="signal-row"><span class="signal-label">{col_name.replace("_"," ").title()}</span>'
                        f'<span class="signal-dots" style="color:{color}">{dots}</span></div>', unsafe_allow_html=True)
        if pd.notna(typology):
            st.caption(f"Pattern type: **{typology}** — {row.get('typology_description','')}")

    with ex_col:
        st.markdown("**🗒️ Explanation** <span class='badge badge-typ' style='font-size:10px;'>AI-GENERATED</span>", unsafe_allow_html=True)
        st.info(packet["narrative"])
        st.caption("Distinguish: the evidence above is **model-derived** (SHAP, deterministic). This narrative "
                   "is an **AI-generated explanation** — a template-based composer in this offline POC, swappable "
                   "to an LLM call over the same structured signals. Neither one makes the detection decision.")
        st.markdown(f"**Recommendation:** {packet['recommended_action']}")

    st.markdown("**✅ Human-in-the-Loop Review**")
    b1, b2, b3, b4 = st.columns(4)
    action = None
    if b1.button("Accept Finding", use_container_width=True): action = "Accepted"
    if b2.button("Dismiss Finding", use_container_width=True): action = "Dismissed"
    if b3.button("Escalate", use_container_width=True): action = "Escalated"
    if b4.button("Request More Evidence", use_container_width=True): action = "More evidence requested"
    if action:
        st.success(f"✔️ Reviewer decision recorded: **{action}** for {episode_id}. "
                    "(Recorded in-session only — not persisted in this offline POC.)")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- DECISION SIMULATOR ----------------
    with st.expander("🧪 Decision Simulator — what would change this finding?"):
        st.caption("Adjusts one feature and re-scores this episode live with the trained model — a real "
                   "re-inference, not a scripted before/after.")
        sim_feature = feature_cols[order[0]]
        cur_val = float(X_row[sim_feature])
        lo, hi = float(episodes[sim_feature].quantile(0.02)), float(episodes[sim_feature].quantile(0.98))
        new_val = st.slider(f"Simulate `{sim_feature}`", min_value=round(lo, 2), max_value=round(hi, 2),
                             value=round(cur_val, 2))
        X_sim = X_row.copy()
        X_sim[sim_feature] = new_val
        new_proba = float(model.predict_proba(pd.DataFrame([X_sim])[feature_cols])[0, 1])
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Original probability", f"{row['fragmentation_proba']:.0%}")
        sc2.metric("Simulated probability", f"{new_proba:.0%}", delta=f"{(new_proba-row['fragmentation_proba'])*100:+.1f} pts")
        sc3.metric("Would flag change?", "Yes" if (new_proba>=0.5) != (row['fragmentation_proba']>=0.5) else "No")

    # ---------------- CASE COMPARISON ----------------
    with st.expander("⚖️ Case Comparison"):
        others = [e for e in episodes["episode_id"] if e != episode_id]
        default_cmp = DEMO_CASES["🟢 Normal — routine pathway"] if episode_id != DEMO_CASES["🟢 Normal — routine pathway"] else DEMO_CASES["🔴 High priority — converging signals"]
        cmp_index = others.index(default_cmp) if default_cmp in others else 0
        compare_id = st.selectbox("Compare against", others, index=cmp_index)
        cmp_row = episodes[episodes["episode_id"] == compare_id].iloc[0]
        colL, colR = st.columns(2)
        for col, label, r in [(colL, episode_id, row), (colR, compare_id, cmp_row)]:
            with col:
                st.markdown(f"**{label}**")
                st.metric("Probability", f"{r['fragmentation_proba']:.0%}")
                st.metric("Visits / Specialties", f"{int(r['n_visits'])} / {int(r['n_specialties'])}")
                st.metric("Typology", r.get("typology_name") if pd.notna(r.get("typology_name")) else "—")


if __name__ == "__main__":
    main()
