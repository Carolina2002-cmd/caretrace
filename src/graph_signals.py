"""
Builds a per-patient specialty-transition graph with NetworkX. Used both to
compute the ping-pong signal in features.py (numerically) and to render a
visual referral graph in the Streamlit dashboard (the "wow moment" visual).
"""

import networkx as nx


def build_transition_graph(patient_claims_df):
    """patient_claims_df: rows for ONE patient, sorted by claim_date."""
    g = nx.DiGraph()
    specialties = patient_claims_df["provider_specialty"].tolist()
    dates = patient_claims_df["claim_date"].tolist()

    for s in set(specialties):
        g.add_node(s)

    for i in range(len(specialties) - 1):
        a, b = specialties[i], specialties[i + 1]
        if g.has_edge(a, b):
            g[a][b]["weight"] += 1
        else:
            g.add_edge(a, b, weight=1)

    return g


def graph_summary(g: nx.DiGraph) -> dict:
    if g.number_of_nodes() == 0:
        return {"n_nodes": 0, "n_edges": 0, "reciprocal_edges": 0, "density": 0.0}
    reciprocal = sum(1 for a, b in g.edges() if g.has_edge(b, a)) // 2
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "reciprocal_edges": reciprocal,
        "density": round(nx.density(g), 3),
    }
