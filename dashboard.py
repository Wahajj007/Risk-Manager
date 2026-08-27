import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics import average_precision_score, precision_score, recall_score

st.set_page_config(page_title="Fraud Radar — Risk Intelligence", layout="wide", initial_sidebar_state="expanded")

# ── Theme ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background: #0B1120;
    color: #E4E9F2;
}

section[data-testid="stSidebar"] {
    background: #0E1626;
    border-right: 1px solid #1E2A42;
}

/* Headers */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    color: #F5F7FA !important;
}
h2 {
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    color: #F5F7FA !important;
    font-size: 1.35rem !important;
    margin-top: 0.5rem !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: #8B95A7 !important;
}

/* Card container for sections */
.fr-card {
    background: #131B2E;
    border: 1px solid #1E2A42;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}

.fr-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
    font-weight: 700;
    color: #3395FF;
    margin-bottom: 0.35rem;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #131B2E;
    border: 1px solid #1E2A42;
    border-radius: 12px;
    padding: 1rem 1.1rem;
}
[data-testid="stMetricLabel"] {
    color: #8B95A7 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: #F5F7FA !important;
    font-variant-numeric: tabular-nums;
    font-weight: 700 !important;
}

/* Buttons */
.stButton button {
    background: #3395FF !important;
    color: #08101F !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
    transition: transform 0.1s ease, box-shadow 0.15s ease;
}
.stButton button:hover {
    box-shadow: 0 0 0 3px rgba(51, 149, 255, 0.25);
    transform: translateY(-1px);
}
.stDownloadButton button {
    background: transparent !important;
    color: #3395FF !important;
    border: 1px solid #2A3B5C !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Divider */
hr { border-color: #1E2A42 !important; margin: 2rem 0 !important; }

/* Top accent line */
.fr-topline {
    height: 3px;
    width: 48px;
    background: #3395FF;
    border-radius: 2px;
    margin-bottom: 1.1rem;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1E2A42 !important; border-radius: 10px; overflow: hidden; }

/* Bordered containers (report card) */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: #131B2E !important;
    border-color: #1E2A42 !important;
    border-radius: 14px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] h3 { color: #F5F7FA !important; }
[data-testid="stVerticalBlockBorderWrapper"] strong { color: #C4CCDA; }

/* Slider */
[data-testid="stSlider"] label { color: #C4CCDA !important; font-weight: 500 !important; }

/* Selectbox */
[data-testid="stSelectbox"] label { color: #C4CCDA !important; font-weight: 500 !important; }

/* Table (markdown) */
table { border-color: #1E2A42 !important; }
thead tr th { background: #131B2E !important; color: #8B95A7 !important; }
tbody tr td { color: #C4CCDA !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="fr-topline"></div>'
    '<div class="fr-eyebrow">Risk Intelligence</div>'
    '<h1 style="margin-bottom:0.3rem;">Fraud Radar</h1>'
    '<p style="color:#8B95A7; font-size:0.98rem; margin-top:0; max-width:640px;">'
    'Robust anomaly detection, calibrated risk scoring, and unsupervised fraud-ring '
    'discovery for real-time transaction defense.</p>',
    unsafe_allow_html=True
)

test_df = pd.read_csv('data/final_test_scored.csv').sort_values('Time').reset_index(drop=True)

# ── Sidebar controls ──────────────────────────────────────────────
st.sidebar.markdown('<div class="fr-eyebrow">Controls</div>', unsafe_allow_html=True)
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.75, 0.05)
st.sidebar.caption("Default (0.75) selected via cost-based analysis")
speed = st.sidebar.slider("Playback speed (tx/sec)", 1, 20, 5)
n_to_show = st.sidebar.slider("Feed window size", 10, 100, 30)

def get_action(prob, threshold):
    if prob >= threshold:
        return "Auto-block"
    elif prob >= threshold * 0.5:
        return "Review"
    else:
        return "Approve"

test_df['Action'] = test_df['FraudProbability'].apply(lambda p: get_action(p, threshold))

# ── Live feed ──────────────────────────────────────────────────────
st.markdown('<div class="fr-eyebrow">Live feed</div><h2>Transaction stream</h2>', unsafe_allow_html=True)

if 'position' not in st.session_state:
    st.session_state.position = 0
if 'running' not in st.session_state:
    st.session_state.running = False

col_a, col_b, col_c = st.columns([1, 1, 5])
if col_a.button("Start"):
    st.session_state.running = True
if col_b.button("Pause"):
    st.session_state.running = False

metrics_placeholder = st.empty()
table_placeholder = st.empty()

def render(position):
    window = test_df.iloc[max(0, position - n_to_show):position].copy()
    seen_so_far = test_df.iloc[:position]

    with metrics_placeholder.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Processed", f"{position:,}")
        c2.metric("Flagged", f"{(seen_so_far['FraudProbability'] >= threshold * 0.5).sum() if position > 0 else 0:,}")
        c3.metric("Auto-blocked", f"{(seen_so_far['FraudProbability'] >= threshold).sum() if position > 0 else 0:,}")
        c4.metric("Confirmed fraud seen", f"{int(seen_so_far['Class'].sum()) if position > 0 else 0:,}")

    with table_placeholder.container():
        display = window[['Time', 'Amount', 'RobustMahalanobisDist', 'FraudProbability', 'Action']].iloc[::-1].copy()
        display = display.rename(columns={
            'RobustMahalanobisDist': 'Anomaly distance',
            'FraudProbability': 'Risk score',
        })
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.NumberColumn(format="%d s"),
                "Amount": st.column_config.NumberColumn(format="$%.2f"),
                "Anomaly distance": st.column_config.NumberColumn(format="%.2f"),
                "Risk score": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0.0, max_value=1.0
                ),
            },
        )

if st.session_state.running:
    while st.session_state.running and st.session_state.position < len(test_df):
        st.session_state.position += 1
        render(st.session_state.position)
        time.sleep(1.0 / speed)
else:
    render(st.session_state.position)

# ── Model fit diagnostics ────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<div class="fr-eyebrow">Validation</div><h2>Model diagnostics</h2>', unsafe_allow_html=True)
st.caption("Measured on a held-out test set with zero overlap with training data.")

y_true_test = test_df['Class']
y_prob_test = test_df['FraudProbability']
y_pred_test = (y_prob_test >= threshold).astype(int)
auprc_test = average_precision_score(y_true_test, y_prob_test)
baseline_auprc = y_true_test.mean()

d1, d2, d3, d4 = st.columns(4)
d1.metric("AUPRC", f"{auprc_test:.3f}", f"{auprc_test/baseline_auprc:.0f}x baseline")
d2.metric("Precision", f"{precision_score(y_true_test, y_pred_test, zero_division=0):.1%}")
d3.metric("Recall", f"{recall_score(y_true_test, y_pred_test, zero_division=0):.1%}")
d4.metric("Baseline AUPRC", f"{baseline_auprc:.4f}")

st.caption(
    "AUPRC is the metric recommended for this dataset's extreme class imbalance. "
    "The model was trained on a disjoint set with zero transaction overlap — this is a "
    "genuine out-of-sample evaluation."
)

# ── Fraud ring graph ────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<div class="fr-eyebrow">Graph intelligence</div><h2>Fraud ring detection</h2>', unsafe_allow_html=True)
st.caption(
    "Transactions clustered by feature similarity among the top 5% most anomalous. "
    "Hover any node for details. Red marks confirmed fraud — clusters were found without "
    "using fraud labels in construction."
)

with open('fraud_ring_graph_interactive.html', 'r', encoding='utf-8') as f:
    graph_html = f.read()
st.components.v1.html(graph_html, height=760, scrolling=True)

st.markdown("**Sensitivity — structure holds across parameter choices**")
st.markdown("""
| Anomaly filter | Edge threshold | High-purity communities (\u226590%) | Fraud transactions covered |
|---|---|---|---|
| Top 3% | Tight | 14 | 147 |
| Top 5% (default) | Tight | 12 | 157 |
| Top 7% | Tight | 10 | 136 |
| Top 5% | Looser | 11 | 204 |
""")

# ── Auto-responder ────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<div class="fr-eyebrow">Automated response</div><h2>Evidence report generator</h2>', unsafe_allow_html=True)
st.caption("Generate a reviewer-ready risk report for any flagged transaction.")

def generate_evidence_report(row, threshold):
    risk_level = "HIGH RISK — AUTO-BLOCKED" if row['FraudProbability'] >= threshold else "MEDIUM RISK — FLAGGED FOR REVIEW"
    report = f"""
### Transaction Risk Report

**Status:** {risk_level}

**Transaction Details**
- Time elapsed since window start: {row['Time']/3600:.1f} hours ({row['Time']/3600/24:.1f} days in)
- Approximate hour of day: {int((row['Time'] // 3600) % 24)}:00
- Amount: ${row['Amount']:.2f}
- Cohort: {row.get('Cohort', 'N/A')}

**Risk Assessment**
- Predicted fraud probability: {row['FraudProbability']:.1%}
- Calibrated probability estimate: {row.get('CalibratedFraudProbability', 0):.2%}
- Decision threshold in effect: {threshold:.0%}

**Why this transaction was flagged**

This transaction's robust Mahalanobis distance from its cohort's normal behavioral
baseline is **{row['RobustMahalanobisDist']:.2f}** — well outside the typical range for
legitimate transactions. This means its behavioral pattern deviates significantly from
what this cohort's normal activity looks like, even after accounting for outlier
contamination using robust covariance estimation (Minimum Covariance Determinant).

**Recommended action**

{"Immediate block, request step-up verification (OTP / 2FA) before allowing the transaction to proceed." if row['FraudProbability'] >= threshold else "Route to manual review queue. Do not auto-block — confidence is not high enough to justify customer friction."}

---
*Generated automatically. Risk scores reflect a model validated with AUPRC={auprc_test:.4f}
(vs. {baseline_auprc:.4f} random baseline) on held-out data.*
"""
    return report

flagged = test_df[test_df['FraudProbability'] >= threshold * 0.5].sort_values('FraudProbability', ascending=False)

if len(flagged) > 0:
    selected_idx = st.selectbox(
        "Select a flagged transaction",
        options=flagged.index,
        format_func=lambda i: f"Time {test_df.loc[i, 'Time']:.0f}s · ${test_df.loc[i, 'Amount']:.2f} · Risk {test_df.loc[i, 'FraudProbability']:.1%}"
    )
    if st.button("Generate report"):
        report_text = generate_evidence_report(test_df.loc[selected_idx], threshold)
        with st.container(border=True):
            st.markdown(report_text)
        st.download_button("Download report", data=report_text,
                            file_name=f"fraud_report_{selected_idx}.md", mime="text/markdown")
else:
    st.info("No flagged transactions at the current threshold.")