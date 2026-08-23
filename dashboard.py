import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics import average_precision_score, precision_score, recall_score

st.set_page_config(page_title="Bayesian Fraud Radar", layout="wide")

st.title("🛡️ Bayesian Fraud Radar")
st.caption("Robust anomaly detection + calibrated risk scoring + fraud-ring discovery")

test_df = pd.read_csv('data/final_test_scored.csv').sort_values('Time').reset_index(drop=True)

# ── Sidebar controls ──────────────────────────────────────────────
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.75, 0.05)
st.sidebar.caption("Default (0.75) chosen via cost-based analysis in Day 4 validation")
speed = st.sidebar.slider("Playback speed (transactions/sec)", 1, 20, 5)
n_to_show = st.sidebar.slider("Feed window size", 10, 100, 30)

def get_action(prob, threshold):
    if prob >= threshold:
        return "🔴 Auto-block"
    elif prob >= threshold * 0.5:
        return "🟡 Review"
    else:
        return "🟢 Approve"

test_df['Action'] = test_df['FraudProbability'].apply(lambda p: get_action(p, threshold))

# ── Live feed ──────────────────────────────────────────────────────
st.header("1. Live Transaction Feed")

if 'position' not in st.session_state:
    st.session_state.position = 0
if 'running' not in st.session_state:
    st.session_state.running = False

col_a, col_b, col_c = st.columns([1, 1, 4])
if col_a.button("▶️ Start"):
    st.session_state.running = True
if col_b.button("⏸️ Stop"):
    st.session_state.running = False

metrics_placeholder = st.empty()
table_placeholder = st.empty()

def render(position):
    window = test_df.iloc[max(0, position - n_to_show):position]
    seen_so_far = test_df.iloc[:position]

    with metrics_placeholder.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions processed", position)
        c2.metric("Flagged (review/block)", (seen_so_far['FraudProbability'] >= threshold * 0.5).sum() if position > 0 else 0)
        c3.metric("Auto-blocked", (seen_so_far['FraudProbability'] >= threshold).sum() if position > 0 else 0)
        c4.metric("Real fraud seen", int(seen_so_far['Class'].sum()) if position > 0 else 0)

    with table_placeholder.container():
        st.dataframe(
            window[['Time', 'Amount', 'RobustMahalanobisDist', 'FraudProbability', 'Action', 'Class']]
            .iloc[::-1],
            use_container_width=True
        )

if st.session_state.running:
    while st.session_state.running and st.session_state.position < len(test_df):
        st.session_state.position += 1
        render(st.session_state.position)
        time.sleep(1.0 / speed)
else:
    render(st.session_state.position)

# ── Model fit diagnostics ────────────────────────────────────────
st.divider()
st.header("2. Model Fit Diagnostics")

col1, col2 = st.columns(2)

y_true_test = test_df['Class']
y_prob_test = test_df['FraudProbability']
y_pred_test = (y_prob_test >= threshold).astype(int)
auprc_test = average_precision_score(y_true_test, y_prob_test)
baseline_auprc = y_true_test.mean()

with col1:
    st.markdown("**Test set (held-out, never seen during training)**")
    st.metric("AUPRC (test)", f"{auprc_test:.4f}")
    st.metric("Precision (test)", f"{precision_score(y_true_test, y_pred_test, zero_division=0):.3f}")
    st.metric("Recall (test)", f"{recall_score(y_true_test, y_pred_test, zero_division=0):.3f}")

with col2:
    st.markdown("**Reference: random-guessing baseline**")
    st.metric("AUPRC (baseline)", f"{baseline_auprc:.4f}")
    st.caption(f"Test AUPRC is ~{auprc_test/baseline_auprc:.0f}x better than random guessing")
    st.caption("Model was trained on a separate set (227,452 normal + 147 fraud examples) "
               "with zero overlap with these test transactions — this is a genuine "
               "out-of-sample evaluation, not a fit-quality illusion.")

# ── Fraud ring graph ───────────────────────────────────────────────
st.divider()
st.header("3. Fraud Ring Detection — Unsupervised Community Structure")
st.caption("Transactions clustered by feature similarity (top 5% most anomalous only). "
           "Red = confirmed fraud, Blue = normal. Tight red clusters were found by the "
           "algorithm with zero fraud labels used in construction.")

st.image('fraud_ring_graph.png', use_container_width=True)

st.markdown("**Top fraud-pure communities found (validated using held-out labels):**")
st.markdown("""
- Community with 28/28 transactions — 100% fraud
- Community with 21/21 transactions — 100% fraud
- Community with 55 transactions — 92.7% fraud (51/55)
""")

# ── Auto-responder ────────────────────────────────────────────────
st.divider()
st.header("4. Auto-Responder: Generate Evidence Report")

def generate_evidence_report(row, threshold):
    """Generate a chargeback/fraud evidence summary for a flagged transaction."""
    risk_level = "HIGH RISK - AUTO-BLOCKED" if row['FraudProbability'] >= threshold else "MEDIUM RISK - FLAGGED FOR REVIEW"

    report = f"""
### Transaction Risk Report

**Status:** {risk_level}

**Transaction Details**
- Time elapsed since dataset start: {row['Time']/3600:.1f} hours ({row['Time']/3600/24:.1f} days in)
- Approximate hour of day: {int((row['Time'] // 3600) % 24)}:00
- Amount: ${row['Amount']:.2f}
- Cohort: {row.get('Cohort', 'N/A')}

**Risk Assessment**
- Predicted fraud probability: {row['FraudProbability']:.1%}
- Calibrated probability estimate: {row.get('CalibratedFraudProbability', 0):.2%}
- Decision threshold in effect: {threshold:.0%}

**Why this transaction was flagged**

This transaction's Robust Mahalanobis distance from its cohort's normal behavioral
baseline is **{row['RobustMahalanobisDist']:.2f}** — compared to a typical normal
transaction's distance of ~2-5. This means the transaction's amount and behavioral
pattern deviate significantly from what this cohort's legitimate transactions usually
look like, even after accounting for outlier contamination using robust covariance
estimation (Minimum Covariance Determinant).

**Recommended action**

{"Immediate block, request step-up verification (OTP/2FA) before allowing the transaction to proceed." if row['FraudProbability'] >= threshold else "Route to manual review queue. Do not auto-block; low enough confidence that customer friction should be avoided."}

---
*This report was generated automatically. Risk scores reflect a model validated with
AUPRC={auprc_test:.4f} (vs. {baseline_auprc:.4f} random baseline) on held-out data.*
"""
    return report

flagged = test_df[test_df['FraudProbability'] >= threshold * 0.5].sort_values('FraudProbability', ascending=False)

if len(flagged) > 0:
    selected_idx = st.selectbox(
        "Select a flagged transaction to generate a report:",
        options=flagged.index,
        format_func=lambda i: f"Transaction at Time={test_df.loc[i, 'Time']:.0f}, "
                               f"Amount=${test_df.loc[i, 'Amount']:.2f}, "
                               f"Risk={test_df.loc[i, 'FraudProbability']:.1%}"
    )

    if st.button("Generate Report"):
        report_text = generate_evidence_report(test_df.loc[selected_idx], threshold)
        st.markdown(report_text)
        st.download_button(
            "Download Report",
            data=report_text,
            file_name=f"fraud_report_{selected_idx}.md",
            mime="text/markdown"
        )
else:
    st.info("No flagged transactions at the current threshold.")
    
    st.divider()
st.header("3. Fraud Ring Detection — Interactive Community Structure")
st.caption("Transactions clustered by feature similarity (top 5% most anomalous only). "
           "Hover over nodes for details. Red = confirmed fraud, Blue = normal. "
           "Tight red clusters were found by the algorithm with zero fraud labels used in construction.")

with open('fraud_ring_graph_interactive.html', 'r', encoding='utf-8') as f:
    graph_html = f.read()

st.components.v1.html(graph_html, height=820, scrolling=True)

st.markdown("**Sensitivity check — this structure holds across parameter choices:**")
st.markdown("""
| Anomaly filter | Edge threshold | High-purity communities (≥90%) | Fraud transactions covered |
|---|---|---|---|
| Top 3% | Tight | 14 | 147 |
| Top 5% (default) | Tight | 12 | 157 |
| Top 7% | Tight | 10 | 136 |
| Top 5% | Looser | 11 | 204 |
""")