# Fraud Radar

An explainable transaction fraud detector built for Razorpay's Buildathon — Track 2: AI Risk Manager.

Instead of a black-box classifier, this project combines **robust statistics**, a **trained, calibrated ML model**, and **unsupervised graph-based fraud-ring detection** to score transaction risk transparently — with metrics validated rigorously enough to catch and fix a real data leakage bug along the way.

## What it does

1. **Robust anomaly scoring** — measures how far a transaction deviates from its cohort's normal behavior using Mahalanobis distance with Minimum Covariance Determinant (MCD) estimation, which resists the "masking effect" that naive statistics suffer from when the training data itself contains outliers or fraud.
2. **Trained, feature-validated classifier** — a logistic regression model combines the robust anomaly score with 8 statistically-validated features (selected via point-biserial correlation testing, not guesswork) to output a calibrated fraud probability.
3. **Unsupervised fraud-ring detection** — a similarity graph over the most anomalous transactions, with Louvain community detection, finds tightly-connected clusters that are up to **100% fraud** — without ever using fraud labels to build the graph. Verified robust across multiple parameter settings, not a one-off result.
4. **Live API + dashboard** — a FastAPI service scores new transactions in ~13ms, and a Streamlit dashboard runs a live transaction feed, model diagnostics, the interactive fraud-ring graph, and an auto-responder that generates evidence reports for flagged transactions.

## Key results

- **AUPRC: 0.79** vs. a random-guessing baseline of 0.006 (~130x improvement) — AUPRC is the metric this dataset's own authors recommend for its extreme class imbalance, and 0.79 sits in range with heavily-tuned public leaderboard entries on this dataset.
- **Robust vs. naive Mahalanobis distance**: in a controlled contamination test, an injected outlier scored 4.996 under naive Mahalanobis distance (nearly invisible) vs. 127.320 under robust MCD (correctly flagged) — a ~25x difference on the identical point.
- **We found and fixed our own data leakage bug.** After expanding features, AUPRC jumped suspiciously high — instead of taking the win, we investigated. We traced it to ~9,000 duplicate transactions in the raw dataset that a naive deduplication step missed, fixed the split, and re-validated the entire pipeline. The 0.79 above is measured on a provably leakage-free test set.
- **Fraud-ring discovery**: unsupervised community detection found clusters with up to 100% fraud purity (28/28 and 21/21 transactions), holding up across four different parameter configurations.
- **Cost-based threshold**: swept decision thresholds against illustrative false-positive/false-negative costs; operating at 0.75 cuts total business cost by ~15% vs. a naive 0.5 default, while still catching over 75% of fraud.
- **Live latency**: ~12.7ms average per scoring request via the FastAPI service.

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) — European card transactions from September 2013, deduplicated to ~275,000 transactions with 473 confirmed fraud (0.17%). Download `creditcard.csv` and place it in `data/` before running (not included in this repo — 147MB, over GitHub's file size limit).

The dataset's own documentation confirms user/entity identifiers were deliberately stripped for confidentiality, so this project uses cohort-based baselines (amount tier × time-of-day) as a defensible stand-in.

## How to run

1. Download `creditcard.csv` from Kaggle into `data/`.
2. Run the pipeline in order:

python 01_data_prep.py # load, deduplicate, build cohorts, split
python 02_anomaly_detection.py # naive + robust Mahalanobis distance
python 03_train_model_v1.py # original 2-feature model (superseded, kept for history)
python 04_validation.py # precision/recall, calibration, cost threshold
python 05_model_final.py # feature validation, leakage fix, final model + AUPRC
python 06_fraud_rings.py # graph construction, community detection, visualization

3. Launch the dashboard: `streamlit run dashboard.py`
4. Launch the API (separate terminal): `uvicorn api:app --reload`, then POST to `http://127.0.0.1:8000/score`

`fraud_model.pkl` and `robust_baselines.pkl` are included, so the API runs immediately without regenerating the full pipeline.

## Track alignment

Built for Track 2 (AI Risk Manager) — a **detector** with measured precision/recall on a genuinely held-out, leakage-verified test set, honest false-positive-cost framing, and a strictly defensive design. The auto-responder (evidence report generation) also touches the "verifier/auto-responder" direction named in the track brief.

## Known limitations

- Uses a single well-known public dataset; no real user/device/IP identifiers were available (confirmed by the dataset's own documentation), so cohort-level and graph-based approaches substitute for true per-user modeling.
- Velocity/device signals in the evidence reports are illustrative proxies, clearly documented as such, not real device/IP data.
- The graph module's fraud-ring finding is descriptive (transactions cluster together) rather than causal (we don't know *why*, since the underlying features are anonymized via PCA).