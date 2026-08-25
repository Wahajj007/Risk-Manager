# Bayesian Fraud Radar

An explainable transaction fraud detector built for Razorpay's hackathon — Track 2: AI Risk Manager.

Instead of a black-box classifier, this project combines **robust statistics**, a **trained, calibrated ML model**, and **unsupervised graph-based fraud-ring detection** to score transaction risk transparently, with honest, cost-aware metrics.

## What it does

1. **Robust anomaly scoring** — measures how far a transaction deviates from its cohort's normal behavior using Mahalanobis distance with Minimum Covariance Determinant (MCD) estimation, which resists the "masking effect" that naive statistics suffer from when the training data itself contains outliers or fraud.
2. **Trained classifier** — a logistic regression model uses the robust anomaly score (plus transaction amount) to output a calibrated fraud probability.
3. **Unsupervised fraud-ring detection** — a similarity graph over the most anomalous transactions, with Louvain community detection, finds tightly-connected clusters that are up to 100% fraud — without ever using fraud labels to build the graph.
4. **Live API + dashboard** — a FastAPI service scores new transactions in ~13ms, and a Streamlit dashboard simulates a live transaction feed, shows model diagnostics, the fraud-ring graph, and an auto-responder that generates human-readable evidence reports for flagged transactions.

## Key results

- **Robust vs. naive Mahalanobis distance**: on a controlled contamination test, an injected outlier scored 4.996 under naive Mahalanobis distance (nearly invisible) vs. 127.320 under robust MCD (correctly flagged) — a ~25x difference on the identical point.
- **AUPRC: 0.1882** vs. a random-guessing baseline of 0.0060 (~31x improvement) — AUPRC is the metric this dataset's own authors recommend for its extreme class imbalance.
- **Cost-based threshold**: swept decision thresholds against illustrative false-positive/false-negative costs, found operating at threshold 0.75 reduces total business cost by ~15% vs. the naive 0.5 default, while still catching 75% of fraud.
- **Fraud-ring discovery**: unsupervised community detection found clusters with up to 100% fraud purity (e.g. 28/28 and 21/21 transactions), validated as robust across multiple parameter settings, not a fluke of one configuration.
- **Live latency**: ~12.7ms average per scoring request via the FastAPI service.

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284,807 European transactions from September 2013, 492 labeled fraud (0.17%). Download `creditcard.csv` and place it in `data/` before running (not included in this repo — 147MB, over GitHub's file size limit).

The dataset's own documentation confirms user/entity identifiers were deliberately stripped for confidentiality, so this project uses cohort-based baselines (amount tier × time-of-day) as a defensible stand-in.

## How to run

1. Download `creditcard.csv` from Kaggle into `data/`.
2. Run the pipeline in order to regenerate all intermediate data:

python explore.py # loads data, builds cohorts, train/test split
python mod2.py # naive + robust Mahalanobis distance
python mod3.py # trains the logistic regression model, saves fraud_model.pkl
python mod4.py # precision/recall, calibration, cost-based threshold analysis
python graph.py # fraud-ring graph construction, community detection, visualizations

3. Launch the dashboard:

streamlit run dashboard.py

4. Launch the API (separate terminal):

uvicorn api:app --reload

   Test with a POST request to `http://127.0.0.1:8000/score`.

`fraud_model.pkl` and `robust_baselines.pkl` are included in this repo, so the API can run immediately without regenerating the full pipeline.

## Track alignment

Built for Track 2 (AI Risk Manager) — a **detector** with measured precision/recall on a genuinely held-out test set, honest false-positive-cost framing, and a strictly defensive design (no offense-capable functionality). The auto-responder component (evidence report generation) also touches the "verifier/auto-responder" direction named in the track brief.

## Known limitations

- Uses a single well-known public dataset; no real user/device/IP identifiers were available (confirmed by the dataset's own documentation), so cohort-level and graph-based approaches substitute for true per-user modeling.
- The classifier intentionally uses a small, interpretable feature set (2 features) rather than the full 28 PCA components — trades some raw predictive power for explainability. Public leaderboards on this dataset with heavier models reach higher AUPRC (0.7-0.85+); this project prioritizes a transparent, defensible reasoning chain over maximizing a single metric.
- Velocity/device signals in the evidence reports are illustrative proxies, clearly documented as such, not real device/IP data.