# Fraud Radar

An end-to-end data science and machine learning pipeline for transaction fraud detection

The project focuses on turning raw transaction data into statistically validated, interpretable fraud-risk signals. It combines **exploratory data analysis, robust statistical methods, feature validation, supervised machine learning, unsupervised learning, model evaluation, and deployment** into a complete analytical workflow.

Rather than relying on a single black-box classifier, Fraud Radar combines multiple analytical approaches to understand abnormal transaction behavior, estimate fraud probability, identify potential fraud clusters, and evaluate the business trade-offs involved in deploying a fraud detection model.

## Data Science Approach

The pipeline follows a structured data science workflow:

**Raw Transactions → Data Cleaning → Exploratory Analysis → Feature Engineering → Statistical Validation → Anomaly Detection → Supervised ML → Unsupervised Learning → Model Evaluation → Deployment**

### 1. Data Preparation & Exploratory Analysis

The raw transaction dataset is cleaned and prepared before modeling. This includes:

* Deduplication of transaction records.
* Construction of behavioral cohorts using transaction amount tiers and time-of-day.
* Train/test splitting designed to prevent duplicate observations from appearing across evaluation boundaries.
* Investigation of class imbalance and its implications for model evaluation.
* Validation of the resulting datasets before model training.

The dataset contains approximately **275,000 transactions and 473 confirmed fraudulent transactions**, making fraud detection a highly imbalanced classification problem.

2. Robust Statistical Anomaly Detection

Instead of treating every observation equally, the project first establishes statistically robust measures of how unusual a transaction is relative to its cohort.

Mahalanobis distance is used to measure multivariate deviation, while **Minimum Covariance Determinant (MCD)** estimation provides a robust covariance estimate that is less affected by extreme observations.

This allows the pipeline to distinguish between:

* Normal observations
* Statistically unusual transactions
* Transactions that are extreme enough to warrant further investigation

A controlled contamination experiment compared conventional Mahalanobis estimation with robust MCD estimation. For the same injected outlier, the naive approach produced a score of **4.996**, while the robust approach produced **127.320**, demonstrating the effect of contaminated covariance estimates on anomaly detection.

### 3. Statistical Feature Validation

Features were not selected purely through model experimentation.

The final model uses **8 statistically validated features**, with point-biserial correlation testing used to examine relationships between candidate variables and the binary fraud target.

This provides a statistical basis for feature selection before introducing the variables into the predictive model.

The workflow therefore separates:

**Feature discovery → Statistical validation → Model training → Model evaluation**

rather than relying exclusively on automated feature importance.

### 4. Supervised Machine Learning

A **logistic regression classifier** combines the robust anomaly signal with the validated transaction-level features to estimate the probability that a transaction is fraudulent.

The model was evaluated using metrics appropriate for severe class imbalance, with particular emphasis on **Area Under the Precision-Recall Curve (AUPRC)**.

Final performance:

* **AUPRC: 0.79**
* Random baseline: **0.006**
* Approximately **130× improvement over the baseline**

The model also produces calibrated fraud probabilities that can be converted into operational decisions using configurable thresholds.

### 5. Unsupervised Fraud-Cluster Analysis

The project also explores fraud behavior without using fraud labels during graph construction.

A similarity graph is constructed from highly anomalous transactions, after which **Louvain community detection** is applied to identify densely connected transaction groups.

This provides a complementary analytical perspective:

**Supervised learning:**
*"How likely is this transaction to be fraudulent?"*

**Unsupervised learning:**
*"Which transactions exhibit structural similarity or cluster together?"*

The analysis identified communities with fraud purity reaching **100% in some discovered clusters**, including groups of 28/28 and 21/21 transactions. These results remained consistent across four different parameter configurations.

The graph analysis is treated as descriptive rather than causal because the underlying transaction features are anonymized through PCA.

## Model Validation & Data Quality

A major part of the project involved validating whether the model's performance was actually trustworthy.

During development, an unexpectedly large increase in AUPRC was investigated rather than accepted at face value.

The investigation identified approximately **9,000 duplicate transactions** in the raw dataset that had not been adequately handled by the initial preprocessing workflow.

The data preparation and validation pipeline was subsequently rebuilt to ensure that duplicate observations could not create leakage between training and testing data.

The final **0.79 AUPRC** was measured only after this leakage issue had been addressed and the evaluation pipeline revalidated.

This became an important part of the project because it demonstrated that model performance was being treated as an analytical result to validate, rather than simply a number to optimize.

## Business-Oriented Model Evaluation

Model evaluation extends beyond predictive performance.

Decision thresholds were evaluated using illustrative false-positive and false-negative costs to understand how the classifier could behave under different operational trade-offs.

A threshold of **0.75** reduced the modeled total business cost by approximately **15% compared with a 0.5 threshold**, while still detecting more than 75% of fraudulent transactions.

This provides a practical connection between:

**Model probability → Decision threshold → Classification outcome → Business cost**

rather than treating accuracy alone as the objective.

## Deployment & Analytics Interface

The final analytical pipeline was converted into a deployable application.

### FastAPI

A FastAPI service exposes the trained model through a scoring endpoint.

* Average scoring latency: **~12.7 ms**
* Returns transaction-level fraud probability
* Converts model output into an actionable risk decision

### Streamlit Dashboard

The dashboard provides:

* Live transaction scoring
* Model diagnostics
* Fraud-risk analysis
* Interactive fraud-ring visualization
* Evidence reports for flagged transactions
* Analytical views of model behavior

This connects the data science workflow to a usable analytical interface.

## Key Results

| Analysis                     |       Result |
| ---------------------------- | -----------: |
| Transactions analyzed        |        ~275K |
| Confirmed fraud              |          473 |
| Fraud prevalence             |       ~0.17% |
| Final AUPRC                  |     **0.79** |
| Random baseline              |    **0.006** |
| Improvement over baseline    |    **~130×** |
| Robust MCD outlier score     |  **127.320** |
| Naive Mahalanobis score      |    **4.996** |
| Maximum fraud-cluster purity |     **100%** |
| Average API latency          | **~12.7 ms** |

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

The dataset contains European card transactions from September 2013. The raw dataset is deduplicated during preprocessing and contains approximately 275,000 transactions with 473 confirmed fraudulent transactions.

The dataset documentation indicates that identifying information such as user/entity identifiers was removed for confidentiality. As a result, the project uses **cohort-based behavioral baselines and transaction-similarity analysis** instead of true user-, device-, or IP-level behavioral modeling.

The original `creditcard.csv` file is not included in the repository because of its size.

## Project Pipeline

```text
Raw Transaction Data
        ↓
Data Cleaning & Deduplication
        ↓
Exploratory Data Analysis
        ↓
Cohort Construction
        ↓
Feature Engineering
        ↓
Statistical Feature Validation
        ↓
Robust Anomaly Detection
        ↓
Logistic Regression
        ↓
Probability Calibration
        ↓
AUPRC / Precision / Recall Evaluation
        ↓
Cost-Based Threshold Analysis
        ↓
Unsupervised Graph Analysis
        ↓
Fraud-Ring Discovery
        ↓
FastAPI + Streamlit Deployment
```

## How to Run

### 1. Dataset

Download `creditcard.csv` from Kaggle and place it inside:

```text
data/creditcard.csv
```

### 2. Run the analytical pipeline

```bash
python 01_data_prep.py
python 02_anomaly_detection.py
python 03_train_model_v1.py
python 04_validation.py
python 05_model_final.py
python 06_fraud_rings.py
```

The scripts cover:

* Data preparation
* Anomaly detection
* Baseline model development
* Model validation
* Final feature validation and training
* Graph-based fraud analysis

### 3. Launch the dashboard

```bash
streamlit run dashboard.py
```

### 4. Launch the API

```bash
uvicorn api:app --reload
```

The scoring endpoint is available at:

```text
http://127.0.0.1:8000/score
```

Pre-trained model artifacts are included, allowing the API to run without regenerating the complete analytical pipeline.

## Track Alignment

Built for **Razorpay Buildathon — Track 2: AI Risk Manager**.

The project approaches the problem as an end-to-end data science system:

* Statistical anomaly detection
* Feature engineering and statistical validation
* Supervised classification
* Probability calibration
* Imbalanced-learning evaluation
* Unsupervised community detection
* Cost-sensitive threshold analysis
* Data leakage investigation
* Model deployment

The evidence-report auto-responder additionally explores the verifier/auto-responder direction described in the track.

## Limitations

* The analysis uses a single public dataset rather than production transaction data.
* User, device, and IP identifiers are unavailable because the original dataset anonymizes these fields.
* Cohort-level behavioral features therefore act as proxies for user-level behavioral modeling.
* Velocity and device information shown in evidence reports are illustrative proxies rather than real production signals.
* The graph analysis identifies transaction communities but does not establish why those transactions are related.
* Because the underlying features are anonymized through PCA, the discovered relationships cannot always be translated into directly interpretable real-world behavioral explanations.

## What This Project Demonstrates

Fraud Radar was designed not simply to train a classifier, but to demonstrate a complete **data science workflow from raw data to deployable analytical output**:

**Clean the data → understand the distribution → engineer features → validate assumptions statistically → build models → evaluate honestly → investigate anomalies in model performance → quantify business trade-offs → deploy the result.**
