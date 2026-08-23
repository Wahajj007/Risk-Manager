from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.spatial.distance import mahalanobis

app = FastAPI(title="Bayesian Fraud Radar API")

# Load model and baselines once at startup
clf = joblib.load('data/fraud_model.pkl')
robust_baselines = joblib.load('data/robust_baselines.pkl')

features = ['LogAmount', 'V1', 'V2', 'V3', 'V4', 'V14', 'V17']
model_features = ['RobustMahalanobisDist', 'LogAmount']

THRESHOLD = 0.75  # matches the cost-justified threshold from Day 4

class Transaction(BaseModel):
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V14: float
    V17: float
    Cohort: str

@app.post("/score")
def score_transaction(txn: Transaction):
    log_amount = np.log1p(txn.Amount)

    if txn.Cohort not in robust_baselines:
        return {"error": f"Unknown cohort '{txn.Cohort}'"}

    mcd = robust_baselines[txn.Cohort]
    x_vec = np.array([log_amount, txn.V1, txn.V2, txn.V3, txn.V4, txn.V14, txn.V17]).reshape(1, -1)
    robust_dist = np.sqrt(mcd.mahalanobis(x_vec))[0]

    model_input = np.array([[robust_dist, log_amount]])
    fraud_prob = clf.predict_proba(model_input)[0, 1]

    if fraud_prob >= THRESHOLD:
        action = "auto-block"
    elif fraud_prob >= THRESHOLD * 0.5:
        action = "flag for review"
    else:
        action = "approve"

    return {
        "fraud_probability": round(float(fraud_prob), 4),
        "robust_mahalanobis_distance": round(float(robust_dist), 4),
        "action": action,
        "threshold_used": THRESHOLD
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": clf is not None}