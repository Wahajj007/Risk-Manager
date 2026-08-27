import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import joblib

test_df = pd.read_csv('data/test_df_scored.csv')

print(test_df.shape)
print(test_df.columns.tolist())
print(test_df[['MahalanobisDist', 'RobustMahalanobisDist', 'Class']].describe())

normal_train_raw = pd.read_csv('data/normal_train.csv')
normal_train_raw['LogAmount'] = np.log1p(normal_train_raw['Amount'])

features = ['LogAmount', 'V1', 'V2', 'V3', 'V4', 'V14', 'V17']

# Refit robust baselines on normal_train (same as Day 2, just redone here for clarity)
robust_baselines = {}
for cohort_name, group in normal_train_raw.groupby('Cohort'):
    mcd = MinCovDet(random_state=42).fit(group[features].values)
    robust_baselines[cohort_name] = mcd

def compute_robust_distance(row, baselines, features):
    cohort = row['Cohort']
    if cohort not in baselines:
        return np.nan
    mcd = baselines[cohort]
    x = row[features].values.astype(float).reshape(1, -1)
    return np.sqrt(mcd.mahalanobis(x))[0]

normal_train_raw['RobustMahalanobisDist'] = normal_train_raw.apply(
    lambda row: compute_robust_distance(row, robust_baselines, features), axis=1
)

# Now split: we need SOME fraud examples to train the classifier on (it can't learn
# fraud patterns from zero fraud examples), so split the fraud transactions in test_df
# into a "train-fraud" portion and a "final-test-fraud" portion, keeping most for final testing
fraud_only = test_df[test_df['Class'] == 1].copy()
fraud_train, fraud_final_test = train_test_split(fraud_only, test_size=0.7, random_state=42)

normal_final_test = test_df[test_df['Class'] == 0].copy()

# Training set for the model: normal_train_raw (all normal) + fraud_train (a small fraud sample)
model_train_df = pd.concat([normal_train_raw, fraud_train], ignore_index=True)

# TRUE final held-out test set: normal_final_test + fraud_final_test (majority of fraud, untouched)
final_test_df = pd.concat([normal_final_test, fraud_final_test], ignore_index=True)

print("Model training set:", model_train_df.shape, model_train_df['Class'].value_counts().to_dict())
print("Final held-out test set:", final_test_df.shape, final_test_df['Class'].value_counts().to_dict())

# Simple velocity proxy: how many transactions in the same cohort within a nearby Time window
# (approximation, since we don't have real per-user velocity - explain this as illustrative in the pitch)
model_train_df = model_train_df.sort_values('Time').reset_index(drop=True)

model_features = ['RobustMahalanobisDist', 'LogAmount']

X_train = model_train_df[model_features].fillna(model_train_df[model_features].median())
y_train = model_train_df['Class']

clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
clf.fit(X_train, y_train)
print("Model trained.")
print("Coefficients:", dict(zip(model_features, clf.coef_[0])))
print("Intercept:", clf.intercept_[0])

# Define X_final_test BEFORE using it anywhere below
X_final_test = final_test_df[model_features].fillna(final_test_df[model_features].median())

calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
calibrated_clf.fit(X_train, y_train)

final_test_df['CalibratedFraudProbability'] = calibrated_clf.predict_proba(X_final_test)[:, 1]

prob_true_cal, prob_pred_cal = calibration_curve(
    final_test_df['Class'], final_test_df['CalibratedFraudProbability'], n_bins=10, strategy='quantile'
)
for pt, pp in zip(prob_true_cal, prob_pred_cal):
    print(f"Calibrated predicted ~{pp:.3f} -> Actual fraud rate: {pt:.3f}")

final_test_df['FraudProbability'] = clf.predict_proba(X_final_test)[:, 1]

print(final_test_df.groupby('Class')['FraudProbability'].describe())

# Save for Day 4 validation
final_test_df.to_csv('data/final_test_scored.csv', index=False)
print("Saved final scored test set.")

joblib.dump(clf, 'data/fraud_model.pkl')
print("Saved trained model to data/fraud_model.pkl")

# Also save the robust baselines, needed to compute RobustMahalanobisDist for new transactions
joblib.dump(robust_baselines, 'data/robust_baselines.pkl')
print("Saved robust baselines to data/robust_baselines.pkl")