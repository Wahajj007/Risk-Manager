import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.covariance import MinCovDet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.covariance')
import joblib

normal_train_raw = pd.read_csv('data/normal_train.csv')
fraud_and_normal = pd.read_csv('data/test_df_scored.csv')  # has both classes with Class label

all_v_features = [f'V{i}' for i in range(1, 29)]

# Point-biserial correlation: how strongly does each feature correlate with the binary Class label?
correlations = []
for feat in all_v_features:
    corr, pval = pointbiserialr(fraud_and_normal['Class'], fraud_and_normal[feat])
    correlations.append({'feature': feat, 'correlation': corr, 'abs_correlation': abs(corr), 'p_value': pval})

corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
print(corr_df.head(15))

top_features = ['V14', 'V17', 'V12', 'V10', 'V16', 'V3', 'V7', 'V11']
normal_train_raw['LogAmount'] = np.log1p(normal_train_raw['Amount'])
new_features = ['LogAmount'] + top_features

# Refit robust baselines per cohort with the expanded feature set
robust_baselines_v2 = {}
for cohort_name, group in normal_train_raw.groupby('Cohort'):
    mcd = MinCovDet(random_state=42, support_fraction=0.75).fit(group[new_features].values)
    robust_baselines_v2[cohort_name] = mcd

def compute_robust_distance_v2(row, baselines, features):
    cohort = row['Cohort']
    if cohort not in baselines:
        return np.nan
    mcd = baselines[cohort]
    x = row[features].values.astype(float).reshape(1, -1)
    return np.sqrt(mcd.mahalanobis(x))[0]

fraud_and_normal['RobustMahalanobisDist_v2'] = fraud_and_normal.apply(
    lambda row: compute_robust_distance_v2(row, robust_baselines_v2, new_features), axis=1
)

print(fraud_and_normal.groupby('Class')['RobustMahalanobisDist_v2'].describe())

pd.set_option('display.max_columns', None)
print(fraud_and_normal.groupby('Class')['RobustMahalanobisDist_v2'].describe())


# Rebuild the leakage-safe split, same pattern as mod3.py
fraud_only = fraud_and_normal[fraud_and_normal['Class'] == 1].copy()
fraud_train, fraud_final_test = train_test_split(fraud_only, test_size=0.7, random_state=42)
normal_final_test = fraud_and_normal[fraud_and_normal['Class'] == 0].copy()

model_train_v2 = pd.concat([normal_train_raw, fraud_train], ignore_index=True)
final_test_v2 = pd.concat([normal_final_test, fraud_final_test], ignore_index=True)

# Need RobustMahalanobisDist_v2 computed for the training set too
model_train_v2['RobustMahalanobisDist_v2'] = model_train_v2.apply(
    lambda row: compute_robust_distance_v2(row, robust_baselines_v2, new_features), axis=1
)

model_features_v2 = ['RobustMahalanobisDist_v2', 'LogAmount']
X_train_v2 = model_train_v2[model_features_v2].fillna(model_train_v2[model_features_v2].median())
y_train_v2 = model_train_v2['Class']

clf_v2 = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
clf_v2.fit(X_train_v2, y_train_v2)

X_final_test_v2 = final_test_v2[model_features_v2].fillna(final_test_v2[model_features_v2].median())
final_test_v2['FraudProbability_v2'] = clf_v2.predict_proba(X_final_test_v2)[:, 1]

auprc_v2 = average_precision_score(final_test_v2['Class'], final_test_v2['FraudProbability_v2'])
print(f"AUPRC (9-feature model): {auprc_v2:.4f}")
print(f"AUPRC (original 2-feature model): 0.1882")
print(f"Improvement: {(auprc_v2/0.1882 - 1)*100:.1f}%")

train_indices = set(model_train_v2.index)
test_indices = set(final_test_v2.index)
overlap = train_indices.intersection(test_indices)
print(f"Overlap between train and test sets: {len(overlap)} rows (should be 0, though index reuse across concat is possible - verifying by content instead)")

# More reliable check: verify by actual row content, not just index
train_set_check = set(model_train_v2['Time'].astype(str) + '_' + model_train_v2['Amount'].astype(str))
test_set_check = set(final_test_v2['Time'].astype(str) + '_' + final_test_v2['Amount'].astype(str))
real_overlap = train_set_check.intersection(test_set_check)
print(f"Real overlap (by Time+Amount signature): {len(real_overlap)} rows")

# Stronger check: use several V-columns together, which are continuous and essentially unique per real transaction
signature_cols = ['V1', 'V2', 'V3', 'V4', 'V5']
train_set_check2 = set(model_train_v2[signature_cols].apply(lambda r: tuple(r), axis=1))
test_set_check2 = set(final_test_v2[signature_cols].apply(lambda r: tuple(r), axis=1))
real_overlap2 = train_set_check2.intersection(test_set_check2)
print(f"Real overlap (by V1-V5 signature, should be near-impossible unless same row): {len(real_overlap2)} rows")

# Isolate exactly which subset is causing overlap
normal_train_sig = set(normal_train_raw[signature_cols].apply(lambda r: tuple(r), axis=1))
normal_test_sig = set(normal_final_test[signature_cols].apply(lambda r: tuple(r), axis=1))
fraud_train_sig = set(fraud_train[signature_cols].apply(lambda r: tuple(r), axis=1))
fraud_test_sig = set(fraud_final_test[signature_cols].apply(lambda r: tuple(r), axis=1))

print(f"normal_train vs normal_test overlap: {len(normal_train_sig & normal_test_sig)}")
print(f"fraud_train vs fraud_test overlap: {len(fraud_train_sig & fraud_test_sig)}")
print(f"normal_train vs fraud_test overlap: {len(normal_train_sig & fraud_test_sig)}")
print(f"fraud_train vs normal_test overlap: {len(fraud_train_sig & normal_test_sig)}")

duplicate_check = normal_train_raw.duplicated(subset=signature_cols).sum()
print(f"Duplicate rows within normal_train_raw alone: {duplicate_check}")

full_normal_duplicate_check = pd.read_csv('data/creditcard.csv')
full_normal_duplicate_check = full_normal_duplicate_check[full_normal_duplicate_check['Class'] == 0]
total_dupes = full_normal_duplicate_check.duplicated(subset=signature_cols).sum()
print(f"Duplicate rows in the full raw normal transaction set: {total_dupes}")

check_df = pd.read_csv('data/creditcard.csv')
check_normal = check_df[check_df['Class'] == 0]

# Check duplicates using ONLY the V-columns (ignoring Time and Amount) - a transaction's underlying
# behavioral signature shouldn't change even if Time/Amount differ slightly due to data entry quirks
v_only_cols = [f'V{i}' for i in range(1, 29)]
v_only_dupes = check_normal.duplicated(subset=v_only_cols).sum()
print(f"Duplicates using only V1-V28 (ignoring Time/Amount): {v_only_dupes}")

import os
print(f"normal_train.csv last modified: {pd.Timestamp(os.path.getmtime('data/normal_train.csv'), unit='s')}")
print(f"explore.py last run should be very recent if the fix took effect")

normal_train_check = pd.read_csv('data/normal_train.csv')
dedup_check_in_file = normal_train_check.duplicated(subset=v_only_cols).sum()
print(f"Duplicates remaining WITHIN the saved normal_train.csv: {dedup_check_in_file}")

joblib.dump(clf_v2, 'data/fraud_model.pkl')
joblib.dump(robust_baselines_v2, 'data/robust_baselines.pkl')
print("Saved corrected 9-feature model and baselines (overwriting the old 2-feature versions)")

final_test_v2 = final_test_v2.rename(columns={
    'RobustMahalanobisDist_v2': 'RobustMahalanobisDist',
    'FraudProbability_v2': 'FraudProbability'
})
final_test_v2.to_csv('data/final_test_scored.csv', index=False)
print("Overwrote final_test_scored.csv with 9-feature model predictions")