import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from sklearn.covariance import MinCovDet


normal_train = pd.read_csv('data/normal_train.csv')
test_df = pd.read_csv('data/test_df.csv')

print(normal_train.shape)
print(test_df.shape)
print(normal_train['Cohort'].value_counts())


normal_train['LogAmount'] = np.log1p(normal_train['Amount'])
test_df['LogAmount'] = np.log1p(test_df['Amount'])

# Start with a small, defensible feature set for the distance calc
features = ['LogAmount', 'V1', 'V2', 'V3', 'V4', 'V14', 'V17']

print(normal_train[features].describe())

def fit_naive_baseline(cohort_data, features):
    """Returns mean vector and inverse covariance matrix for a cohort."""
    X = cohort_data[features].values
    mean_vec = X.mean(axis=0)
    cov_matrix = np.cov(X, rowvar=False)
    inv_cov = inv(cov_matrix)
    return mean_vec, inv_cov

# Fit a baseline per cohort
cohort_baselines = {}
for cohort_name, group in normal_train.groupby('Cohort'):
    mean_vec, inv_cov = fit_naive_baseline(group, features)
    cohort_baselines[cohort_name] = (mean_vec, inv_cov)

print(f"Fitted baselines for {len(cohort_baselines)} cohorts")

# Sanity check: compute distance for a few training rows against their own cohort baseline
sample_row = normal_train.iloc[0]
cohort = sample_row['Cohort']
mean_vec, inv_cov = cohort_baselines[cohort]
x = sample_row[features].values.astype(float)

dist = mahalanobis(x, mean_vec, inv_cov)
print(f"Sample transaction cohort: {cohort}, Mahalanobis distance: {dist:.3f}")

def compute_distance(row, baselines, features):
    cohort = row['Cohort']
    if cohort not in baselines:
        return np.nan
    mean_vec, inv_cov = baselines[cohort]
    x = row[features].values.astype(float)
    return mahalanobis(x, mean_vec, inv_cov)

test_df['MahalanobisDist'] = test_df.apply(
    lambda row: compute_distance(row, cohort_baselines, features), axis=1
)

print(test_df.groupby('Class')['MahalanobisDist'].describe())

def fit_robust_baseline(cohort_data, features):
    X = cohort_data[features].values
    mcd = MinCovDet(random_state=42).fit(X)
    return mcd

robust_baselines = {}
for cohort_name, group in normal_train.groupby('Cohort'):
    mcd = fit_robust_baseline(group, features)
    robust_baselines[cohort_name] = mcd

print(f"Fitted robust baselines for {len(robust_baselines)} cohorts")

def compute_robust_distance(row, baselines, features):
    cohort = row['Cohort']
    if cohort not in baselines:
        return np.nan
    mcd = baselines[cohort]
    x = row[features].values.astype(float).reshape(1, -1)
    return np.sqrt(mcd.mahalanobis(x))[0]

test_df['RobustMahalanobisDist'] = test_df.apply(
    lambda row: compute_robust_distance(row, robust_baselines, features), axis=1
)

print(test_df.groupby('Class')['RobustMahalanobisDist'].describe())

# Controlled experiment: inject synthetic outliers into ONE cohort's training data,
# then check whether naive vs robust baselines get dragged by them

test_cohort = 'high_afternoon'
clean_group = normal_train[normal_train['Cohort'] == test_cohort][features].values

# Inject 5% synthetic extreme outliers into the training pool
n_outliers = int(0.05 * len(clean_group))
np.random.seed(42)
synthetic_outliers = np.random.normal(loc=15, scale=1, size=(n_outliers, len(features)))
contaminated_group = np.vstack([clean_group, synthetic_outliers])

# Fit naive on contaminated data
naive_mean = contaminated_group.mean(axis=0)
naive_cov = np.cov(contaminated_group, rowvar=False)
naive_inv_cov = inv(naive_cov)

# Fit robust on contaminated data
mcd_contaminated = MinCovDet(random_state=42).fit(contaminated_group)

# Now test: does a genuinely normal point (from clean, uncontaminated data) 
# still look "normal" under each baseline?
test_point = clean_group[0]

naive_dist_after_contamination = mahalanobis(test_point, naive_mean, naive_inv_cov)
robust_dist_after_contamination = np.sqrt(mcd_contaminated.mahalanobis(test_point.reshape(1, -1)))[0]

print(f"Naive distance for a normal point (baseline contaminated with outliers): {naive_dist_after_contamination:.3f}")
print(f"Robust distance for the same normal point (same contaminated baseline): {robust_dist_after_contamination:.3f}")

outlier_point = synthetic_outliers[0]

naive_dist_for_outlier = mahalanobis(outlier_point, naive_mean, naive_inv_cov)
robust_dist_for_outlier = np.sqrt(mcd_contaminated.mahalanobis(outlier_point.reshape(1, -1)))[0]

print(f"Naive distance for an INJECTED OUTLIER (contaminated baseline): {naive_dist_for_outlier:.3f}")
print(f"Robust distance for the SAME injected outlier (contaminated baseline): {robust_dist_for_outlier:.3f}")
test_df.to_csv('data/test_df_scored.csv', index=False)
print("Saved scored test set to data/test_df_scored.csv")