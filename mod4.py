import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve

final_test_df = pd.read_csv('data/final_test_scored.csv')

print(final_test_df.shape)
print(final_test_df['Class'].value_counts())
print(final_test_df['FraudProbability'].describe())


y_true = final_test_df['Class']
y_prob = final_test_df['FraudProbability']

# Start with the standard 0.5 threshold just to see where we land
y_pred_default = (y_prob >= 0.5).astype(int)

print("=== Threshold = 0.5 ===")
print(classification_report(y_true, y_pred_default, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred_default))

# Illustrative costs — document these assumptions clearly in your pitch
cost_false_positive = 5    # lost goodwill/support cost of wrongly blocking a legit transaction
cost_false_negative = 100  # average fraud loss when a real fraud gets through

thresholds = np.arange(0.1, 0.95, 0.05)
results = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    total_cost = fp * cost_false_positive + fn * cost_false_negative
    results.append({'threshold': round(t, 2), 'precision': round(precision, 3),
                     'recall': round(recall, 3), 'false_positives': fp,
                     'false_negatives': fn, 'total_cost': total_cost})

results_df = pd.DataFrame(results)
print(results_df)

best_row = results_df.loc[results_df['total_cost'].idxmin()]
print("\nBest threshold by cost:", best_row)



prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')

for pt, pp in zip(prob_true, prob_pred):
    print(f"Predicted ~{pp:.3f} -> Actual fraud rate: {pt:.3f}")
    
print(final_test_df[['FraudProbability', 'CalibratedFraudProbability']].describe())

prob_true_cal, prob_pred_cal = calibration_curve(
    final_test_df['Class'], final_test_df['CalibratedFraudProbability'], n_bins=10, strategy='quantile'
)
for pt, pp in zip(prob_true_cal, prob_pred_cal):
    print(f"Calibrated predicted ~{pp:.3f} -> Actual fraud rate: {pt:.3f}")