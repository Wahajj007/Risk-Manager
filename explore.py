import pandas as pd

df = pd.read_csv('data/creditcard.csv')

print(df.shape)
print(df.columns.tolist())
print(df['Class'].value_counts())
print(df['Class'].value_counts(normalize=True))
print(df.head())
# Derive hour-of-day from Time (seconds since first transaction, wraps every 24h)
df['Hour'] = (df['Time'] // 3600) % 24

# Look at Amount distribution to decide bucket cutoffs
print(df['Amount'].describe())
print(df['Hour'].value_counts().sort_index())

# Amount tier based on the distribution we just saw
def amount_tier(a):
    if a <= 5.6:
        return 'low'
    elif a <= 22:
        return 'mid_low'
    elif a <= 77:
        return 'mid_high'
    else:
        return 'high'

df['AmountTier'] = df['Amount'].apply(amount_tier)

# Time-of-day bucket
def time_period(h):
    if 0 <= h < 6:
        return 'night'
    elif 6 <= h < 12:
        return 'morning'
    elif 12 <= h < 18:
        return 'afternoon'
    else:
        return 'evening'

df['TimePeriod'] = df['Hour'].apply(time_period)

# Combine into a cohort label
df['Cohort'] = df['AmountTier'] + '_' + df['TimePeriod']

print(df['Cohort'].value_counts())

from sklearn.model_selection import train_test_split

# Only non-fraud transactions can be used to learn "normal" baselines
normal_df = df[df['Class'] == 0].copy()
fraud_df = df[df['Class'] == 1].copy()

# Split normal transactions: most go into baseline-building, some held out for testing
normal_train, normal_test = train_test_split(normal_df, test_size=0.2, random_state=42)

# Test set = held-out normal transactions + ALL fraud transactions
test_df = pd.concat([normal_test, fraud_df], ignore_index=True)

print("Baseline (normal_train):", normal_train.shape)
print("Test set:", test_df.shape)
print(test_df['Class'].value_counts())