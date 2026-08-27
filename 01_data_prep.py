import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/creditcard.csv')

before_dedup = len(df)
v_cols = [f'V{i}' for i in range(1, 29)]
before_dedup = len(df)
df = df.drop_duplicates(subset=v_cols, keep='first').reset_index(drop=True)
print(f"Removed {before_dedup - len(df)} duplicate rows ({before_dedup} -> {len(df)})")


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



normal_df = df[df['Class'] == 0].copy()
fraud_df = df[df['Class'] == 1].copy()

normal_train, normal_test = train_test_split(normal_df, test_size=0.2, random_state=42)
test_df = pd.concat([normal_test, fraud_df], ignore_index=True)

normal_train.to_csv('data/normal_train.csv', index=False)
test_df.to_csv('data/test_df.csv', index=False)
print("Saved splits to data/")

