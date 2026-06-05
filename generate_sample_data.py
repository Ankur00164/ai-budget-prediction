"""
Generate a sample financial budget dataset for testing the AI Budget Prediction System.
Run: python generate_sample_data.py
"""
import random
import pandas as pd
from datetime import date, timedelta

random.seed(42)

DEPARTMENTS = ['Finance', 'HR', 'IT', 'Marketing', 'Operations',
               'Sales', 'Legal', 'R&D', 'Admin', 'Procurement']
CATEGORIES  = ['Salaries', 'Software', 'Hardware', 'Training', 'Travel',
               'Advertising', 'Logistics', 'Utilities', 'Supplies', 'Consulting']
REGIONS     = ['North', 'South', 'East', 'West', 'Central']
PAYMENTS    = ['Bank Transfer', 'Credit Card', 'Cash', 'Online', 'Cheque']

def random_date(start='2023-01-01', end='2024-12-31'):
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    return s + timedelta(days=random.randint(0, (e - s).days))

rows = []
for i in range(1, 501):
    budget = round(random.uniform(10000, 150000), 2)
    # Introduce realistic deviations
    deviation_factor = random.gauss(1.0, 0.15)   # mean=100%, std=15%
    actual = round(budget * deviation_factor, 2)
    rows.append({
        'Date':           random_date(),
        'Department':     random.choice(DEPARTMENTS),
        'Category':       random.choice(CATEGORIES),
        'Region':         random.choice(REGIONS),
        'Budget Amount':  budget,
        'Actual Amount':  actual,
        'Payment Method': random.choice(PAYMENTS),
        'Transaction ID': f'TXN{i:04d}',
    })

df = pd.DataFrame(rows)
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv('sample_budget_data.csv', index=False)
df.to_excel('sample_budget_data.xlsx', index=False)
print(f"[OK] Sample dataset created: {len(df)} rows")
print(f"   -> sample_budget_data.csv")
print(f"   -> sample_budget_data.xlsx")
