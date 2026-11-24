"""Generate the custom dataset used by the notebook (customer_segments.csv).

A synthetic 3-class problem with numeric + categorical features and a few
missing values, so the pandas exploration step has something to find.
Replace data/customer_segments.csv with your own CSV to use real data.
"""

import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 900
classes = ["budget", "standard", "premium"]
y = rng.choice(classes, size=N, p=[0.40, 0.38, 0.22])

center = {
    "budget": dict(age=28, income=32_000, spend=250, visits=4),
    "standard": dict(age=39, income=58_000, spend=620, visits=8),
    "premium": dict(age=47, income=104_000, spend=1_400, visits=14),
}
region_probs = {
    "budget": [0.45, 0.30, 0.15, 0.10],
    "standard": [0.25, 0.30, 0.25, 0.20],
    "premium": [0.10, 0.20, 0.30, 0.40],
}
regions = ["rural", "suburban", "urban", "metro"]

rows = []
for label in y:
    c = center[label]
    rows.append(
        {
            "age": int(np.clip(rng.normal(c["age"], 8), 18, 80)),
            "annual_income": round(float(np.clip(rng.normal(c["income"], c["income"] * 0.25), 8_000, None)), 2),
            "monthly_spend": round(float(np.clip(rng.normal(c["spend"], c["spend"] * 0.35), 20, None)), 2),
            "visits_per_month": int(np.clip(rng.normal(c["visits"], 3), 0, 31)),
            "region": rng.choice(regions, p=region_probs[label]),
            "segment": label,
        }
    )

df = pd.DataFrame(rows)

# Inject a few missing values to practise cleaning
for col, n in [("annual_income", 12), ("monthly_spend", 8)]:
    df.loc[rng.choice(df.index, n, replace=False), col] = np.nan

os.makedirs("data", exist_ok=True)
df.to_csv("data/customer_segments.csv", index=False)
print(f"Wrote data/customer_segments.csv with {len(df)} rows")
print(df["segment"].value_counts())
