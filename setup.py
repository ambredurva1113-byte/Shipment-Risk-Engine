"""
Run this once to set up the entire project.
It generates data, engineers features, and trains the model.
After this, launch the dashboard with: streamlit run app.py
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("Step 1: Generating Dataset...")
print("=" * 50)
exec(open("generate_data.py").read())

print("\n" + "=" * 50)
print("Step 2: Feature Engineering + Trust Scores...")
print("=" * 50)
from trust_engine import run_feature_engineering, run_trust_engine
import pandas as pd

df = pd.read_csv("shipment_data.csv")
df = run_feature_engineering(df)
df.to_csv("shipment_features.csv", index=False)

trust_df = run_trust_engine(df)
trust_df.to_csv("supplier_trust.csv", index=False)

print("\nTrust Scores:")
print(trust_df[["Supplier_Name", "Trust_Score", "Recommendation"]].sort_values(
    "Trust_Score", ascending=False).to_string(index=False))

print("\n" + "=" * 50)
print("Step 3: Training ML Model...")
print("=" * 50)
from train_model import train_model
train_model()

print("\n" + "=" * 50)
print("SETUP COMPLETE!")
print("=" * 50)
print("\nRun the dashboard:")
print("  streamlit run app.py")
