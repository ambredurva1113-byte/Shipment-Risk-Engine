import pandas as pd
import numpy as np


def compute_delay_ratio(row):
    if row["Expected_Days"] == 0:
        return 0
    return round(row["Delay_Days"] / row["Expected_Days"], 3)


def compute_loss_score(row):
    return round(row["Return_Rate"] + row["Damage_Rate"], 2)


def build_supplier_profiles(df):
    """
    For each supplier, compute their historical averages.
    This becomes the basis for the trust score.
    """
    profile = df.groupby("Supplier_Name").agg(
        avg_delay=("Delay_Days", "mean"),
        avg_damage=("Damage_Rate", "mean"),
        avg_return=("Return_Rate", "mean"),
        total_shipments=("Shipment_ID", "count"),
        high_risk_count=("Risk_Label", lambda x: (x == "High").sum()),
        on_time_count=("Delay_Days", lambda x: (x == 0).sum()),
    ).reset_index()

    profile["on_time_rate"] = profile["on_time_count"] / profile["total_shipments"]
    profile["high_risk_rate"] = profile["high_risk_count"] / profile["total_shipments"]

    return profile


def compute_trust_score(row):
    """
    Trust Score = weighted combination of delay, damage, return, consistency.
    Scale: 0 to 100. Higher = more trustworthy.
    """

    # Delay component (40%) — normalize delay to 0-100 penalty
    # Assuming max avg delay of 15 days as worst case
    delay_penalty = min(row["avg_delay"] / 15, 1) * 100
    delay_score = 100 - delay_penalty

    # Damage component (30%)
    damage_penalty = min(row["avg_damage"] / 25, 1) * 100
    damage_score = 100 - damage_penalty

    # Return component (20%)
    return_penalty = min(row["avg_return"] / 20, 1) * 100
    return_score = 100 - return_penalty

    # Consistency component (10%) — based on on-time delivery rate
    consistency_score = row["on_time_rate"] * 100

    trust = (
        0.40 * delay_score +
        0.30 * damage_score +
        0.20 * return_score +
        0.10 * consistency_score
    )

    return round(trust, 1)


def get_recommendation(trust_score):
    if trust_score >= 80:
        return "✅ Preferred Supplier — continue orders"
    elif trust_score >= 60:
        return "⚠️ Monitor Closely — request SLA guarantees"
    elif trust_score >= 40:
        return "🔴 High Risk — reduce order volume, find alternatives"
    else:
        return "❌ Blacklist Candidate — escalate to management"


def apply_business_rules(row):
    """
    Hybrid intelligence: ML prediction + business rules override.
    Business rules can override ML if hard thresholds are crossed.
    """
    risk = row["Risk_Label"]  # ML prediction

    # Hard overrides based on business logic
    if row["Damage_Rate"] > 20:
        risk = "High"

    if row["Delay_Days"] > 10:
        risk = "High"

    if row["Return_Rate"] > 15 and row["Damage_Rate"] > 12:
        risk = "High"

    # Downgrade only if everything looks clean
    if row["Delay_Days"] == 0 and row["Damage_Rate"] < 3 and row["Return_Rate"] < 3:
        risk = "Low"

    return risk


def run_feature_engineering(df):
    df = df.copy()
    df["Delay_Ratio"] = df.apply(compute_delay_ratio, axis=1)
    df["Loss_Score"] = df.apply(compute_loss_score, axis=1)
    df["Risk_Label_Final"] = df.apply(apply_business_rules, axis=1)
    return df


def run_trust_engine(df):
    profiles = build_supplier_profiles(df)
    profiles["Trust_Score"] = profiles.apply(compute_trust_score, axis=1)
    profiles["Recommendation"] = profiles["Trust_Score"].apply(get_recommendation)
    return profiles


if __name__ == "__main__":
    df = pd.read_csv("shipment_data.csv")
    df = run_feature_engineering(df)

    trust_df = run_trust_engine(df)
    print("\n=== Supplier Trust Scores ===")
    print(trust_df[["Supplier_Name", "Trust_Score", "avg_delay", "avg_damage", "Recommendation"]]
          .sort_values("Trust_Score", ascending=False).to_string(index=False))

    df.to_csv("shipment_features.csv", index=False)
    trust_df.to_csv("supplier_trust.csv", index=False)
    print("\nFiles saved.")
