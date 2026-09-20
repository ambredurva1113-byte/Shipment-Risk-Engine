"""
trust_engine.py  --  DESIGN THINKING STAGE 2 (DEFINE) + feature engineering
----------------------------------------------------------------------------
Problem statement (Define):
  "Importers need to know which supplier to trust and which shipment will go
   wrong BEFORE dispatch, because today they only learn after the loss."

So the model may only use information available BEFORE the shipment leaves.
The supplier's trust score is therefore computed from PAST shipments only
(no peeking at the current shipment's outcome)  ->  no data leakage.
"""
import numpy as np
import pandas as pd

W_DELAY, W_DAMAGE, W_RETURN = 0.40, 0.35, 0.25   # business weights (returns hurt, delay hurts most often)
SMOOTH_K = 5                                      # shrink small histories toward the global average


def trust_score(delay_rate, damage_rate, return_rate):
    """0-100 score. 100 = perfect supplier."""
    penalty = W_DELAY * delay_rate + W_DAMAGE * damage_rate + W_RETURN * return_rate
    return np.clip(100 * (1 - 1.4 * penalty), 0, 100)


def supplier_trust_table(df):
    """Final trust table using each supplier's full history (for the dashboard)."""
    g = df.groupby(["supplier_id", "supplier_name"]).agg(
        shipments=("shipment_id", "count"),
        delay_rate=("delayed", "mean"),
        damage_rate=("damaged", "mean"),
        return_rate=("returned", "mean"),
        avg_delay_days=("delay_days", "mean"),
    ).reset_index()
    g["trust_score"] = trust_score(g.delay_rate, g.damage_rate, g.return_rate).round(1)
    g["trust_band"] = pd.cut(g.trust_score, [-1, 50, 70, 101], labels=["Untrusted", "Watch", "Trusted"])
    return g.sort_values("trust_score", ascending=False)


def add_prior_features(df):
    """
    For every shipment, compute the supplier's record from EARLIER shipments only.
    Uses expanding mean shifted by one row, smoothed toward the global rate.
    """
    df = df.sort_values(["ship_date", "shipment_id"]).reset_index(drop=True)
    gl = {c: df[c].mean() for c in ["delayed", "damaged", "returned"]}

    for col, name in [("delayed", "prior_delay_rate"), ("damaged", "prior_damage_rate"),
                      ("returned", "prior_return_rate")]:
        cum = df.groupby("supplier_id")[col].cumsum() - df[col]          # sum of PREVIOUS rows
        n = df.groupby("supplier_id").cumcount()                          # number of PREVIOUS rows
        df[name] = (cum + SMOOTH_K * gl[col]) / (n + SMOOTH_K)
    df["prior_shipments"] = df.groupby("supplier_id").cumcount()

    # last-5 trend: is the supplier getting worse recently? (Empathize: "he seems to be slipping lately")
    bad = ((df.delayed + df.damaged + df.returned) > 0).astype(int)
    prev_bad = bad.groupby(df.supplier_id).shift(1)
    df["recent_bad_rate"] = (prev_bad.groupby(df.supplier_id)
                             .transform(lambda s: s.rolling(5, min_periods=1).mean())
                             .fillna(bad.mean()))

    df["prior_delay_days"] = (df.groupby("supplier_id")["delay_days"].cumsum() - df["delay_days"])
    df["prior_delay_days"] = (df["prior_delay_days"] / df["prior_shipments"].clip(lower=1)).fillna(0)

    df["trust_score"] = trust_score(df.prior_delay_rate, df.prior_damage_rate, df.prior_return_rate).round(1)
    df["high_value"] = (df.order_value > 500000).astype(int)
    df["value_lakh"] = (df.order_value / 100000).round(2)
    return df


if __name__ == "__main__":
    raw = pd.read_csv("shipment_data.csv", parse_dates=["ship_date"])
    feats = add_prior_features(raw)
    feats.to_csv("shipment_features.csv", index=False)
    table = supplier_trust_table(raw)
    table.to_csv("supplier_trust.csv", index=False)
    print(table[["supplier_id", "supplier_name", "shipments", "trust_score", "trust_band"]].to_string(index=False))
    print("\nSaved shipment_features.csv and supplier_trust.csv")
