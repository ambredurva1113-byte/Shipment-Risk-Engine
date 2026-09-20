"""
risk_rules.py -- Business Rules Layer (DESIGN THINKING: Prototype)
Hard thresholds a manager can explain. Rules only ever ESCALATE risk, never lower it.
"""
import numpy as np
import pandas as pd

RANK = {"Low": 0, "Medium": 1, "High": 2}

RULE_TEXT = {
    "R1": "Supplier trust score below 30",
    "R2": "High-value order (> Rs 5 lakh) with a return-prone supplier",
    "R3": "Monsoon month with a weak supplier (trust < 50)",
    "R4": "Supplier trust score below 50",
}


def rules_triggered(df):
    """Boolean frame: which rule fired for which row, with the level it escalates to."""
    return pd.DataFrame({
        "R1": df.trust_score < 30,
        "R2": (df.high_value == 1) & (df.prior_return_rate > 0.15),
        "R3": (df.monsoon == 1) & (df.trust_score < 50),
        "R4": df.trust_score < 50,
    }, index=df.index)


RULE_LEVEL = {"R1": "High", "R2": "High", "R3": "High", "R4": "Medium"}


def apply_business_rules(df, ml_pred):
    """
    PROTOTYPE: Business Rules Layer. Hard thresholds override the ML so a
    manager can always explain a decision.  Rules only ever ESCALATE risk.
    """
    pred = pd.Series(ml_pred, index=df.index).copy()
    rank = RANK
    fired = rules_triggered(df)
    for r in ["R4", "R1", "R2", "R3"]:
        mask = fired[r]
        cur = pred[mask].map(rank)
        pred.loc[mask] = np.where(cur < rank[RULE_LEVEL[r]], RULE_LEVEL[r], pred[mask])
    return pred.values


