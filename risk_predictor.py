"""
risk_predictor.py -- DESIGN THINKING STAGE 4: PROTOTYPE (explainable output)
----------------------------------------------------------------------------
A manager does not trust a black box. For every planned shipment we return:
  1. risk level  (Low / Medium / High)
  2. probabilities
  3. TOP REASONS why (model-agnostic, computed by swapping each input with
     values seen in training and measuring the change in P(High))
  4. business rules that fired
  5. a RECOMMENDED ACTION
"""
import joblib
import numpy as np
import pandas as pd
from risk_rules import rules_triggered, apply_business_rules, RULE_TEXT

CITY_DISTANCE = {"Mumbai": 30, "Thane": 45, "Pune": 150, "Nashik": 190, "Aurangabad": 330,
                 "Kolhapur": 380, "Solapur": 410, "Nagpur": 850, "Amravati": 750, "Nanded": 620}

ACTIONS = {
    "High": "DO NOT ship on open terms. Switch supplier, or require pre-shipment inspection + staged payment/escrow, "
            "and insure the cargo.",
    "Medium": "Ship with live tracking, add 3-5 buffer days to the delivery promise, and confirm packaging.",
    "Low": "Proceed as planned. Standard monitoring is enough.",
}

_model = joblib.load("risk_model.pkl")
_meta = joblib.load("model_meta.pkl")
_hist = pd.read_csv("shipment_features.csv")           # reference distribution for explanations
_raw = pd.read_csv("shipment_data.csv", parse_dates=["ship_date"]).sort_values("ship_date")
_FEATS = _meta["features"]
_SMOOTH_K = 5


def supplier_state(supplier_id):
    """Supplier's CURRENT record, built from all past shipments (same smoothing as training)."""
    h = _raw[_raw.supplier_id == supplier_id]
    g = {c: _raw[c].mean() for c in ["delayed", "damaged", "returned"]}
    n = len(h)
    rate = lambda c: (h[c].sum() + _SMOOTH_K * g[c]) / (n + _SMOOTH_K)
    bad = ((h.delayed + h.damaged + h.returned) > 0).astype(int)
    d, m, r = rate("delayed"), rate("damaged"), rate("returned")
    from trust_engine import trust_score
    return {
        "prior_delay_rate": d, "prior_damage_rate": m, "prior_return_rate": r,
        "recent_bad_rate": float(bad.tail(5).mean()) if n else float(bad.mean()),
        "prior_delay_days": float(h.delay_days.mean()) if n else 0.0,
        "prior_shipments": n, "trust_score": float(round(trust_score(d, m, r), 1)),
    }


def build_row(supplier_id, city, product, transport_mode, order_value, weight_kg, month):
    dist = CITY_DISTANCE[city]
    row = {"supplier_id": supplier_id, "city": city, "product": product, "transport_mode": transport_mode,
           "order_value": order_value, "weight_kg": weight_kg, "distance_km": dist,
           "planned_days": int(7 + dist / 60 + (5 if transport_mode == "Sea+Road" else 0)),
           "month": month, "monsoon": int(month in (6, 7, 8, 9)), **supplier_state(supplier_id)}
    row["high_value"] = int(order_value > 500000)
    return pd.DataFrame([row])


def _p_high(df):
    return _model.predict_proba(df[_FEATS])[:, list(_model.classes_).index("High")]


REASON_TEXT = {
    "supplier_id": "Supplier track record ({supplier_id})",
    "city": "Destination city ({city})",
    "product": "Product type ({product})",
    "transport_mode": "Transport mode ({transport_mode})",
    "order_value": "Order value (Rs {order_value:,})",
    "weight_kg": "Shipment weight ({weight_kg} kg)",
    "distance_km": "Route distance ({distance_km} km)",
    "planned_days": "Planned transit time ({planned_days} days)",
    "month": "Shipping month ({month})",
    "monsoon": "Monsoon season",
    "prior_delay_rate": "Supplier's past delay rate ({prior_delay_rate:.0%})",
    "prior_damage_rate": "Supplier's past damage rate ({prior_damage_rate:.0%})",
    "prior_return_rate": "Supplier's past return rate ({prior_return_rate:.0%})",
    "recent_bad_rate": "Supplier's last-5-shipments trouble rate ({recent_bad_rate:.0%})",
    "prior_delay_days": "Supplier's average delay length ({prior_delay_days:.1f} days)",
    "trust_score": "Supplier trust score ({trust_score:.0f}/100)",
    "prior_shipments": "Amount of supplier history ({prior_shipments} shipments)",
}


def explain(row, top=3, min_effect=0.02):
    """Contribution of feature j = P(High | actual) - mean P(High | feature j swapped with training values)."""
    p_actual = _p_high(row)[0]
    contrib = {}
    for f in _FEATS:
        swapped = _hist[_FEATS + ["high_value"]].copy()
        swapped[f] = row.iloc[0][f]                       # force actual value onto every reference row...
        p_with = _p_high(swapped).mean()
        base = _p_high(_hist[_FEATS + ["high_value"]]).mean()
        contrib[f] = p_with - base                        # ...how much does THIS value alone move P(High)?
    vals = row.iloc[0].to_dict()
    fmt = lambda f, v: {"factor": REASON_TEXT[f].format(**vals), "effect_on_P_high": round(v, 3)}
    risky = sorted([(f, v) for f, v in contrib.items() if v >= min_effect], key=lambda kv: -kv[1])[:top]
    safe = sorted([(f, v) for f, v in contrib.items() if v <= -min_effect], key=lambda kv: kv[1])[:top]
    return [fmt(f, v) for f, v in risky], [fmt(f, v) for f, v in safe]


def predict_shipment(supplier_id, city, product, transport_mode, order_value, weight_kg, month):
    row = build_row(supplier_id, city, product, transport_mode, order_value, weight_kg, month)
    proba = _model.predict_proba(row[_FEATS])[0]
    p = dict(zip(_model.classes_, proba))
    ml = "High" if p["High"] >= _meta["alert_threshold"] else max(p, key=p.get)
    final = apply_business_rules(row, [ml])[0]
    fired = rules_triggered(row).iloc[0]
    drivers, protective = explain(row)
    name = _raw.loc[_raw.supplier_id == supplier_id, "supplier_name"].iloc[0]
    return {
        "supplier": f"{supplier_id} - {name}", "trust_score": row.trust_score.iloc[0],
        "ml_prediction": ml, "final_risk": final,
        "probabilities": {k: round(float(v), 3) for k, v in p.items()},
        "rules_fired": [RULE_TEXT[r] for r, v in fired.items() if v],
        "top_risk_drivers": drivers, "protective_factors": protective,
        "recommended_action": ACTIONS[final],
    }


if __name__ == "__main__":
    import json
    print("--- SCENARIO A: risky supplier, fragile goods, monsoon, far city ---")
    print(json.dumps(predict_shipment("S07", "Nagpur", "Electronics", "Road", 620000, 1800, 7), indent=2))
    print("\n--- SCENARIO B: trusted supplier, short route, dry season ---")
    print(json.dumps(predict_shipment("S01", "Pune", "Machinery Parts", "Rail", 300000, 1500, 2), indent=2))
