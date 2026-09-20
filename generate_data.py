"""
generate_data.py  --  DESIGN THINKING STAGE 1: EMPATHIZE
---------------------------------------------------------
Persona: an importer in Mumbai who lost money on late / damaged / returned
shipments and only found out AFTER the loss.

Their pain points drive what we simulate:
  * some suppliers are chronically late           -> delay
  * fragile goods arrive damaged                  -> damage
  * bad orders get sent back                      -> return
  * monsoon months + long routes make it worse    -> season / distance

Output: shipment_data.csv (500 shipments, 10 suppliers, 10 cities, 6 products)
NOTE: data is synthetic because real supplier data is confidential.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 500

# ---- 10 suppliers, each with its own reliability profile --------------------
suppliers = pd.DataFrame({
    "supplier_id": [f"S{i:02d}" for i in range(1, 11)],
    "supplier_name": ["Shenzhen ElectroTech", "Guangzhou Textiles", "Ningbo Machinery",
                      "Dubai Chem Traders", "Shanghai FoodCorp", "Foshan Furniture",
                      "Yiwu Bulk Goods", "Tianjin Steelworks", "Hanoi Garments", "Istanbul Trade Co"],
    #                 P(delay) P(damage) P(return)
    "p_delay":  [0.10, 0.18, 0.12, 0.45, 0.25, 0.30, 0.55, 0.15, 0.22, 0.38],
    "p_damage": [0.05, 0.08, 0.06, 0.20, 0.15, 0.25, 0.30, 0.07, 0.10, 0.18],
    "p_return": [0.02, 0.05, 0.03, 0.15, 0.08, 0.12, 0.25, 0.04, 0.07, 0.12],
})

# ---- 10 Maharashtra cities: (distance from JNPT port km, route difficulty) ---
cities = {
    "Mumbai": (30, 1.00), "Thane": (45, 1.00), "Pune": (150, 1.05),
    "Nashik": (190, 1.10), "Aurangabad": (330, 1.15), "Kolhapur": (380, 1.20),
    "Solapur": (410, 1.20), "Nagpur": (850, 1.35), "Amravati": (750, 1.30), "Nanded": (620, 1.30),
}

# ---- 6 product categories: (fragility, avg order value INR) ------------------
products = {
    "Electronics": (1.5, 450000), "Textiles": (0.7, 200000), "Machinery Parts": (0.9, 600000),
    "Chemicals": (1.2, 300000), "Food Items": (1.4, 150000), "Furniture": (1.3, 250000),
}

modes = {"Road": 1.00, "Rail": 0.90, "Sea+Road": 1.15}

rows = []
ceiling = []
from scipy.stats import gamma as _gamma
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 365, N)), unit="D")
for i in range(N):
    s = suppliers.iloc[rng.integers(0, 10)]
    city = rng.choice(list(cities))
    prod = rng.choice(list(products))
    mode = rng.choice(list(modes), p=[0.55, 0.25, 0.20])
    dist, route_diff = cities[city]
    fragility, avg_val = products[prod]
    month = dates[i].month
    monsoon = int(month in (6, 7, 8, 9))
    value = int(max(30000, rng.normal(avg_val, avg_val * 0.35)))
    weight = int(max(50, rng.normal(1500, 700)))
    planned_days = int(7 + dist / 60 + (5 if mode == "Sea+Road" else 0))

    # Outcome probabilities = supplier baseline x context multipliers
    p_delay = min(0.95, s.p_delay * route_diff * modes[mode] * (1.35 if monsoon else 1.0))
    p_damage = min(0.90, s.p_damage * fragility * (1.25 if monsoon else 1.0) * (1 + dist / 2500))
    p_return = min(0.80, s.p_return * (1 + 0.5 * (value > 500000)) * (1.15 if monsoon else 1.0))

    # --- theoretical best-case (Bayes) accuracy: what a model that knew the TRUE probabilities could reach
    _scale = 0.9 + 2.5 * s.p_delay + 1.2 * monsoon + dist / 600
    p_big = 1 - _gamma.cdf(5.5, 2.0, scale=_scale)                 # P(delay > 5 days | delayed)
    p_not_high = (1 - p_delay) * (1 - p_return) + p_delay * (1 - p_damage) * (1 - p_return) * (1 - p_big)
    p_low = (1 - p_delay) * (1 - p_damage) * (1 - p_return)
    ceiling.append(max(p_low, 1 - p_not_high, 1 - p_low - (1 - p_not_high)))
    delayed = rng.random() < p_delay
    damaged = rng.random() < p_damage
    returned = rng.random() < p_return
    # delay length depends on supplier slowness, monsoon and distance (not pure luck)
    scale = 0.9 + 2.5 * s.p_delay + 1.2 * monsoon + dist / 600
    delay_days = int(np.clip(round(rng.gamma(2.0, scale)), 1, 15)) if delayed else 0

    rows.append(dict(
        shipment_id=f"SH{i+1:04d}", ship_date=dates[i].date(), supplier_id=s.supplier_id,
        supplier_name=s.supplier_name, city=city, product=prod, transport_mode=mode,
        order_value=value, weight_kg=weight, distance_km=dist, planned_days=planned_days,
        month=month, monsoon=monsoon, delayed=int(delayed), delay_days=delay_days,
        damaged=int(damaged), returned=int(returned),
    ))

df = pd.DataFrame(rows)

# ---- TARGET: shipment risk outcome (what the importer actually cares about) --
# High   = returned, OR damaged+delayed, OR delayed more than 5 days
# Medium = any delay or any damage
# Low    = clean shipment
def label(r):
    if r.returned or (r.damaged and r.delayed) or r.delay_days > 5:
        return "High"
    if r.delayed or r.damaged:
        return "Medium"
    return "Low"

df["risk_level"] = df.apply(label, axis=1)
df.to_csv("shipment_data.csv", index=False)
print(f"Saved shipment_data.csv  ({len(df)} rows)")
print(df["risk_level"].value_counts().to_string())

print(f"\nBayes-optimal accuracy ceiling for this data: {np.mean(ceiling):.1%}")
open("bayes_ceiling.txt", "w").write(f"{np.mean(ceiling):.4f}")
