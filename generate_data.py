import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

suppliers = [
    "Shenzhen Traders", "Mumbai Port Co", "Delhi Import Hub",
    "Guangzhou Exports", "Chennai Logistics", "Pune Supply Chain",
    "Kolkata Goods Ltd", "Nagpur Freight", "Aurangabad Traders",
    "Nashik Distributors"
]

categories = ["Electronics", "Textiles", "Machinery", "Chemicals", "Pharmaceuticals", "Food & Agri"]

maharashtra_cities = [
    "Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad",
    "Solapur", "Kolhapur", "Thane", "Amravati", "Nanded"
]

weather_options = ["Low", "Medium", "High"]
months = list(range(1, 13))

# Supplier profiles — some are naturally bad, some are good
# This makes the data realistic
supplier_profiles = {
    "Shenzhen Traders":    {"delay_bias": 5,  "damage_bias": 12, "return_bias": 8},
    "Mumbai Port Co":      {"delay_bias": 2,  "damage_bias": 4,  "return_bias": 3},
    "Delhi Import Hub":    {"delay_bias": 3,  "damage_bias": 6,  "return_bias": 5},
    "Guangzhou Exports":   {"delay_bias": 8,  "damage_bias": 15, "return_bias": 10},
    "Chennai Logistics":   {"delay_bias": 1,  "damage_bias": 3,  "return_bias": 2},
    "Pune Supply Chain":   {"delay_bias": 2,  "damage_bias": 5,  "return_bias": 4},
    "Kolkata Goods Ltd":   {"delay_bias": 6,  "damage_bias": 9,  "return_bias": 7},
    "Nagpur Freight":      {"delay_bias": 4,  "damage_bias": 7,  "return_bias": 5},
    "Aurangabad Traders":  {"delay_bias": 3,  "damage_bias": 5,  "return_bias": 4},
    "Nashik Distributors": {"delay_bias": 1,  "damage_bias": 2,  "return_bias": 2},
}

records = []

for i in range(1, 501):
    supplier = random.choice(suppliers)
    profile = supplier_profiles[supplier]

    expected_days = random.randint(7, 30)
    delay_days = max(0, int(np.random.normal(profile["delay_bias"], 3)))
    shipping_days = expected_days + delay_days

    damage_rate = round(max(0, np.random.normal(profile["damage_bias"], 3)), 2)
    return_rate = round(max(0, np.random.normal(profile["return_bias"], 2)), 2)

    import_cost = round(random.uniform(5000, 200000), 2)
    quantity = random.randint(50, 5000)

    weather_risk = random.choice(weather_options)
    port_congestion = round(random.uniform(1, 10), 1)
    payment_delay = max(0, int(np.random.normal(5, 4)))
    month = random.choice(months)
    region = random.choice(maharashtra_cities)
    category = random.choice(categories)

    # Risk label — using business rules, not random
    risk_score = 0

    if delay_days > 7:
        risk_score += 2
    elif delay_days > 3:
        risk_score += 1

    if damage_rate > 15:
        risk_score += 2
    elif damage_rate > 8:
        risk_score += 1

    if return_rate > 10:
        risk_score += 2
    elif return_rate > 5:
        risk_score += 1

    if weather_risk == "High":
        risk_score += 1

    if port_congestion > 7:
        risk_score += 1

    if risk_score >= 5:
        risk_label = "High"
    elif risk_score >= 2:
        risk_label = "Medium"
    else:
        risk_label = "Low"

    records.append({
        "Shipment_ID": f"SHP{str(i).zfill(4)}",
        "Supplier_Name": supplier,
        "Product_Category": category,
        "Import_Cost": import_cost,
        "Quantity": quantity,
        "Shipping_Days": shipping_days,
        "Expected_Days": expected_days,
        "Delay_Days": delay_days,
        "Damage_Rate": damage_rate,
        "Return_Rate": return_rate,
        "Region": region,
        "Month": month,
        "Weather_Risk": weather_risk,
        "Port_Congestion_Score": port_congestion,
        "Payment_Delay": payment_delay,
        "Risk_Label": risk_label
    })

df = pd.DataFrame(records)
df.to_csv("shipment_data.csv", index=False)
print(f"Dataset created: {len(df)} records")
print(df["Risk_Label"].value_counts())
print(df.head(3))
