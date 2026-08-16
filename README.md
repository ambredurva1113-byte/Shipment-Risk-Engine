#Shipment Trust & Risk Intelligence System
**Supplier Trust Scoring & Shipment Risk Prediction for Import-Export Businesses**

Built for businesses that need to know — which supplier to trust and which shipment will go wrong before it does.

---

## Problem Statement
Most import businesses find out about bad suppliers after the damage is done.
This system catches them early —

**Supplier History → Risk Scoring → Trust Score → Decision**

---

##Description  

| Description |
|---|---|
| Trust Score Engine | Scores every supplier 0–100 based on delay, damage & return history |
| Risk Predictor | Classifies shipments as Low / Medium / High risk |
| Business Rules Layer | Overrides ML when hard thresholds are crossed |
| Region Analysis | City-wise risk breakdown across Maharashtra |

---

## Tech Stack
- **Language:** Python 3.12
- **Dashboard:** Streamlit
- **ML:** Scikit-learn (Random Forest)
- **Charts:** Plotly
- **Data:** Pandas + CSV

---

## Dataset
- 500 shipments
- 10 suppliers with unique risk profiles
- 10 Maharashtra cities
- 6 product categories

---

## Setup
```bash
pip install -r requirements.txt
python setup.py
streamlit run app.py
```

---

## Folder Structure
```
ShipmentRiskEngine/
├── generate_data.py    ← synthetic dataset creation
├── trust_engine.py     ← trust score + feature engineering
├── train_model.py      ← Random Forest training
├── app.py              ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

**Developed by Durva Sagar Ambre · TY Project · 2026**

---
