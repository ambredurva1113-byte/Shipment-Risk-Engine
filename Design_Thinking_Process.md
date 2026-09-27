# Design Thinking Process
## Shipment Trust & Risk Intelligence System

**Project by:** Durva Sagar Ambre
**Domain:** Import–Export / Supply Chain Risk Management
**Approach:** Human-Centered Design Thinking (5-Stage Model)

---

## 1. Empathize

**Goal:** Understand who is affected by unreliable shipments and suppliers, and how the problem actually shows up in their day-to-day work.

**Stakeholders identified:**
- **Import/export business owners** — place orders with suppliers across multiple cities and have no early signal on which supplier is likely to cause a problem.
- **Procurement/operations managers** — manually track supplier history in spreadsheets or memory, with no consistent scoring system.
- **Logistics/warehouse teams** — deal with the fallout (delayed inventory, damaged goods, returns) after a bad shipment has already happened.
- **Finance teams** — absorb the cost of returns, penalties, and rush replacement orders.

**Empathy findings (pain points):**
- Businesses discover a supplier is unreliable only **after** repeated delays, damage, or returns — the cost has already been incurred.
- Trust in a supplier is currently a "gut feeling" built over time, not a measurable, comparable score.
- There is no city/region-level visibility — a business can't easily see if certain regions (e.g. specific Maharashtra cities) are systematically riskier.
- Decisions on "should we order from this supplier again" are reactive, not predictive.

**Empathy method used:** Problem framing based on common import/export operational patterns — supplier delay history, damage/return rates, and regional variation — rather than a single anecdote, since these are the recurring, well-documented pain points in supply chain literature and practice.

---

## 2. Define

**Problem statement:**
> "Import/export businesses have no early-warning system to identify which suppliers and shipments are likely to fail (via delay, damage, or return) before the failure occurs, leading to reactive rather than preventive decision-making."

**Point-of-view (POV) statement:**
> A procurement manager at an import/export business **needs** a way to quantify supplier reliability and predict shipment risk **because** currently they only find out a supplier is untrustworthy after damage is done, which costs time, money, and customer trust.

**Defined success criteria for a solution:**
1. Convert scattered supplier history (delays, damages, returns) into a single, comparable **Trust Score (0–100)**.
2. Predict the **risk level of a new/incoming shipment** (Low / Medium / High) before it arrives.
3. Allow **hard business rules** to override the ML model when a threshold is non-negotiable (e.g. a supplier with a history of fraud should always be flagged, regardless of model confidence).
4. Present all of this in a way a non-technical operations person can use — not a raw model output.
5. Surface **regional patterns** (city-wise risk) so sourcing decisions can factor in geography.

---

## 3. Ideate

**Solutions considered:**

| Option | Description | Why chosen / rejected |
|---|---|---|
| Manual scoring checklist | Ops team fills a spreadsheet checklist per supplier | Rejected — still manual, inconsistent between people, doesn't scale, no predictive power |
| Pure rule-based system | Hard-coded thresholds (e.g. "if delay > 5 days, flag") | Rejected as the *sole* solution — too rigid, misses complex patterns across multiple weak signals, can't rank suppliers on a continuous scale |
| Pure ML model | A classifier predicts risk with no human-defined rules | Rejected as the *sole* solution — a pure black-box model could greenlight a supplier who technically fits patterns but has an obvious disqualifying issue (e.g. known fraud), which is risky for a business-critical decision |
| **Hybrid: ML model + Business Rules Layer + Trust Score Engine** | Random Forest model predicts shipment risk from historical + shipment features; a separate trust-scoring engine aggregates supplier history into a 0–100 score; a rules layer can override the ML output for hard thresholds | **Selected** — combines the pattern-recognition strength of ML with the safety and explainability of business rules; gives both a granular score (trust) and a categorical decision (risk level) |

**Key design decisions from ideation:**
- Separate the **Trust Score Engine** (supplier-level, historical) from the **Risk Predictor** (shipment-level, ML-based) — these answer two different questions ("who do I trust" vs. "will this specific shipment go wrong") and combining them into one number would hide useful information.
- Add a **Business Rules Layer** on top of the ML model so the system is not purely a black box — important for stakeholder trust in a business context where a wrong prediction has real financial cost.
- Present results through a **Streamlit dashboard** rather than a static report, so the tool can be used continuously as new shipments come in, not just as a one-time analysis.
- Include **region analysis** as a first-class feature (not an afterthought) since geography-linked risk was one of the clearest patterns from the Empathize stage.

---

## 4. Prototype

**What was built:**

| Component | Description |
|---|---|
| `generate_data.py` | Synthetic dataset generator — 500 shipments, 10 suppliers with distinct risk profiles, 10 Maharashtra cities, 6 product categories |
| `trust_engine.py` | Feature engineering + Trust Score computation (0–100) per supplier based on delay/damage/return history |
| `train_model.py` | Trains a Random Forest classifier on engineered features to predict shipment risk (Low / Medium / High) |
| `risk_model.pkl`, `model_meta.pkl` | Serialized trained model and metadata for reuse without retraining |
| Business Rules Layer | Overrides the ML prediction when a hard threshold is crossed (e.g. a supplier already flagged as high-risk historically) |
| `app.py` | Streamlit dashboard — the interactive front-end where a user enters/selects shipment details and sees the predicted risk, trust score, and regional breakdown, visualized with Plotly |

**Why this counts as a prototype (not a final product):**
- Uses synthetic data (500 shipments) rather than a live business's real transaction history — sufficient to validate the *approach*, not yet production data.
- The dashboard is a single-page interactive tool, not yet integrated into an existing ERP/procurement system.
- Model comparison (Random Forest vs. alternative algorithms) was tested to validate model choice before finalizing.

**Fast iteration approach:** Because the model, trust engine, and dashboard are separated into independent modules (`trust_engine.py`, `train_model.py`, `app.py`), each part could be prototyped and revised independently — e.g. retraining the model or changing scoring logic doesn't require rebuilding the dashboard.

---

## 5. Test

**What was evaluated:**
- **Model performance** — evaluated using a confusion matrix and comparison across multiple candidate models to select the best-performing classifier (Random Forest) for predicting Low/Medium/High risk.
- **Feature importance analysis** — used to confirm the model was relying on sensible, explainable signals (e.g. delay history, damage rate) rather than spurious correlations, which matters for stakeholder trust in a business setting.
- **Dashboard usability** — the Streamlit app was run end-to-end (`generate_data.py` → `setup.py` → `train_model.py` → `app.py`) to confirm a non-technical user could go from raw data to a risk decision without touching code.

**Feedback loop / what testing revealed:**
- Confirming feature importance validated the earlier Ideate-stage decision to keep a rules layer — some high-impact features are the same ones a rules-based override would target, showing the two approaches reinforce rather than duplicate each other.
- Region-wise breakdown (Test stage output) confirmed the Empathize-stage assumption that geography is a meaningful risk factor, justifying its place as a core dashboard feature rather than a nice-to-have.

**Next iteration (what testing suggests for a future version):**
- Replace synthetic data with a pilot batch of a real business's historical shipment records to validate the model on real-world noise and edge cases.
- Add a feedback mechanism where actual shipment outcomes (delivered on time / damaged / returned) are logged back into the system to periodically retrain and improve the model (a closed feedback loop).
- Extend regional analysis beyond Maharashtra as the business scales to other states/countries.

---

## Summary: Design Thinking → Product Mapping

| Design Thinking Stage | Project Artifact |
|---|---|
| Empathize | Problem framing around procurement managers' reactive decision-making |
| Define | Problem statement + POV statement (README "Problem Statement" section) |
| Ideate | Hybrid ML + Rules + Trust Score architecture decision |
| Prototype | `generate_data.py`, `trust_engine.py`, `train_model.py`, `app.py`, trained `.pkl` model |
| Test | Confusion matrix, model comparison, feature importance, regional validation |
