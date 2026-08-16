# Shipment-Risk-Engine

A decision-support system for import-export businesses that scores supplier reliability and predicts shipment risk — built with Python and Streamlit.

What it does :-
Assigns a Trust Score (0–100) to every supplier based on delay history, damage rate, and return rate.
Predicts shipment risk level — Low / Medium / High — using a Random Forest classifier.
Applies a business rules layer on top of ML to override predictions when hard thresholds are crossed.
Region-wise risk breakdown across Maharashtra cities.

How it works:-
generate_data.py — creates 500 realistic shipments with supplier profiles
trust_engine.py — computes Delay Ratio, Loss Score, and Supplier Trust Score
train_model.py — trains Random Forest (82% accuracy)
app.py — 4-page Streamlit dashboard
Tech Stack

Python · Pandas · Scikit-learn · Streamlit · Plotly
