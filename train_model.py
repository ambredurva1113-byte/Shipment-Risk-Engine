import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle

def train_model():
    df = pd.read_csv("shipment_features.csv")

    # Features that make business sense — not random columns
    features = [
        "Delay_Days", "Delay_Ratio", "Damage_Rate",
        "Return_Rate", "Loss_Score", "Port_Congestion_Score",
        "Payment_Delay", "Weather_Risk_Encoded", "Import_Cost", "Quantity"
    ]

    # Encode weather risk
    weather_map = {"Low": 0, "Medium": 1, "High": 2}
    df["Weather_Risk_Encoded"] = df["Weather_Risk"].map(weather_map)

    # Encode target
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    df["Target"] = df["Risk_Label_Final"].map(label_map)

    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest — simple, explainable, good for viva
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,       # prevents overfitting
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Low", "Medium", "High"]))

    # Feature importance — great for viva explanation
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nTop Features:")
    print(importance_df.to_string(index=False))

    # Save model and mappings
    with open("risk_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save label encoder info too
    model_meta = {
        "features": features,
        "label_map": label_map,
        "weather_map": weather_map,
        "reverse_label": {0: "Low", 1: "Medium", 2: "High"}
    }
    with open("model_meta.pkl", "wb") as f:
        pickle.dump(model_meta, f)

    print("\nModel saved to risk_model.pkl")
    return model, model_meta, importance_df


if __name__ == "__main__":
    train_model()
