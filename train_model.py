"""
train_model.py  --  DESIGN THINKING STAGES 2-5 applied to the ML model
-----------------------------------------------------------------------
DEFINE    : missing a High-risk shipment costs far more than a false alarm
            -> class_weight='balanced' and we track RECALL of the High class.
IDEATE    : we do not jump to one model; we compare 4 ideas
            (rule-only trust score, Logistic Regression, Random Forest, Gradient Boosting).
PROTOTYPE : final system = ML model + Business Rules Layer (hard thresholds
            override the ML) + probabilities + reasons (see risk_predictor.py).
TEST      : 5-fold stratified cross-validation, hold-out test, confusion matrix,
            permutation feature importance, ML vs ML+Rules comparison.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score, roc_auc_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from risk_rules import apply_business_rules

SEED = 42
CLASSES = ["Low", "Medium", "High"]

CAT = ["product", "transport_mode"]
# Feature choices (Empathize -> what the importer worries about), pruned for redundancy:
#  * city dropped        : 1-to-1 with distance_km
#  * month dropped       : monsoon flag captures the seasonal effect
#  * trust_score dropped : it is a formula of the 3 prior rates (kept for rules + dashboard)
#  * supplier_id dropped : rates describe the supplier, so brand-new suppliers also work
NUM = ["order_value", "distance_km", "monsoon",
       "prior_delay_rate", "prior_damage_rate", "prior_return_rate",
       "recent_bad_rate", "prior_delay_days"]
FEATURES = CAT + NUM


# ------------------------------------------------------------------ helpers
def preprocessor(scale=False):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                              ("num", num, NUM)])


def build_models():
    return {
        "Trust-score rules only": None,   # baseline, no ML
        "Logistic Regression": Pipeline([
            ("prep", preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1))]),
        "Random Forest": Pipeline([
            ("prep", preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=4, max_features="sqrt",
                                           class_weight="balanced", random_state=SEED, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([
            ("prep", preprocessor()),
            ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                   class_weight="balanced", random_state=SEED))]),
    }


def rule_baseline(df):
    """Idea #1: no ML, just supplier trust-score bands."""
    return np.where(df.trust_score < 50, "High", np.where(df.trust_score < 70, "Medium", "Low"))


def metrics(y_true, y_pred, high_score=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "high_recall": recall_score(y_true, y_pred, labels=["High"], average=None)[0],
        "high_precision": precision_score(y_true, y_pred, labels=["High"], average=None, zero_division=0)[0],
    }
    if high_score is not None:
        m["auc_high"] = roc_auc_score(np.asarray(y_true) == "High", high_score)
    return m


TARGET_RECALL = 0.70   # DEFINE: "catch at least 70% of High-risk shipments"


def p_high(model, Xf):
    return model.predict_proba(Xf)[:, list(model.classes_).index("High")]


def decide(model, Xdf, threshold):
    """ML label, but escalate to High whenever P(High) >= alert threshold."""
    proba = model.predict_proba(Xdf[FEATURES])
    cls = list(model.classes_)
    pred = np.array(cls)[proba.argmax(axis=1)]
    return np.where(proba[:, cls.index("High")] >= threshold, "High", pred)


def pick_threshold(y_true, score, target=TARGET_RECALL):
    """Highest threshold that still reaches the target High-risk recall."""
    yt = np.asarray(y_true) == "High"
    best = 0.05
    for th in np.arange(0.05, 0.95, 0.01):
        if (score[yt] >= th).mean() >= target:
            best = th
    return round(float(best), 2)


# ------------------------------------------------------------------ load
df = pd.read_csv("shipment_features.csv")
X, y = df[list(dict.fromkeys(FEATURES + ["high_value", "trust_score"]))], df["risk_level"]
ceiling = float(open("bayes_ceiling.txt").read())
print(f"Rows: {len(df)} | class balance:\n{y.value_counts(normalize=True).round(3).to_string()}")
print(f"Best possible accuracy on this synthetic data (Bayes ceiling): {ceiling:.1%}\n")

# Hold-out is touched ONCE at the very end. Everything else (comparison, model choice,
# alert threshold) uses only the 80% training part.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ------------------------------------------------------------------ IDEATE + TEST: compare 4 ideas (5-fold CV on train)
results, oof_scores = [], {}
for name, model in build_models().items():
    if model is None:
        pred = rule_baseline(X_tr)
        score = 100 - X_tr.trust_score.values          # lower trust = higher risk
    else:
        proba = cross_val_predict(model, X_tr[FEATURES], y_tr, cv=skf, method="predict_proba")
        classes = sorted(y_tr.unique())
        pred = np.array(classes)[proba.argmax(axis=1)]
        score = proba[:, classes.index("High")]
    oof_scores[name] = score
    results.append({"model": name, **metrics(y_tr, pred, score)})
    results.append({"model": name + " + Rules", **metrics(y_tr, apply_business_rules(X_tr, pred), score)})

comp = pd.DataFrame(results).round(3)
comp.to_csv("model_comparison.csv", index=False)
print("=== IDEATE: MODEL COMPARISON (5-fold CV on training data) ===")
print(comp.to_string(index=False), "\n")

# ------------------------------------------------------------------ choose model by how well it RANKS High-risk shipments
ml_rows = comp[(~comp.model.str.contains("Rules|rules"))]
best_name = ml_rows.sort_values("auc_high", ascending=False).iloc[0].model
print(f"Selected model (best High-risk AUC): {best_name}")

# ------------------------------------------------------------------ DEFINE: tune alert threshold for target recall (CV, train only)
threshold = pick_threshold(y_tr, oof_scores[best_name])
# Trade-off table: the business chooses the dial (catch more risky shipments vs. fewer false alarms)
_yt = (y_tr == "High").values; _sc = oof_scores[best_name]
tradeoff = pd.DataFrame([{
    "alert_threshold": round(th, 2),
    "high_risk_caught_%": round(100 * (_sc[_yt] >= th).mean(), 1),
    "alerts_that_were_right_%": round(100 * _yt[_sc >= th].mean(), 1) if (_sc >= th).any() else np.nan,
    "shipments_flagged_%": round(100 * (_sc >= th).mean(), 1),
} for th in np.arange(0.15, 0.56, 0.05)])
tradeoff.to_csv("threshold_tradeoff.csv", index=False)
print("=== THRESHOLD TRADE-OFF (5-fold CV, training data) ===")
print(tradeoff.to_string(index=False), "\n")
print(f"Alert threshold: flag High when P(High) >= {threshold}  (target recall {TARGET_RECALL:.0%})\n")

# ------------------------------------------------------------------ TEST: one-time hold-out evaluation
model_eval = build_models()[best_name].fit(X_tr[FEATURES], y_tr)
hs = p_high(model_eval, X_te[FEATURES])
pred_plain = model_eval.predict(X_te[FEATURES])                       # default argmax
pred_ml = decide(model_eval, X_te, threshold)                          # + alert threshold
pred_sys = apply_business_rules(X_te, pred_ml)                         # + business rules  = full system

hold = {
    "Plain ML (argmax)": metrics(y_te, pred_plain, hs),
    "ML + alert threshold": metrics(y_te, pred_ml, hs),
    "Full system (ML + threshold + rules)": metrics(y_te, pred_sys, hs),
    "Trust-score rules only": metrics(y_te, rule_baseline(X_te), 100 - X_te.trust_score.values),
}
hold_df = pd.DataFrame(hold).T.round(3)
hold_df.to_csv("holdout_results.csv")
print("=== TEST: HOLD-OUT (100 unseen shipments) ===")
print(hold_df.to_string(), "\n")
print("Full system classification report:")
print(classification_report(y_te, pred_sys, labels=CLASSES, zero_division=0))

flag = pred_sys == "High"
base = (y_te == "High").mean()
lift = (y_te[flag] == "High").mean() / base if flag.sum() else float("nan")
print(f"Flagged {flag.sum()} of {len(y_te)} shipments as High. "
      f"Of these, {(y_te[flag] == 'High').mean():.0%} were truly High vs {base:.0%} base rate -> lift {lift:.2f}x\n")

# confusion matrix plot
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, p, ttl in zip(ax, [pred_plain, pred_sys], ["Plain ML", "Full system (ML + alert threshold + rules)"]):
    ConfusionMatrixDisplay(confusion_matrix(y_te, p, labels=CLASSES), display_labels=CLASSES).plot(
        ax=a, cmap="Blues", colorbar=False)
    a.set_title(ttl, fontsize=10)
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# permutation importance = "why does the model think so?"
pi = permutation_importance(model_eval, X_te[FEATURES], y_te, n_repeats=20, random_state=SEED,
                            scoring="roc_auc_ovr", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
plt.figure(figsize=(8, 6))
imp.tail(12).plot.barh(color="#1f77b4")
plt.title("What drives shipment risk? (permutation importance)")
plt.xlabel("Drop in ROC-AUC when feature is shuffled"); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150); plt.close()
imp.sort_values(ascending=False).round(4).to_csv("feature_importance.csv", header=["importance"])
print("Top risk drivers:\n", imp.sort_values(ascending=False).head(6).round(4).to_string(), "\n")

# model comparison plot
plot_df = comp[~comp.model.str.contains("Rules|rules")].set_index("model")[["auc_high", "high_recall", "macro_f1"]]
plot_df.plot.bar(figsize=(9, 5), rot=15)
plt.title("Ideate: which idea ranks risky shipments best? (5-fold CV)"); plt.ylabel("score"); plt.ylim(0, 1)
plt.tight_layout(); plt.savefig("model_comparison.png", dpi=150); plt.close()

# ------------------------------------------------------------------ save production model (trained on ALL data)
final = build_models()[best_name].fit(X[FEATURES], y)
joblib.dump(final, "risk_model.pkl")
meta = {
    "best_model": best_name, "features": FEATURES, "cat": CAT, "num": NUM, "classes": CLASSES,
    "alert_threshold": threshold, "target_recall": TARGET_RECALL, "bayes_ceiling_accuracy": ceiling,
    "baseline_num": {c: float(df[c].median()) for c in NUM},
    "baseline_cat": {c: df[c].mode()[0] for c in CAT},
    "cv_results": comp.to_dict(orient="records"),
    "holdout": hold_df.reset_index().rename(columns={"index": "system"}).to_dict(orient="records"),
    "lift_high": float(lift),
}
joblib.dump(meta, "model_meta.pkl")
with open("metrics.json", "w") as f:
    json.dump(meta, f, indent=2, default=float)
print("Saved: risk_model.pkl, model_meta.pkl, metrics.json, holdout_results.csv, model_comparison.csv,")
print("       confusion_matrix.png, feature_importance.png, model_comparison.png")
