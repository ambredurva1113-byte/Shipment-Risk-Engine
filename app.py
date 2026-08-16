import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shipment Trust Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("shipment_features.csv")
    trust_df = pd.read_csv("supplier_trust.csv")
    return df, trust_df

@st.cache_resource
def load_model():
    with open("risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta

df, trust_df = load_data()
model, meta = load_model()

# ── sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.title("🚢 Shipment Trust System")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🏭 Supplier Trust",
    "🔍 Risk Checker",
    "🗺️ Region Analysis"
])

# ── risk color helper ─────────────────────────────────────────────────────────
RISK_COLOR = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
TRUST_COLOR = lambda score: "#22c55e" if score >= 70 else ("#f59e0b" if score >= 45 else "#ef4444")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Shipment Overview Dashboard")
    st.markdown("High-level snapshot of all shipments and risk distribution.")
    st.markdown("---")

    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    high_risk = (df["Risk_Label_Final"] == "High").sum()
    avg_delay = df["Delay_Days"].mean()
    avg_damage = df["Damage_Rate"].mean()

    col1.metric("Total Shipments", total)
    col2.metric("High Risk Shipments", high_risk, delta=f"{high_risk/total:.0%} of total", delta_color="inverse")
    col3.metric("Avg Delay (days)", f"{avg_delay:.1f}")
    col4.metric("Avg Damage Rate", f"{avg_damage:.1f}%")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Distribution")
        risk_counts = df["Risk_Label_Final"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig = px.pie(
            risk_counts, names="Risk", values="Count",
            color="Risk",
            color_discrete_map=RISK_COLOR,
            hole=0.4
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Shipments by Category")
        cat_risk = df.groupby(["Product_Category", "Risk_Label_Final"]).size().reset_index(name="Count")
        fig2 = px.bar(
            cat_risk, x="Product_Category", y="Count",
            color="Risk_Label_Final",
            color_discrete_map=RISK_COLOR,
            barmode="stack"
        )
        fig2.update_layout(showlegend=True, xaxis_title="", yaxis_title="Shipments")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Delay Trend by Month")
    monthly = df.groupby("Month")["Delay_Days"].mean().reset_index()
    monthly.columns = ["Month", "Avg_Delay"]
    fig3 = px.line(monthly, x="Month", y="Avg_Delay", markers=True)
    fig3.update_traces(line_color="#6366f1", line_width=2.5)
    fig3.update_layout(yaxis_title="Avg Delay (days)", xaxis_title="Month")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("Worst Suppliers (by High Risk shipments)")
    worst = df[df["Risk_Label_Final"] == "High"]["Supplier_Name"].value_counts().reset_index()
    worst.columns = ["Supplier", "High Risk Count"]
    fig4 = px.bar(worst, x="High Risk Count", y="Supplier", orientation="h", color="High Risk Count",
                  color_continuous_scale="Reds")
    fig4.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SUPPLIER TRUST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Supplier Trust":
    st.title("🏭 Supplier Trust Score Analysis")
    st.markdown("Each supplier is scored 0–100 based on delay history, damage rate, return rate & consistency.")
    st.markdown("---")

    trust_sorted = trust_df.sort_values("Trust_Score", ascending=False).reset_index(drop=True)

    # Big trust score cards
    st.subheader("Supplier Rankings")
    for _, row in trust_sorted.iterrows():
        score = row["Trust_Score"]
        color = TRUST_COLOR(score)
        rec = row["Recommendation"]

        with st.container():
            cols = st.columns([3, 1, 1, 1, 4])
            cols[0].markdown(f"**{row['Supplier_Name']}**")
            cols[1].markdown(f"Delay: `{row['avg_delay']:.1f}d`")
            cols[2].markdown(f"Damage: `{row['avg_damage']:.1f}%`")
            cols[3].markdown(
                f"<span style='color:{color}; font-size:20px; font-weight:bold'>{score}/100</span>",
                unsafe_allow_html=True
            )
            cols[4].markdown(rec)
        st.markdown("---")

    st.subheader("Trust Score Bar Chart")
    fig = px.bar(
        trust_sorted, x="Supplier_Name", y="Trust_Score",
        color="Trust_Score",
        color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
        range_color=[0, 100],
        text="Trust_Score"
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(xaxis_title="", yaxis_range=[0, 110], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Drill-down
    st.markdown("---")
    st.subheader("Deep Dive — Select a Supplier")
    selected = st.selectbox("Choose supplier", trust_sorted["Supplier_Name"].tolist())

    s_data = df[df["Supplier_Name"] == selected]
    s_trust = trust_sorted[trust_sorted["Supplier_Name"] == selected].iloc[0]

    col1, col2 = st.columns(2)
    col1.metric("Trust Score", f"{s_trust['Trust_Score']}/100")
    col1.metric("Total Shipments", int(s_trust["total_shipments"]))
    col2.metric("Avg Delay Days", f"{s_trust['avg_delay']:.1f}")
    col2.metric("On-Time Rate", f"{s_trust['on_time_rate']:.0%}")

    st.markdown(f"**Recommendation:** {s_trust['Recommendation']}")

    risk_breakdown = s_data["Risk_Label_Final"].value_counts().reset_index()
    risk_breakdown.columns = ["Risk", "Count"]
    fig2 = px.pie(risk_breakdown, names="Risk", values="Count", color="Risk",
                  color_discrete_map=RISK_COLOR, title="Risk Breakdown")
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK CHECKER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Risk Checker":
    st.title("🔍 Shipment Risk Checker")
    st.markdown("Look up an existing shipment OR manually enter details to predict risk.")
    st.markdown("---")

    mode = st.radio("Mode", ["Look up Shipment ID", "Enter Shipment Manually"])

    if mode == "Look up Shipment ID":
        shipment_id = st.selectbox("Select Shipment ID", df["Shipment_ID"].tolist())
        row = df[df["Shipment_ID"] == shipment_id].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Supplier", row["Supplier_Name"])
        col2.metric("Delay Days", int(row["Delay_Days"]))
        col3.metric("Damage Rate", f"{row['Damage_Rate']}%")

        col4, col5, col6 = st.columns(3)
        col4.metric("Return Rate", f"{row['Return_Rate']}%")
        col5.metric("Port Congestion", row["Port_Congestion_Score"])
        col6.metric("Weather Risk", row["Weather_Risk"])

        risk = row["Risk_Label_Final"]
        color = RISK_COLOR[risk]
        st.markdown(
            f"<div style='background:{color}22; border-left:5px solid {color}; padding:16px; border-radius:8px; margin-top:12px'>"
            f"<h3 style='color:{color}; margin:0'>Risk Level: {risk}</h3>"
            f"<p style='margin:4px 0 0 0'>Loss Score: {row['Loss_Score']} | Delay Ratio: {row['Delay_Ratio']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Supplier trust for this shipment's supplier
        if row["Supplier_Name"] in trust_df["Supplier_Name"].values:
            t = trust_df[trust_df["Supplier_Name"] == row["Supplier_Name"]].iloc[0]
            st.markdown(f"\n**Supplier Trust Score:** {t['Trust_Score']}/100 — {t['Recommendation']}")

    else:
        col1, col2 = st.columns(2)
        with col1:
            delay_days = st.slider("Delay Days", 0, 20, 3)
            damage_rate = st.slider("Damage Rate (%)", 0.0, 30.0, 5.0)
            return_rate = st.slider("Return Rate (%)", 0.0, 25.0, 4.0)
            port_congestion = st.slider("Port Congestion Score (1-10)", 1.0, 10.0, 5.0)
        with col2:
            payment_delay = st.slider("Payment Delay (days)", 0, 30, 5)
            expected_days = st.slider("Expected Shipping Days", 5, 30, 14)
            import_cost = st.number_input("Import Cost (₹)", 5000, 200000, 50000)
            quantity = st.number_input("Quantity", 50, 5000, 500)
            weather_risk = st.selectbox("Weather Risk", ["Low", "Medium", "High"])

        if st.button("Predict Risk", type="primary"):
            delay_ratio = delay_days / expected_days if expected_days > 0 else 0
            loss_score = return_rate + damage_rate
            weather_encoded = meta["weather_map"][weather_risk]

            input_data = pd.DataFrame([{
                "Delay_Days": delay_days,
                "Delay_Ratio": delay_ratio,
                "Damage_Rate": damage_rate,
                "Return_Rate": return_rate,
                "Loss_Score": loss_score,
                "Port_Congestion_Score": port_congestion,
                "Payment_Delay": payment_delay,
                "Weather_Risk_Encoded": weather_encoded,
                "Import_Cost": import_cost,
                "Quantity": quantity
            }])

            pred = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            risk = meta["reverse_label"][pred]
            color = RISK_COLOR[risk]

            st.markdown(
                f"<div style='background:{color}22; border-left:5px solid {color}; padding:20px; border-radius:8px; margin-top:16px'>"
                f"<h2 style='color:{color}; margin:0'>Predicted Risk: {risk}</h2>"
                f"<p style='margin:8px 0 0 0'>Low: {proba[0]:.0%} | Medium: {proba[1]:.0%} | High: {proba[2]:.0%}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Explain the main driver
            reasons = []
            if delay_days > 7:
                reasons.append("Delay exceeds 7 days (major risk factor)")
            if damage_rate > 15:
                reasons.append("Damage rate above 15% threshold")
            if return_rate > 10:
                reasons.append("Return rate is high")
            if weather_risk == "High":
                reasons.append("High weather risk in transit region")

            if reasons:
                st.markdown("**Why this risk level?**")
                for r in reasons:
                    st.markdown(f"- {r}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REGION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Region Analysis":
    st.title("🗺️ Region-wise Risk Analysis")
    st.markdown("Maharashtra city-level breakdown of shipment risk and delays.")
    st.markdown("---")

    region_summary = df.groupby("Region").agg(
        Total_Shipments=("Shipment_ID", "count"),
        Avg_Delay=("Delay_Days", "mean"),
        Avg_Damage=("Damage_Rate", "mean"),
        High_Risk=("Risk_Label_Final", lambda x: (x == "High").sum())
    ).reset_index()

    region_summary["High_Risk_Rate"] = (region_summary["High_Risk"] / region_summary["Total_Shipments"] * 100).round(1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("High Risk Rate by City")
        fig = px.bar(
            region_summary.sort_values("High_Risk_Rate", ascending=False),
            x="Region", y="High_Risk_Rate",
            color="High_Risk_Rate",
            color_continuous_scale="Reds",
            text="High_Risk_Rate"
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(yaxis_title="High Risk %", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Avg Delay by City")
        fig2 = px.bar(
            region_summary.sort_values("Avg_Delay", ascending=False),
            x="Region", y="Avg_Delay",
            color="Avg_Delay",
            color_continuous_scale="Oranges",
            text="Avg_Delay"
        )
        fig2.update_traces(texttemplate="%{text:.1f}d", textposition="outside")
        fig2.update_layout(yaxis_title="Avg Delay (days)", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Detailed Table")
    st.dataframe(
        region_summary.sort_values("High_Risk_Rate", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.subheader("Category Risk by Region")
    cat_region = df.groupby(["Region", "Product_Category"])["Risk_Label_Final"].apply(
        lambda x: (x == "High").sum()
    ).reset_index()
    cat_region.columns = ["Region", "Category", "High_Risk_Count"]
    fig3 = px.density_heatmap(
        cat_region, x="Region", y="Category", z="High_Risk_Count",
        color_continuous_scale="YlOrRd"
    )
    st.plotly_chart(fig3, use_container_width=True)
