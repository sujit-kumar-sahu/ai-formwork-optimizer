
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Smart Formwork Optimizer", layout="wide")

st.title("🏗 AI-Driven Smart Formwork Optimization Platform")

st.markdown("## 📌 Project Inputs")

col1, col2 = st.columns(2)

with col1:
    total_columns = st.number_input("Total Number of Columns", min_value=1, value=100)
    project_days = st.number_input("Project Duration (days)", min_value=1, value=20)
    design_strength = st.number_input("Design Strength (MPa)", min_value=10.0, value=25.0)
    required_percent = st.slider("Required Strength Percentage", 0.5, 0.9, 0.7)
    formwork_cost_per_set = st.number_input("Formwork Cost per Set (₹)", value=15000)

with col2:
    cement = st.number_input("Cement Content (kg/m³)", value=400)
    wc_ratio = st.number_input("Water-Cement Ratio", value=0.42)
    admixture = st.number_input("Admixture %", value=1.2)
    temperature = st.number_input("Ambient Temperature (°C)", value=30)
    carbon_per_set = st.number_input("Carbon per Formwork Set (kg CO2)", value=50)

run = st.button("🚀 Run Full Optimization")

# =====================
# Synthetic ML Dataset
# =====================
np.random.seed(42)
data_size = 400

data = pd.DataFrame({
    "cement_content": np.random.uniform(300, 450, data_size),
    "water_cement_ratio": np.random.uniform(0.35, 0.55, data_size),
    "admixture_percent": np.random.uniform(0, 2, data_size),
    "temperature": np.random.uniform(15, 40, data_size),
    "age_days": np.random.uniform(0.5, 7, data_size)
})

data["strength_MPa"] = (
    0.08 * data["cement_content"]
    - 40 * data["water_cement_ratio"]
    + 3 * data["admixture_percent"]
    + 0.6 * data["temperature"]
    + 5 * np.log(data["age_days"] + 1)
)

X = data[["cement_content","water_cement_ratio","admixture_percent","temperature","age_days"]]
y = data["strength_MPa"]

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X, y)

# =====================
# RUN OPTIMIZATION
# =====================
if run:

    required_strength = required_percent * design_strength
    predicted_time = None
    days = np.arange(0.5, 7, 0.1)
    strengths = []

    for day in days:
        input_data = pd.DataFrame([{
            "cement_content": cement,
            "water_cement_ratio": wc_ratio,
            "admixture_percent": admixture,
            "temperature": temperature,
            "age_days": day
        }])
        strength = model.predict(input_data)[0]
        strengths.append(strength)

        if predicted_time is None and strength >= required_strength:
            predicted_time = day

    daily_casting = total_columns / project_days
    minimum_formwork = daily_casting * predicted_time
    reduction_percent = (1 - minimum_formwork/total_columns) * 100

    traditional_cost = total_columns * formwork_cost_per_set
    optimized_cost = minimum_formwork * formwork_cost_per_set
    cost_savings = traditional_cost - optimized_cost

    carbon_savings = (total_columns - minimum_formwork) * carbon_per_set

    # =====================
    # DISPLAY RESULTS
    # =====================
    st.markdown("---")
    st.header("📊 Optimization Dashboard")

    colA, colB, colC = st.columns(3)

    colA.metric("Stripping Time (days)", f"{predicted_time:.2f}")
    colB.metric("Minimum Formwork Sets", f"{minimum_formwork:.2f}")
    colC.metric("Inventory Reduction (%)", f"{reduction_percent:.2f}")

    st.markdown("---")
    st.subheader("💰 Financial Impact")

    st.write(f"Traditional Cost: ₹{traditional_cost:,.0f}")
    st.write(f"Optimized Cost: ₹{optimized_cost:,.0f}")
    st.success(f"Cost Savings: ₹{cost_savings:,.0f}")

    st.markdown("---")
    st.subheader("🌱 Sustainability Impact")

    st.write(f"Estimated Carbon Reduction: {carbon_savings:.2f} kg CO₂")

    # =====================
    # Strength Curve Plot
    # =====================
    st.markdown("---")
    st.subheader("📈 Strength vs Time Curve")

    fig, ax = plt.subplots()
    ax.plot(days, strengths)
    ax.axhline(required_strength, linestyle='--')
    ax.set_xlabel("Age (days)")
    ax.set_ylabel("Strength (MPa)")
    ax.set_title("Concrete Strength Development Curve")
    st.pyplot(fig)

    # =====================
    # Generate BoQ Table
    # =====================
    st.markdown("---")
    st.subheader("📦 Optimized BoQ Summary")

    boq = pd.DataFrame({
        "Item": ["Formwork Sets Required"],
        "Quantity": [round(minimum_formwork)],
        "Unit Cost (₹)": [formwork_cost_per_set],
        "Total Cost (₹)": [round(optimized_cost)]
    })

    st.dataframe(boq)

    csv = boq.to_csv(index=False).encode('utf-8')
    st.download_button("Download Optimized BoQ", csv, "Optimized_BoQ.csv", "text/csv")

    st.markdown("---")
    st.subheader("🚀 Future Enhancements")

    st.write("• Digital Twin Integration")
    st.write("• IoT-based real-time curing sensors")
    st.write("• Reinforcement Learning dynamic scheduler")
    st.write("• Cloud deployment for enterprise use")

    st.success("Full AI Optimization Completed Successfully!")
