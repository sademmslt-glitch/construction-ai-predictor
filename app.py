import streamlit as st
import pandas as pd
import joblib

# Load models
reg_model = joblib.load("regression_model.pkl")
clf_model = joblib.load("classification_model.pkl")

st.set_page_config(page_title="AI Construction Predictor", page_icon="🏗️")

st.title("🏗️ AI-based Construction Project Prediction System")

st.write("""
This web app predicts:
- Final Project Cost (SAR)
- Probability of Delay (%)

Based on real planning parameters and realistic simulated company data.
""")

# ---- User Inputs ----

project_type = st.selectbox(
    "Project Type",
    [
        "Residential Building",
        "Non-Residential Building",
        "Electrical Works",
        "Network & Communication",
        "Finishing & Tiling",
        "Renovation"
    ]
)

project_size = st.number_input("Project Size (m²)", min_value=50, max_value=8000, value=500)
workers = st.number_input("Number of Workers", min_value=1, max_value=200, value=20)
budget = st.number_input("Estimated Budget (SAR)", min_value=50000, max_value=4000000, value=300000)
duration = st.number_input("Expected Duration (months)", min_value=1, max_value=36, value=8)

# ---- Feature Engineering ----
cost_pressure = budget / project_size
worker_density = workers / project_size

# ---- Encoding project type ----
project_types = [
    "Residential Building",
    "Non-Residential Building",
    "Electrical Works",
    "Network & Communication",
    "Finishing & Tiling",
    "Renovation"
]

encoded_project = [1 if project_type == p else 0 for p in project_types]

# ---- Input DataFrame ----
input_data = pd.DataFrame([[
    project_size,
    workers,
    budget,
    duration,
    cost_pressure,
    worker_density,
    *encoded_project
]], columns=[
    "Project_Size",
    "Num_Workers",
    "Budget",
    "Duration",
    "Cost_Pressure",
    "Worker_Density",
    "Project_Type_Residential Building",
    "Project_Type_Non-Residential Building",
    "Project_Type_Electrical Works",
    "Project_Type_Network & Communication",
    "Project_Type_Finishing & Tiling",
    "Project_Type_Renovation"
])

# ---- Prediction ----
if st.button("Predict Project Outcome"):
    predicted_cost = reg_model.predict(input_data)[0]
    delay_probability = clf_model.predict_proba(input_data)[0][1] * 100

    st.subheader("🔍 Prediction Results")
    st.metric("Predicted Final Cost (SAR)", f"{predicted_cost:,.0f}")
    st.metric("Delay Probability (%)", f"{delay_probability:.1f}%")

    if delay_probability > 60:
        st.warning("⚠️ High risk of project delay. Consider increasing workforce or adjusting schedule.")
    elif delay_probability > 40:
        st.info("ℹ️ Medium delay risk. Monitor project resources carefully.")
    else:
        st.success("✅ Low delay risk. Project plan looks stable.")
