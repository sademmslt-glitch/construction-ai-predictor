import streamlit as st
import pandas as pd
import joblib

# ✅ Load trained models
reg_model = joblib.load("regression_model.pkl")
clf_model = joblib.load("classification_model.pkl")

st.set_page_config(page_title="AI Construction Predictor", page_icon="🏗️")

st.title("🏗️ AI-based Construction Project Prediction System")

st.write("""
This web app predicts:
- Final Project Cost (SAR)
- Probability of Delay (%)

Based on realistic simulated company data.
It also provides smart recommendations to help decision-making.
""")

# -------- User Inputs --------

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

# -------- Feature Engineering --------
cost_pressure = budget / project_size
worker_density = workers / project_size

# -------- Build input row as dict --------
input_dict = {
    "Project_Size": project_size,
    "Num_Workers": workers,
    "Budget": budget,
    "Duration": duration,
    "Cost_Pressure": cost_pressure,
    "Worker_Density": worker_density,
    "Project_Type_Residential Building": 1 if project_type == "Residential Building" else 0,
    "Project_Type_Non-Residential Building": 1 if project_type == "Non-Residential Building" else 0,
    "Project_Type_Electrical Works": 1 if project_type == "Electrical Works" else 0,
    "Project_Type_Network & Communication": 1 if project_type == "Network & Communication" else 0,
    "Project_Type_Finishing & Tiling": 1 if project_type == "Finishing & Tiling" else 0,
    "Project_Type_Renovation": 1 if project_type == "Renovation" else 0,
}

input_data = pd.DataFrame([input_dict])

# ✅ Align features for EACH model separately
input_data_reg = input_data.reindex(
    columns=reg_model.feature_names_in_, fill_value=0
)

input_data_clf = input_data.reindex(
    columns=clf_model.feature_names_in_, fill_value=0
)

# -------- Prediction --------
if st.button("Predict Project Outcome"):
    predicted_cost = reg_model.predict(input_data_reg)[0]
    delay_probability = clf_model.predict_proba(input_data_clf)[0][1] * 100

    st.subheader("🔍 Prediction Results")
    st.metric("Predicted Final Cost (SAR)", f"{predicted_cost:,.0f}")
    st.metric("Delay Probability (%)", f"{delay_probability:.1f}%")

    # -------- Risk Level --------
    if delay_probability > 60:
        st.warning("🔴 High Delay Risk")
    elif delay_probability > 40:
        st.info("🟡 Medium Delay Risk")
    else:
        st.success("🟢 Low Delay Risk")

    # -------- Smart Recommendations --------
    st.subheader("🛠️ Smart Recommendations")

    if delay_probability > 60:
        st.write("🔴 **High Risk Project**")
        st.write("- Increase number of workers.")
        st.write("- Add contingency budget (5–10%).")
        st.write("- Review project schedule carefully.")
        st.write("- Strengthen supervision and follow-up.")
    
    elif delay_probability > 40:
        st.write("🟡 **Medium Risk Project**")
        st.write("- Slightly increase workforce if possible.")
        st.write("- Monitor material delivery and expenses closely.")
        st.write("- Prepare a small contingency budget.")
    
    else:
        st.write("🟢 **Low Risk Project**")
        st.write("- Current plan is stable.")
        st.write("- Maintain the same resources and schedule.")
        st.write("- Continue regular monitoring.")
