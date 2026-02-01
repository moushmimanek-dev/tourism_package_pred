import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="moushmim/tourism_package_pred_model", filename="best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer choosing a 'Wellness Tourism Package' from "Visit with Us" based on input parameters.
Please enter the customer data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=100, value=37)
Type_Of_Contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
City_Tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Duration_Of_Pitch = st.number_input("Duration of Pitch (min)", min_value=1, max_value=200, value=15)
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
No_Of_Person_Visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
No_Of_Followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=0)
Product_Pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe"])
Preferred_Property_Star = st.selectbox("Preferred Property Star", ["3 Star", "4 Star", "5 Star"])
Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
Number_Of_Trips = st.number_input("Number of Trips", min_value=0, max_value=10, value=0)
Passport = st.selectbox("Passport", ["Yes", "No"])
Pitch_Satisfaction_Score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=10, value=5)
Own_Car = st.selectbox("Own Car", ["Yes", "No"])
Number_Of_Children_Visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=2)
Designation = st.selectbox("Designation", ["Executive", "Senior Manager", "Manager", "Senior Executive", "Director", "Other"])
Monthly_Income = st.number_input("Monthly Income", min_value=0, max_value=100000, value=50000)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': Type_Of_Contact,
    'CityTier': City_Tier,
    'DurationOfPitch': Duration_Of_Pitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': No_Of_Person_Visiting,
    'NumberOfFollowups': No_Of_Followups,
    'ProductPitched': Product_Pitched,
    'PreferredPropertyStar': Preferred_Property_Star,
    'MaritalStatus': Marital_Status,
    'NumberOfTrips': Number_Of_Trips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': Pitch_Satisfaction_Score,
    'OwnCar': 1 if Own_Car == "Yes" else 0,
    'NumberOfChildrenVisiting': Number_Of_Children_Visiting,
    'Designation': Designation,
    'MonthlyIncome': Monthly_Income
}])


if st.button("Predict Package Preference"):
    prediction = model.predict(input_data)[0]
    result = "Customer Will Purchase the Wellness Package" if prediction == 1 else "Customer Will Not Purchase the Wellness Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
