from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras


project_dir = Path(__file__).parent


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(project_dir / "churn_preprocessor.joblib")
    model = keras.models.load_model(
        project_dir / "churn_model.keras",
        compile=False,
    )
    return preprocessor, model


st.title("Customer churn prediction")

with st.form("customer_details"):
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    credit_score = st.number_input("Credit score", min_value=0, value=650)
    age = st.number_input("Age", min_value=18, value=40)
    tenure = st.number_input("Years with the bank", min_value=0, value=5)
    balance = st.number_input("Account balance", min_value=0.0, value=50000.0)
    products = st.number_input("Number of products", min_value=1, value=2)
    has_card = st.checkbox("Has a credit card", value=True)
    is_active = st.checkbox("Active member", value=True)
    salary = st.number_input("Estimated salary", min_value=0.0, value=75000.0)

    submitted = st.form_submit_button("Predict churn")

if submitted:
    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": int(has_card),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": salary,
    }])

    preprocessor, model = load_artifacts()
    features = preprocessor.transform(customer).astype("float32")
    score = float(model.predict(features, verbose=0)[0, 0])

    st.metric("Churn score", f"{score:.1%}")
    if score >= 0.5:
        st.write("Predicted outcome: Leaves the bank")
    else:
        st.write("Predicted outcome: Stays with the bank")