import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide"
)

# ---------------- LOAD OBJECTS ----------------
model = tf.keras.models.load_model("regression_model.h5")

label_encoder_gender = pickle.load(
     open("Lb_Encoding\label_encoder_gender.pkl", "rb")
)

onehot_encoder_geo = pickle.load(
    open("One_Hot_Encoding\onehot_encoder_geo.pkl", "rb")
)

scaler = pickle.load(
    open("scaled_data\scaler.pkl", "rb")
)

# ---------------- UI ----------------
st.title("💰 Customer Estimated Salary Predictor")
st.caption("AI-based prediction using banking behavior")
st.divider()

with st.container(border=True):
    st.subheader("📋 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        geography = st.selectbox(
            "Geography",
            onehot_encoder_geo.categories_[0]
        )
        gender = st.selectbox(
            "Gender",
            label_encoder_gender.classes_
        )
        age = st.slider("Age", 18, 92)

    with col2:
        credit_score = st.number_input("Credit Score", 300, 900)
        balance = st.number_input("Balance", min_value=0.0)
        tenure = st.slider("Tenure", 0, 10)

    with col3:
        num_of_products = st.slider("Products", 1, 4)
        has_cr_card = st.toggle("Has Credit Card")
        is_active_member = st.toggle("Active Member")
        exited = st.toggle("Exited")

# ---------------- PREPARE INPUT ----------------
input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [int(has_cr_card)],
    "IsActiveMember": [int(is_active_member)],
    "Exited": [int(exited)]
})

geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out()
)

final_input = pd.concat([input_df, geo_df], axis=1)

final_input_scaled = scaler.transform(final_input)

# ---------------- PREDICTION ----------------
st.divider()
if st.button("🚀 Predict Estimated Salary", use_container_width=True):
    prediction = model.predict(final_input_scaled)
    salary = prediction[0][0]

    st.success("Prediction Successful ✅")
    st.metric("Estimated Salary", f"₹ {salary:,.2f}")
    st.balloons()

st.divider()
st.caption("Built by Pooja Yadav • ANN Regression Model")