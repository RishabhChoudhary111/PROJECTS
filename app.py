import streamlit as st
import pandas as pd
import joblib

st.title("Creditworthiness Predictor")

clf = joblib.load("credit_rf_model.pkl")
st.write("Enter financial attributes:")

# Example inputs:
checking = st.selectbox("Checking Account Status", ["<0","0-200",">=200","none"])
duration = st.number_input("Duration (months)", 1, 100)
credit_amount = st.number_input("Credit Amount", value=1000)
purpose = st.selectbox("Purpose", ["car","education","furniture","others"])
# … add more features

if st.button("Predict"):
    df = pd.DataFrame([{
        "checking": checking,
        "duration": duration,
        "credit_amount": credit_amount,
        "purpose": purpose
        # … include rest
    }])
    df_encoded = pd.get_dummies(df)
    # Ensure missing columns add zeros
    feature_names = joblib.load("model_features.pkl")
    model_input = df_encoded.reindex(columns=feature_names, fill_value=0)

    pred = clf.predict(model_input)[0]
    st.success("Risk: **Bad**" if pred else "Risk: **Good**")
