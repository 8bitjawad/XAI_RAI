import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
from llm_engine import generate_explanation
from core import RAIWrapper

model = joblib.load("model.pkl")

feature_names = joblib.load("features.pkl")
X_test_full = joblib.load("X_test_full.pkl")
y_test = joblib.load("y_test.pkl")

X_train = pd.DataFrame(
    np.zeros((1, len(feature_names))),
    columns=feature_names
)

from core import wrap

rai_model = wrap(
    model,
    X_train,
    sensitive_column="trader_group"
)

st.title("Universal XAI + Responsible AI Layer")

st.write("Demo: Model Prediction with Explanation and Fairness Checks")

st.header("Input Features")

open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")
close_price = st.number_input("Close Price")
volume = st.number_input("Volume")
run_button = st.button("Run Prediction")

if run_button:

    if rai_model is None:
        st.error("Model not loaded yet.")
    else:
        sample = pd.DataFrame([[open_price, high_price, low_price, close_price, volume]],
        columns=feature_names
)

        result = rai_model.predict(sample)

        st.subheader("Prediction")
        prediction = result["prediction"]

        if prediction == 1:
            prediction_text = "Price is likely to go UP"
        else:
            prediction_text = "Price is likely to go DOWN"
        st.write(prediction_text)

        st.subheader("Confidence")
        st.write(result["confidence"])

        st.subheader("Top Factors")

        for feature, value in result["top_factors"]:
            st.write(f"{feature} ({round(value,3)})")

        plot = rai_model.explain_engine.shap_plot(sample)

        st.subheader("Feature Impact Visualization")

        st.pyplot(plot)

        fairness = rai_model.fairness_report(X_test_full, y_test)

        st.subheader("Responsible AI Report")

        st.write(f"Protected Attribute: {fairness['Protected Attribute']}")
        st.write(f"Prediction Difference: {fairness['Prediction Difference (%)']}%")
        st.write(f"Bias Detected: {fairness['Bias Detected']}")

        explanation = generate_explanation(result)

        st.subheader("AI Explanation")

        st.write(explanation)