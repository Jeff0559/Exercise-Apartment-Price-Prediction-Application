import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from model_training import MODEL_PATH, REPORT_PATH, train_and_persist_model


def load_model_and_report():
    """Loads model and training report, and trains a model if artifacts are missing."""
    if not MODEL_PATH.exists() or not REPORT_PATH.exists():
        with st.spinner("No trained model found. Training model now..."):
            train_and_persist_model()

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    with REPORT_PATH.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    return model, report


def main():
    st.set_page_config(page_title="Apartment Price Prediction", page_icon="🏠", layout="centered")
    st.title("Apartment Price Prediction App")
    st.write(
        "Enter apartment features and get an estimated apartment price based on the best "
        "trained regression model."
    )

    model, report = load_model_and_report()

    apartment_size = st.number_input("Apartment size (sqm)", min_value=20.0, max_value=250.0, value=70.0, step=1.0)
    number_of_rooms = st.number_input("Number of rooms", min_value=1, max_value=10, value=3, step=1)
    location_score = st.slider("Location score", min_value=1.0, max_value=10.0, value=6.0, step=0.1)

    input_df = pd.DataFrame(
        [
            {
                "apartment_size_sqm": apartment_size,
                "number_of_rooms": float(number_of_rooms),
                "location_score": location_score,
            }
        ]
    )

    if st.button("Predict apartment price"):
        prediction = float(model.predict(input_df)[0])
        st.metric("Predicted apartment price", f"EUR {prediction:,.0f}")

    with st.expander("Model details and training results"):
        st.write(f"Selected model: {report['best_model']['model_name']}")
        st.write(f"Iteration: {report['best_model']['iteration']}")
        st.write(f"Hyperparameters: {report['best_model']['hyperparameters']}")
        st.dataframe(pd.DataFrame(report["all_results"]))


if __name__ == "__main__":
    main()