"""
Streamlit Web Application – Zurich Apartment Price Prediction
=============================================================
Loads the persisted best regression model and provides an interactive
interface for predicting monthly rental prices (CHF) for apartments in
the Zurich area.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from model_training import MODEL_PATH, REPORT_PATH, train_and_persist_model

# ── Geographic constant ────────────────────────────────────────────────────────
ZURICH_CENTER_LAT = 47.3769
ZURICH_CENTER_LON = 8.5417


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns the Haversine distance (km) between two geographic coordinates."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@st.cache_resource
def load_model_and_report():
    """
    Loads the persisted model artifact and the JSON report.
    If the artifacts are missing, the training pipeline is triggered automatically.
    """
    if not MODEL_PATH.exists() or not REPORT_PATH.exists():
        with st.spinner("Kein trainiertes Modell gefunden – Training wird gestartet …"):
            train_and_persist_model()

    with MODEL_PATH.open("rb") as fh:
        artifact = pickle.load(fh)          # dict: {"model": pipeline, "features": [...]}

    with REPORT_PATH.open("r", encoding="utf-8") as fh:
        report = json.load(fh)

    return artifact, report


# ── Page layout ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Zürich Mietpreisvorhersage",
        page_icon="🏠",
        layout="centered",
    )

    st.title("🏠 Zürich Mietpreisvorhersage")
    st.markdown(
        "Gib die Merkmale deiner Wohnung ein und erhalte eine Schätzung des "
        "monatlichen Mietpreises (CHF) im Grossraum Zürich – basierend auf einem "
        "Machine-Learning-Regressionsmodell."
    )

    artifact, report = load_model_and_report()
    model_pipeline: object = artifact["model"]
    model_features: list[str] = artifact["features"]
    best_info: dict = report["best_model"]

    st.divider()

    # ── Input controls ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        apartment_size = st.number_input(
            "Wohnfläche (m²)",
            min_value=20.0,
            max_value=250.0,
            value=75.0,
            step=5.0,
            help="Nettowohnfläche der Wohnung in Quadratmetern.",
        )
        number_of_rooms = st.number_input(
            "Anzahl Zimmer",
            min_value=1.0,
            max_value=10.0,
            value=3.5,
            step=0.5,
            help="Schweizer Zimmeranzahl inkl. Wohnzimmer (z. B. 3.5).",
        )

    with col2:
        furnished = st.selectbox(
            "Möbliert?",
            options=[("Nein", 0), ("Ja", 1)],
            format_func=lambda x: x[0],
            index=0,
            help="Ist die Wohnung möbliert?",
        )[1]

        parking = st.selectbox(
            "Parkplatz / Garage?",
            options=[("Nein", 0), ("Ja", 1)],
            format_func=lambda x: x[0],
            index=0,
            help="Ist ein Parkplatz oder eine Garage verfügbar?",
        )[1]

    # ── Distance input ─────────────────────────────────────────────────────────
    st.markdown("##### 📍 Distanz zum Zürcher Stadtzentrum")
    dist_mode = st.radio(
        "Eingabemethode",
        options=["Manuell (km)", "Aus Koordinaten berechnen"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if dist_mode == "Manuell (km)":
        distance_to_zurich_center = st.slider(
            "Distanz zum Stadtzentrum Zürich (km)",
            min_value=0.0,
            max_value=40.0,
            value=5.0,
            step=0.5,
        )
    else:
        c_lat, c_lon = st.columns(2)
        lat_input = c_lat.number_input(
            "Breitengrad (Latitude)",
            value=47.3800,
            format="%.4f",
            step=0.001,
        )
        lon_input = c_lon.number_input(
            "Längengrad (Longitude)",
            value=8.5200,
            format="%.4f",
            step=0.001,
        )
        distance_to_zurich_center = haversine_km(
            ZURICH_CENTER_LAT, ZURICH_CENTER_LON, lat_input, lon_input
        )
        st.info(f"Berechnete Distanz: **{distance_to_zurich_center:.2f} km**")

    st.divider()

    # ── Prediction ─────────────────────────────────────────────────────────────
    rooms_per_sqm = number_of_rooms / apartment_size if apartment_size > 0 else 0.0

    all_input = {
        "apartment_size_sqm":        apartment_size,
        "number_of_rooms":           number_of_rooms,
        "rooms_per_sqm":             rooms_per_sqm,
        "distance_to_zurich_center": distance_to_zurich_center,
        "furnished":                 furnished,
        "parking":                   parking,
    }
    input_df = pd.DataFrame([{f: all_input[f] for f in model_features}])

    if st.button("💰 Mietpreis schätzen", use_container_width=True, type="primary"):
        prediction = float(model_pipeline.predict(input_df)[0])
        st.success(f"### Geschätzter Mietpreis: **CHF {prediction:,.0f} / Monat**")
        st.caption(
            "Hinweis: Dieses Modell wurde mit synthetisch generierten Daten trainiert "
            "und dient ausschliesslich zu Demonstrationszwecken im universitären Kontext."
        )

    # ── Model details expander ─────────────────────────────────────────────────
    with st.expander("🔍 Modelldetails & Trainingsresultate"):
        col_a, col_b = st.columns(2)
        col_a.metric("Bestes Modell",  best_info["model_name"])
        col_a.metric("Iteration",      best_info["iteration"])
        col_b.metric("RMSE (CV)",      f"CHF {best_info['rmse_mean']:,.0f}")
        col_b.metric("R² (CV)",        f"{best_info['r2_mean']:.3f}")

        st.markdown("**Hyperparameter des besten Modells:**")
        hp = best_info.get("hyperparameters", {})
        hp_rows = [(k, str(v)) for k, v in sorted(hp.items()) if v is not None]
        if hp_rows:
            st.dataframe(
                pd.DataFrame(hp_rows, columns=["Parameter", "Wert"]),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("**Alle Modell-Resultate (5-Fold Cross-Validation):**")
        results_df = pd.DataFrame(report["all_results"])[
            ["iteration", "model_name", "rmse_mean", "mae_mean", "r2_mean"]
        ].rename(columns={
            "iteration":  "Iteration",
            "model_name": "Modell",
            "rmse_mean":  "RMSE (CHF)",
            "mae_mean":   "MAE (CHF)",
            "r2_mean":    "R²",
        })
        results_df["RMSE (CHF)"] = results_df["RMSE (CHF)"].map("{:.0f}".format)
        results_df["MAE (CHF)"]  = results_df["MAE (CHF)"].map("{:.0f}".format)
        results_df["R²"]         = results_df["R²"].map("{:.3f}".format)
        st.dataframe(results_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
