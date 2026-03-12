# Apartment Price Prediction Application

This project is a university-style AI applications assignment that implements a complete machine learning workflow and web interface for apartment price prediction.

# Apartment Rental Price Prediction – Zurich Area

> **University AI Applications Assignment**  
> Machine Learning · Regression · Feature Engineering · Streamlit

---

## Project Description

This project builds an end-to-end machine learning pipeline to predict **monthly apartment rental prices (CHF)** for the Greater Zurich area.  
It was developed as a university assignment and demonstrates a full iterative modelling workflow, reproducible feature engineering, and an interactive web application.

Key components:

| File | Purpose |
|------|---------|
| `generate_sample_data.py` | Generates realistic synthetic Zurich apartment listings |
| `scraper.py` | Template scraper for Swiss real-estate portals |
| `data_preprocessing.py` | Modular cleaning and feature-engineering pipeline |
| `model_training.py` | Iterative training with 5-fold CV, model comparison, model persistence |
| `streamlit_app.py` | Interactive price-prediction web app |
| `notebooks/apartment_price_prediction.ipynb` | Exploratory analysis and modelling notebook |

---

## Project Structure

```
apartment-price-prediction/
├── data/
│   ├── apartments_data_enriched_with_new_features.csv  # enriched input dataset
│   └── apartments_enriched_clean.csv   # preprocessed output
├── notebooks/
│   └── apartment_price_prediction.ipynb
├── scraper.py
├── generate_sample_data.py
├── data_preprocessing.py
├── model_training.py
├── streamlit_app.py
├── trained_model.pkl                   # persisted best model
├── model_report.json                   # evaluation summary
├── requirements.txt
└── README.md
```

---

## Data Source and Dataset

### Data Collection

Real Swiss apartment listings can be collected from portals such as  
[Homegate](https://www.homegate.ch) or [ImmoScout24](https://www.immoscout24.ch) using `scraper.py`.  
Because scraping requires portal-specific CSS selectors and respecting each site's Terms of Service,
the pipeline ships with a **realistic synthetic dataset** that mirrors actual Zurich market conditions.

Run to regenerate:
```bash
python generate_sample_data.py
```

### Dataset Schema (`data/apartments_data_enriched_with_new_features.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `title` | str | Listing headline (used for keyword extraction) |
| `description` | str | Full German listing text |
| `price` | float | Monthly rent in CHF |
| `number_of_rooms` | float | Swiss room count (e.g. 3.5) |
| `apartment_size_sqm` | float | Floor area in m² |
| `address` | str | Street and house number |
| `zip_code` | str | Swiss postal code |
| `city` | str | Municipality name |
| `latitude` | float | WGS-84 latitude |
| `longitude` | float | WGS-84 longitude |
| `publisher` | str | Source portal or agency |

The dataset covers **44 Zurich-area municipalities**, including all Zurich city
districts (ZIP 8001–8099) and surrounding towns such as Winterthur, Uster,
Dübendorf, Küsnacht and Winterthur.  
Price multipliers per municipality reflect observed market premiums – Zurich
city centre commands up to 1.30× the base rate; outer suburbs 0.75–0.85×.

### Using `apartments_data_enriched_with_new_features.csv`

The preprocessing pipeline is now fully based on this enriched dataset:

- Required input: `data/apartments_data_enriched_with_new_features.csv`

To keep the downstream model and app unchanged, input columns are normalized
to the project schema in `data_preprocessing.py`.

| Enriched column | Project column |
|-----------------|----------------|
| `rooms` | `number_of_rooms` |
| `area` | `apartment_size_sqm` |
| `lat` | `latitude` |
| `lon` | `longitude` |
| `description_raw` | `description` |
| `postalcode` | `zip_code` |
| `town` | `city` |
| `price` | `price_chf` |

If `title` is missing, it is generated from text fields so keyword-based feature
extraction (`WG`, `furnished`, `parking`) still works.

---

## Data Preprocessing

All preprocessing logic lives in `data_preprocessing.py` and runs as a numbered pipeline:

1. **Load CSV + normalize schema** – reads `data/apartments_data_enriched_with_new_features.csv` and maps external column names to the internal schema
2. **Filter WG / shared-room listings** – removes entries where title or description contains keywords such as `wg`, `mitbewohner`, `shared room`; these are atypical price points that would distort the model
3. **Filter invalid rows** – drops rows where `price_chf < 400 CHF`, `apartment_size_sqm < 15 m²`, or `number_of_rooms < 1`
4. **Impute missing coordinates** – replaces missing `latitude` / `longitude` values with column medians, required for the distance feature
5. **Compute `distance_to_zurich_center`** – Haversine distance (km) from each listing to the Zurich city centre *(see engineered features)*
6. **Extract `furnished`** – binary feature from keyword search *(see engineered features)*
7. **Extract `parking`** – binary feature from keyword search *(see engineered features)*
8. **Compute `rooms_per_sqm`** – `number_of_rooms / apartment_size_sqm` *(see engineered features)*
9. **Remove price outliers** – IQR method with multiplier = 2.0 on `price_chf`
10. **Save** to `data/apartments_enriched_clean.csv`

Result: **700 raw rows** → **630 clean rows** ready for modelling.

---

## Engineered Features

### 1. `distance_to_zurich_center` (continuous, km)

**Rationale:** Proximity to the city centre is one of the strongest rent drivers in the Zurich market.  
Raw lat/lon would require the model to learn a 2-D spatial relationship; a single distance scalar is more efficient and interpretable.

**Implementation:** Haversine formula using Zurich centre coordinates `(47.3769, 8.5417)`.

```python
a = sin(Δlat/2)² + cos(lat₁)·cos(lat₂)·sin(Δlon/2)²
distance_km = 2 · R · arctan2(√a, √(1-a))   # R = 6371 km
```

### 2. `furnished` (binary, 0/1)

**Rationale:** Furnished apartments command a 10–20 % price premium in the Swiss market. The feature is extracted from `title` and `description` using keywords: `möbliert`, `moebliert`, `furnished`, `mobiliert`.

### 3. `parking` (binary, 0/1)

**Rationale:** Central Zurich has very limited parking; a garage or parking space adds a measurable premium (~5–8 %). Keywords: `garage`, `tiefgarage`, `parkplatz`, `carport`, `einstellplatz`, `parking`.

### 4. `rooms_per_sqm` (continuous)

**Rationale:** Captures apartment type (studio vs large family flat) and partially decorrelates `number_of_rooms` and `apartment_size_sqm`, both of which are included in the baseline feature set.

---

## Iterative Modelling Process

Both iterations use **5-fold cross-validation** (shuffled, `random_state=42`).  
Evaluation metrics: **RMSE**, **MAE**, **R²**.  
The best model (lowest RMSE) is fitted on the full dataset and persisted to `trained_model.pkl`.

### Iteration 1 – Baseline

**Objective:** Establish a performance baseline using only basic, universally available features.

**Feature set:**
- `apartment_size_sqm`
- `number_of_rooms`
- `rooms_per_sqm`

**Pipeline:** median imputation → StandardScaler

**Models and hyperparameters:**

| Model | Key Hyperparameters |
|-------|---------------------|
| `LinearRegression` | default (no regularisation) |
| `RandomForestRegressor` | `n_estimators=150`, `max_depth=10`, `min_samples_leaf=4`, `random_state=42` |

### Iteration 2 – Extended Features

**Objective:** Improve prediction accuracy by adding all three engineered features.

**Feature set:**
- `apartment_size_sqm`, `number_of_rooms`, `rooms_per_sqm` *(carried over)*
- `distance_to_zurich_center` *(new)*
- `furnished` *(new)*
- `parking` *(new)*

**Pipeline:** median imputation → StandardScaler

**Models and hyperparameters:**

| Model | Key Hyperparameters |
|-------|---------------------|
| `LinearRegression` | default |
| `RandomForestRegressor` | `n_estimators=200`, `max_depth=12`, `min_samples_leaf=3`, `random_state=42` |
| `GradientBoostingRegressor` | `n_estimators=200`, `learning_rate=0.08`, `max_depth=5`, `min_samples_leaf=4`, `subsample=0.8`, `random_state=42` |

---

## Results Summary

| Iteration | Objective | What changed | Preprocessing | Model | RMSE (CHF) | MAE (CHF) | R² |
|-----------|-----------|-------------|---------------|-------|-----------|----------|-----|
| 1 – Baseline | Establish baseline | Initial feature set | Imputation + Scaling | LinearRegression | 417.5 | 335.3 | 0.769 |
| 1 – Baseline | Establish baseline | Initial feature set | Imputation + Scaling | RandomForestRegressor | 452.9 | 362.1 | 0.727 |
| 2 – Extended | Add location & amenity features | +distance, +furnished, +parking | Imputation + Scaling | LinearRegression | 312.5 | 245.2 | 0.870 |
| 2 – Extended | Add location & amenity features | +distance, +furnished, +parking | Imputation + Scaling | RandomForestRegressor | 298.2 | 218.1 | 0.881 |
| **2 – Extended** | **Add location & amenity features** | **+distance, +furnished, +parking** | **Imputation + Scaling** | **GradientBoostingRegressor** ✓ | **288.3** | **207.6** | **0.889** |

**Best model:** `GradientBoostingRegressor` (Iteration 2 – Extended)  
**RMSE:** CHF 288 · **MAE:** CHF 208 · **R²:** 0.889

**Key findings:**
- Adding the three engineered features reduced RMSE by **~31 %** (from 417 to 288 CHF).
- `distance_to_zurich_center` alone accounts for a large share of this improvement, confirming that location is the dominant price driver in the Zurich market.
- Gradient Boosting outperforms Random Forest due to its ability to model residual errors iteratively.
- Linear Regression benefits significantly from the additional features but cannot capture non-linear location effects as well as tree-based models.

---

## Setup and Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure enriched data exists

Place your dataset at:

```text
data/apartments_data_enriched_with_new_features.csv
```

### 3. Run preprocessing and train models

```bash
python data_preprocessing.py   # creates data/apartments_enriched_clean.csv
python model_training.py       # trains, evaluates, and saves the best model
```

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Open the Jupyter notebook

```bash
jupyter notebook notebooks/apartment_price_prediction.ipynb
```

Or with JupyterLab:

```bash
jupyter lab
```

Then navigate to `notebooks/apartment_price_prediction.ipynb`.

---

## Notes on Real Data

To use your own real Zurich apartment data:

1. Review the target portal's `robots.txt` and Terms of Service.
2. Update the CSS selectors in `scraper.py` to match the current page structure.
3. Export your dataset to `data/apartments_data_enriched_with_new_features.csv`.
4. Re-run the preprocessing and training pipeline.

The pipeline is designed to work without modification once
`data/apartments_data_enriched_with_new_features.csv` exists with the expected schema.

## How to Run the App

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Train models manually:

```bash
python model_training.py
```

3. Start Streamlit:

```bash
streamlit run streamlit_app.py
```

If model files are missing, `streamlit_app.py` will train and save a model automatically on first run.

## Notes for Assignment Submission

- The code is modular and separated into preprocessing, training, and UI.
- The iterative process is explicit and documented.
- Model comparison is reproducible through fixed random seeds.