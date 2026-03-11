# Apartment Price Prediction Application

This project is a university-style AI applications assignment that implements a complete machine learning workflow and web interface for apartment price prediction.

## Project Description

The application predicts apartment prices based on:
- apartment size (sqm)
- number of rooms
- location score

It uses scikit-learn regression models and compares them through an iterative modeling process. A Streamlit web app allows interactive predictions.

## Project Structure

```
app.py
model_training.py
data_preprocessing.py
requirements.txt
README.md
```

## Dataset Explanation

This project uses a synthetic dataset generated in `data_preprocessing.py`.

Generated features:
- `apartment_size_sqm`
- `number_of_rooms`
- `location_score`

Target:
- `price_eur`

The synthetic target is created from a realistic formula with noise:
- larger apartments cost more
- more rooms increase value
- better location score increases value
- an interaction term (`size * location_score`) models non-linear behavior

A controlled amount of missing values is injected to demonstrate robust preprocessing.

## Data Preprocessing

Implemented preprocessing steps:
- missing value handling with `SimpleImputer(strategy="median")`
- feature scaling with `StandardScaler`
- optional outlier removal using IQR filtering (used in iteration 2)
- feature selection using `SelectKBest(f_regression, k=2)` (used in iteration 2)

## Iterative Modeling Process

The training pipeline in `model_training.py` runs two iterations and evaluates all models using 5-fold cross-validation.

Metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R²

Models included:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Iteration Summary Table

| Iteration | Preprocessing Configuration | Models | Objective |
|---|---|---|---|
| Iteration 1 - Baseline | Imputation + Scaling | Linear Regression, Random Forest Regressor | Build baseline performance |
| Iteration 2 - Advanced | Outlier Removal + Imputation + Scaling + Feature Selection | Linear Regression, Random Forest Regressor, Gradient Boosting Regressor | Improve generalization and reduce error |

During training, the best model is selected by lowest cross-validated RMSE and saved as `trained_model.pkl`. The full evaluation report is saved as `model_report.json`.

## Hyperparameters and Evaluation Results

For every model, the script stores:
- model name
- hyperparameters (`estimator.get_params()`)
- cross-validation means and standard deviations for RMSE, MAE, R²

These are available in:
- `model_report.json`
- Streamlit app section: **Model details and training results**

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
streamlit run app.py
```

If model files are missing, `app.py` will train and save a model automatically on first run.

## Notes for Assignment Submission

- The code is modular and separated into preprocessing, training, and UI.
- The iterative process is explicit and documented.
- Model comparison is reproducible through fixed random seeds.