# Sydney Weather Prediction API

This project provides a machine learning API for predicting short-term weather comfort and hazard levels in Sydney, Australia.

The project contains two prediction tasks:

1. Climate Comfort Index (CCI) prediction
2. Weather Hazard Category (WHC) prediction

The models were trained using historical weather data from the Open-Meteo Historical Weather API.

---

## Project Objectives

### Climate Comfort Index

The CCI model predicts the average Climate Comfort Index for the next three days.

The final model is a Random Forest Regressor.

Test performance:

- MAE: 6.1035
- RMSE: 7.8207
- R²: 0.3697

### Weather Hazard Category

The WHC model predicts the Weather Hazard Category exactly seven days ahead.

The final model is an XGBoost Classifier.

Test performance:

- Accuracy: 0.4693
- Macro F1: 0.2413
- Weighted F1: 0.4547

The WHC model performs better on Low Risk and Moderate Risk conditions, but it has difficulty identifying rare High Risk and Extreme Risk events because of strong class imbalance.

---

## Input Features

Both models use the following weather variables:

- temperature_2m
- relative_humidity_2m
- precipitation
- cloud_cover
- wind_speed_10m
- wind_gusts_10m
- snowfall

Two additional time-based features are generated automatically:

- month
- day_of_year

---

## Machine Learning Models

### CCI

Three regression models were evaluated:

- Linear Regression
- Random Forest
- XGBoost

Random Forest achieved the best validation performance and was selected as the final model.

### WHC

Three classification models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

XGBoost achieved the strongest overall validation performance and was selected as the final model.

---

## API

The application is built using FastAPI.

Available endpoints:

- `GET /`
- `GET /health`
- `POST /predict/cci`
- `POST /predict/whc`

---

## Online Deployment

The API is deployed on Render.

Base URL:

https://three6120-at2-weather-prediction-api.onrender.com

Interactive API documentation:

https://three6120-at2-weather-prediction-api.onrender.com/docs

Health check:

https://three6120-at2-weather-prediction-api.onrender.com/health

---

## Example CCI Request

```json
{
  "date": "2025-01-15",
  "temperature_2m": 22,
  "relative_humidity_2m": 65,
  "precipitation": 0,
  "cloud_cover": 30,
  "wind_speed_10m": 12,
  "wind_gusts_10m": 25,
  "snowfall": 0
}
Example response:
{
  "prediction_type": "Climate Comfort Index",
  "prediction_horizon": "Next 3 days average",
  "cci": 78.39
}
## Example WHC Request
{
  "date": "2025-01-15",
  "temperature_2m": 22,
  "relative_humidity_2m": 65,
  "precipitation": 10,
  "cloud_cover": 70,
  "wind_speed_10m": 20,
  "wind_gusts_10m": 45,
  "snowfall": 0
}
Example response:
{
  "prediction_type": "Weather Hazard Category",
  "prediction_horizon": "Exactly 7 days ahead",
  "whc_class": 0,
  "risk_level": "Low Risk"
}
Project Structure
36120-AT2-weather-prediction-api/
├── app.py
├── requirements.txt
├── Dockerfile
└── models/
    ├── cci_random_forest.pkl
    ├── cci_features.pkl
    ├── whc_xgboost.pkl
    └── whc_features.pkl
Technologies
Python
Pandas
Scikit-learn
XGBoost
FastAPI
Docker
Render
GitHub
Limitations

The CCI model provides a useful baseline for short-term comfort prediction, but prediction error still remains.

The WHC model is strongly affected by class imbalance. High Risk and Extreme Risk events are rare in the dataset, and the current model does not reliably identify these severe classes.

Future improvements may include lagged weather features, rolling statistics, class-sensitive learning and additional severe-weather data.
