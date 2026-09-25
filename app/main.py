
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import date
from pathlib import Path


app = FastAPI(
    title="Sydney Weather Prediction API",
    description="API for predicting Climate Comfort Index and Weather Hazard Category",
    version="1.0"
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


cci_model = joblib.load(
    MODEL_DIR / "cci_random_forest.pkl"
)

cci_features = joblib.load(
    MODEL_DIR / "cci_features.pkl"
)

whc_model = joblib.load(
    MODEL_DIR / "whc_xgboost.pkl"
)

whc_features = joblib.load(
    MODEL_DIR / "whc_features.pkl"
)


class WeatherInput(BaseModel):
    date: date
    temperature_2m: float
    relative_humidity_2m: float
    precipitation: float
    cloud_cover: float
    wind_speed_10m: float
    wind_gusts_10m: float
    snowfall: float


def prepare_features(data: WeatherInput, feature_list):
    input_data = {
        "temperature_2m": data.temperature_2m,
        "relative_humidity_2m": data.relative_humidity_2m,
        "precipitation": data.precipitation,
        "cloud_cover": data.cloud_cover,
        "wind_speed_10m": data.wind_speed_10m,
        "wind_gusts_10m": data.wind_gusts_10m,
        "snowfall": data.snowfall,
        "month": data.date.month,
        "day_of_year": data.date.timetuple().tm_yday
    }

    df = pd.DataFrame([input_data])

    return df[feature_list]


@app.get("/")
def home():
    return {
        "message": "Sydney Weather Prediction API",
        "cci_endpoint": "/predict/cci",
        "whc_endpoint": "/predict/whc"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict/cci")
def predict_cci(data: WeatherInput):
    features = prepare_features(
        data,
        cci_features
    )

    prediction = cci_model.predict(features)[0]

    return {
        "prediction_type": "Climate Comfort Index",
        "prediction_horizon": "Next 3 days average",
        "cci": round(float(prediction), 2)
    }


@app.post("/predict/whc")
def predict_whc(data: WeatherInput):
    features = prepare_features(
        data,
        whc_features
    )

    prediction = int(
        whc_model.predict(features)[0]
    )

    labels = {
        0: "Low Risk",
        1: "Moderate Risk",
        2: "High Risk",
        3: "Extreme Risk"
    }

    return {
        "prediction_type": "Weather Hazard Category",
        "prediction_horizon": "Exactly 7 days ahead",
        "whc_class": prediction,
        "risk_level": labels[prediction]
    }
