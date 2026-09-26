from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_WEATHER = {
    "date": "2025-01-15",
    "temperature_2m": 22,
    "relative_humidity_2m": 65,
    "precipitation": 0,
    "cloud_cover": 30,
    "wind_speed_10m": 12,
    "wind_gusts_10m": 25,
    "snowfall": 0
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_cci_prediction():
    response = client.post(
        "/predict/cci",
        json=VALID_WEATHER
    )

    assert response.status_code == 200

    result = response.json()

    assert "cci" in result
    assert result["prediction_type"] == "Climate Comfort Index"


def test_whc_prediction():
    response = client.post(
        "/predict/whc",
        json=VALID_WEATHER
    )

    assert response.status_code == 200

    result = response.json()

    assert "whc_class" in result
    assert "risk_level" in result


def test_missing_parameter():
    invalid_data = VALID_WEATHER.copy()

    del invalid_data["temperature_2m"]

    response = client.post(
        "/predict/cci",
        json=invalid_data
    )

    assert response.status_code == 422


def test_invalid_humidity():
    invalid_data = VALID_WEATHER.copy()

    invalid_data["relative_humidity_2m"] = 150

    response = client.post(
        "/predict/cci",
        json=invalid_data
    )

    assert response.status_code == 422


def test_negative_precipitation():
    invalid_data = VALID_WEATHER.copy()

    invalid_data["precipitation"] = -10

    response = client.post(
        "/predict/whc",
        json=invalid_data
    )

    assert response.status_code == 422


def test_invalid_cloud_cover():
    invalid_data = VALID_WEATHER.copy()

    invalid_data["cloud_cover"] = 150

    response = client.post(
        "/predict/cci",
        json=invalid_data
    )

    assert response.status_code == 422


def test_invalid_date():
    invalid_data = VALID_WEATHER.copy()

    invalid_data["date"] = "not-a-date"

    response = client.post(
        "/predict/whc",
        json=invalid_data
    )

    assert response.status_code == 422
