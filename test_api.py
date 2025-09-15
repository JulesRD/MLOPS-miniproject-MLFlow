import numpy as np
import requests

BASE_URL = "http://localhost:8000"

def test_predict():
    # Example input, adapt to your model
    data = [5.0, 3.2, 4.1, 0.4]
    response = requests.post(f"{BASE_URL}/predict", json={"data": data})
    assert response.status_code == 200, response.text
    print("Predictions:", response.json())


def test_update_model():
    new_model_uri = "models:/Iris-model/1"  # Example: new version in MLflow
    response = requests.post(f"{BASE_URL}/update-model", json={"model_uri": new_model_uri})
    assert response.status_code == 200, response.text
    print("Update response:", response.json())

    # Test predict again to ensure updated model works
    data = [3.0, 7.0, 8.0, 9.0]
    response2 = requests.post(f"{BASE_URL}/predict", json={"data": data})
    assert response2.status_code == 200, response2.text
    print("Predictions with updated model:", response2.json())


if __name__ == "__main__":
    test_predict()
    test_update_model()
