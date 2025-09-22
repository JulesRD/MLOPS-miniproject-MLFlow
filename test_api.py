import requests
from collections import Counter

BASE_URL = "http://localhost:8000"

def test_predict(num_trials=50):
    """
    Test /predict endpoint and track which model is used for canary.
    """
    data = [5.0, 3.2, 4.1, 0.4]
    used_models = []

    for _ in range(num_trials):
        response = requests.post(f"{BASE_URL}/predict", json={"data": data})
        assert response.status_code == 200, response.text
        resp_json = response.json()
        print("Predictions:", resp_json["predictions"], "| Used model:", resp_json["used_model"])
        used_models.append(resp_json["used_model"])

    counts = Counter(used_models)
    print("Canary usage counts:", counts)
    return counts

def test_update_next_model():
    """
    Test /update-model endpoint (updates next model only)
    """
    new_model_uri = "models:/Iris-model/2"  # Replace with valid MLflow model
    response = requests.post(f"{BASE_URL}/update-model", json={"model_uri": new_model_uri})
    assert response.status_code == 200, response.text
    print("Update next model response:", response.json())

def test_accept_next_model():
    """
    Test /accept-next-model endpoint (promote next model to current)
    """
    response = requests.post(f"{BASE_URL}/accept-next-model")
    assert response.status_code == 200, response.text
    print("Accept next model response:", response.json())

def run_all_tests():
    print("=== Initial predict (both models should be the same) ===")
    counts_before = test_predict()

    print("\n=== Update next model ===")
    test_update_next_model()

    print("\n=== Predict after updating next model (canary routing) ===")
    counts_canary = test_predict()

    print("\n=== Accept next model ===")
    test_accept_next_model()

    print("\n=== Predict after accepting next model ===")
    counts_after_accept = test_predict()

if __name__ == "__main__":
    run_all_tests()