from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_valid():
    response = client.post("/predict", json={"features": [0.1] * 29})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data or "error" in data
