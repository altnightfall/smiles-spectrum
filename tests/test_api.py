"""Тесты API"""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_predict_valid_smiles():
    """Тест эндпоинта /predict с валидной SMILES"""
    smiles = "CCO"  # этанол
    response = client.post("/predict", json={"smiles": smiles})
    assert response.status_code == 200
    data = response.json()
    assert "pred_vector" in data
    assert isinstance(data["pred_vector"], list)
    assert len(data["pred_vector"]) > 0
    assert all(isinstance(x, float) for x in data["pred_vector"])


def test_predict_invalid_smiles():
    """Тест эндпоинта /predict с некорректной SMILES"""
    smiles = "INVALID_SMILES"
    response = client.post("/predict", json={"smiles": smiles})
    assert response.status_code == 200
    data = response.json()
    assert data["pred_vector"] == []


def test_predict_empty_smiles():
    """Тест эндпоинта /predict с пустой SMILES"""
    response = client.post("/predict", json={"smiles": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["pred_vector"] == []


def test_predict_batch():
    """Тест эндпоинта /predict со списком SMILES"""
    smiles_list = ["CCO", "INVALID", ""]
    response = client.post("/predict_batch", json={"smiles_list": smiles_list})
    assert response.status_code == 200
    data = response.json()
    assert "pred_vectors" in data
    assert isinstance(data["pred_vectors"], list)
    assert len(data["pred_vectors"]) == 3
    assert isinstance(data["pred_vectors"][0], list) and len(data["pred_vectors"][0]) > 0
    assert data["pred_vectors"][1] == []  # INVALID
    assert data["pred_vectors"][2] == []  # empty string
