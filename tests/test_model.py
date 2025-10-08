"""Тесты модели и данных"""

import os
import pickle

import numpy as np
import pytest
from tensorflow.keras.models import (  # pylint: disable=import-error,no-name-in-module
    load_model,
)

from backend.app.models import train_model_mass as tm

DATA_DIR = "train_data"
MODEL_PATH = "backend/app/models/spectrum_predictor.keras"
SCALER_PATH = os.path.join(DATA_DIR, "spectrum_scaler.pkl")


def test_smiles_to_fp():
    """Тест SMILES → fingerprint"""
    smiles = "CCO"
    fp = tm.smiles_to_fp(smiles)
    assert fp is not None, "Fingerprint не должен быть None"
    assert fp.shape[0] == 2048, "Fingerprint должен иметь размер 2048"
    assert isinstance(fp[0], np.float32), "Элементы должны быть float32"


def test_spectrum_to_vector():
    """Тест спектра → вектор"""
    peaks = [(50, 100), (100, 200), (999, 50)]
    vec = tm.spectrum_to_vector(peaks)
    assert vec.shape[0] == 1001, "Вектор спектра должен иметь размер 1001"
    assert np.all((vec >= 0) & (vec <= 1)), "Вектор спектра должен быть нормализован"


def test_build_dataset():
    """Тест сборки датасета"""
    if not os.path.exists(DATA_DIR):
        pytest.skip("Датасет не найден, пропуск")
    X, y, _ = tm.build_dataset(limit=5)
    assert X.shape[0] == y.shape[0], "Количество X и y должно совпадать"
    assert X.shape[1] == 2048, "Размер X должен быть 2048"
    assert y.shape[1] == 1001, "Размер y должен быть 1001"


def test_model_inference():
    """Тест загрузки модели и инференса"""
    if not os.path.exists(DATA_DIR):
        pytest.skip("Датасет не найден, пропуск")
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Модель не найдена, пропуск инференса")
    model = load_model(MODEL_PATH)
    X, _, _ = tm.build_dataset(limit=1)
    pred = model.predict(X)
    assert pred.shape[1] == 1001, "Выход модели должен иметь размер 1001"

    # проверка scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        assert scaler.shape[0] == pred.shape[0], "Scaler должен соответствовать числу образцов"
