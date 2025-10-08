import os
import pickle
import re
import zipfile

import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import (  # pylint: disable=import-error,no-name-in-module
    layers,
    models,
)
from tensorflow.keras.callbacks import (  # pylint: disable=import-error,no-name-in-module
    EarlyStopping,
)
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/record/11487030/files/MassBank/MassBank-data-2024.06.zip"
DATA_DIR = "train_data"
ARCHIVE_PATH = os.path.join(DATA_DIR, "MassBank-data.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "MassBank-data")


def download_dataset():
    """Скачивание датасета"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ARCHIVE_PATH):
        print("🔽 Скачиваем MassBank...")
        r = requests.get(ZENODO_URL, stream=True, timeout=500)
        with open(ARCHIVE_PATH, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)
    else:
        print("✅ Архив уже скачан.")

    if not os.path.exists(EXTRACT_DIR):
        print("📦 Распаковываем архив...")
        with zipfile.ZipFile(ARCHIVE_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
    else:
        print("✅ Датасет уже распакован.")


def parse_massbank_record(path):
    """Парсинг MassBank файлов"""
    smiles, peaks = None, []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("CH$SMILES:"):
                smiles = line.split(":", 1)[1].strip()
            elif re.match(r"^\s*\d", line):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz, intensity = map(float, parts[:2])
                        peaks.append((mz, intensity))
                    except ValueError:
                        continue
    return smiles, peaks


def smiles_to_fp(smiles, n_bits=2048):
    """SMILES → fingerprint"""
    try:
        mol = Chem.MolFromSmiles(smiles)  # pylint: disable=no-member
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)  # pylint: disable=no-member
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)  # pylint: disable=no-member
        return arr
    except Exception:
        return None


def spectrum_to_vector(peaks, max_mz=1000, step=1.0):
    """Спектр → вектор"""
    bins = np.arange(0, max_mz + step, step)
    vec = np.zeros(len(bins), dtype=np.float32)
    for mz, intensity in peaks:
        idx = int(mz // step)
        if idx < len(vec):
            vec[idx] = max(vec[idx], intensity)
    if vec.max() > 0:
        vec /= vec.max()  # нормализация в диапазон [0,1]
    return vec


def build_dataset(limit=2000):
    """Подготовка датасета с сохранением SMILES"""
    X, y, smiles_list = [], [], []
    files = []
    for root, _, fs in os.walk(EXTRACT_DIR):
        for name in fs:
            if name.endswith(".txt"):
                files.append(os.path.join(root, name))

    print(f"🔍 Найдено {len(files)} файлов.")
    for path in tqdm(files[:limit]):
        smiles, peaks = parse_massbank_record(path)
        if smiles and peaks:
            fp = smiles_to_fp(smiles)
            if fp is None:
                continue
            vec = spectrum_to_vector(peaks)
            X.append(fp)
            y.append(vec)
            smiles_list.append(smiles)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if len(y.shape) == 1:
        y = np.stack(y, axis=0)

    spectrum_max = np.max(y, axis=1, keepdims=True)
    spectrum_max[spectrum_max == 0] = 1.0
    y /= spectrum_max
    with open(os.path.join(DATA_DIR, "spectrum_scaler.pkl"), "wb") as f:
        pickle.dump(spectrum_max, f)

    return X, y, smiles_list


def build_model(input_dim, output_dim):
    """Модель"""
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(output_dim, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate_smiles_predictions(smiles_list, true_spectra, model):
    """
    Функция оценки SMILES
    """
    x_fp = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_fp(smi)
        if fp is not None:
            x_fp.append(fp)
            valid_indices.append(i)

    if not x_fp:
        return None

    x_fp = np.array(x_fp, dtype=np.float32)
    y_true = true_spectra[valid_indices]
    y_pred = model.predict(x_fp, verbose=0)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"MSE": mse, "MAE": mae, "R2": r2}


def main():
    """Основной блок"""
    download_dataset()
    x, y, smiles_list = build_dataset(limit=15000)
    print("✅ Датасет собран:", x.shape, y.shape)

    model = build_model(x.shape[1], y.shape[1])
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(x, y, validation_split=0.1, epochs=50, batch_size=32, callbacks=[early_stop])

    model.save("backend/app/models/spectrum_predictor.keras")
    print("💾 Модель сохранена в backend/app/models/spectrum_predictor.keras")

    metrics = evaluate_smiles_predictions(smiles_list=smiles_list, true_spectra=y, model=model)
    print("Метрики на наборе:", metrics)


if __name__ == "__main__":
    main()
