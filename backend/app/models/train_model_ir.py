"""Train ir model"""

import os
import pickle
import zipfile

import numpy as np
import pandas as pd
import requests
from tensorflow.keras import (  # pylint: disable=import-error,no-name-in-module
    layers,
    models,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,  # pylint: disable=import-error,no-name-in-module
)
from tqdm import tqdm

# 📁 URL и директории
URL = "https://figshare.com/ndownloader/articles/24593022/versions/1"
DATA_DIR = "train_data_ir"
ARCHIVE_PATH = os.path.join(DATA_DIR, "IR-dataset.zip")
EXTRACT_DIR = DATA_DIR


def download_dataset():
    """Скачивание датасета IR"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ARCHIVE_PATH):
        print("🔽 Скачиваем IR-dataset...")
        r = requests.get(URL, stream=True, timeout=500)
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


def parse_csv_record(path, min_cm=650, max_cm=4000, step=1.0):
    """Парсинг CSV спектра → вектор с пропуском первой строки и гибким определением колонок"""
    try:
        df = pd.read_csv(path, skiprows=1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ Ошибка чтения файла {path}: {e}")
        return None

    cm_col = next((c for c in df.columns if "cm" in c.lower()), None)
    t_col = next((c for c in df.columns if "%" in c), None)

    if cm_col is None or t_col is None:
        print(f"❌ Не найдено cm или %T в файле: {path}")
        return None

    df = df[(df[cm_col] >= min_cm) & (df[cm_col] <= max_cm)]
    bins = np.arange(min_cm, max_cm + step, step)
    vec = np.interp(bins, df[cm_col], df[t_col])
    vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-8)
    return vec.astype(np.float32)


def get_material_name(filename):
    """Извлечение названия материала из имени файла (до дефиса)"""
    base = os.path.basename(filename)
    name = base.split("-")[0].upper()
    return name


def build_dataset():
    """Build IR dataset"""
    x_indices = []
    y_spectra = []

    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"🔍 Найдено {len(files)} CSV-файлов.")

    material_set = set()

    for path in tqdm(files):
        vec = parse_csv_record(path)
        if vec is None:
            continue

        material = get_material_name(path)
        if not material:
            print(f"❌ Не удалось определить материал для файла: {path}")
            continue

        material_set.add(material)
        y_spectra.append(vec)

    if not material_set:
        raise ValueError("Нет подходящих файлов для построения датасета!")

    material_list = sorted(list(material_set))
    with open(os.path.join(DATA_DIR, "material_list.pkl"), "wb") as f:
        pickle.dump(material_list, f)

    x_indices = [material_list.index(get_material_name(path)) for path in files if parse_csv_record(path) is not None]

    x = np.array(x_indices, dtype=np.int32).reshape(-1, 1)  # индекс как вход
    y = np.array(y_spectra, dtype=np.float32)
    return x, y, material_list


def build_model(output_dim, n_materials):
    """Build IR model"""
    model = models.Sequential(
        [
            layers.Input(shape=(1,), dtype="int32"),
            layers.Embedding(input_dim=n_materials, output_dim=16),  # embedding для материала
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(output_dim, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    """Основной блок"""
    download_dataset()

    x, y, material_list = build_dataset()
    print("✅ Датасет собран:", x.shape, y.shape)

    n_materials = len(material_list)
    output_dim = y.shape[1]

    model = build_model(output_dim=output_dim, n_materials=n_materials)
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(x, y, validation_split=0.1, epochs=50, batch_size=16, callbacks=[early_stop])

    model.save("backend/app/models/ir_spectrum_predictor.keras")
    print("💾 Модель сохранена")


if __name__ == "__main__":
    main()
