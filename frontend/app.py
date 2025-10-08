"""Frontend"""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL = st.sidebar.text_input("API URL", DEFAULT_API_URL)

tab = st.sidebar.radio("Выберите модель", ("MS", "IR"))

if tab == "MS":
    st.title("MS-модель (Mass Spectrometry Prediction)")
    smiles_input = st.text_area("Введите SMILES (по одному в строке)", value="CCO\nCCN\nCCC")
    input_list = [s.strip() for s in smiles_input.splitlines() if s.strip()]

elif tab == "IR":
    st.title("IR-модель (FTIR Prediction)")

    # Получаем список материалов с backend
    try:
        resp = requests.get(f"{API_URL}/materials", timeout=5)
        resp.raise_for_status()
        materials = resp.json().get("materials", [])
    except requests.RequestException:
        materials = []
        st.warning("Не удалось загрузить список материалов с сервера.")

    if materials:
        material_input = st.selectbox("Выберите материал", [""] + materials)
    else:
        material_input = st.text_input("Введите название материала (например ABS)")

    input_list = [material_input.strip()] if material_input.strip() else []

if st.button("Predict"):
    if not input_list:  # pylint: disable=possibly-used-before-assignment
        st.warning("Введите данные для предсказания")
    else:
        for item in input_list:
            st.subheader(f"{'SMILES' if tab == 'MS' else 'Материал'}: {item}")
            with st.spinner(f"Предсказание спектра для {item}..."):
                try:
                    if tab == "MS":
                        resp = requests.post(f"{API_URL}/predict", json={"smiles": item}, timeout=10)
                    else:
                        resp = requests.post(f"{API_URL}/predict_ir", json={"material_name": item}, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    pred = data.get("pred_vector", [])
                    if not pred:
                        st.warning("Не удалось предсказать спектр для этого значения")
                        continue

                    # Волновые числа или m/z
                    if tab == "MS":
                        x_bins = list(range(1001))
                        xlabel = "m/z"
                    else:
                        x_bins = list(range(650, 4001))  # IR спектр
                        xlabel = "cm⁻¹"

                    df = pd.DataFrame({"x": x_bins[: len(pred)], "intensity": [float(max(i, 0)) for i in pred]})

                    chart = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(
                            x=alt.X("x:Q", scale=alt.Scale(domain=[x_bins[0], x_bins[-1]]), title=xlabel),
                            y=alt.Y("intensity:Q", title="Intensity"),
                        )
                        .properties(width=800, height=300)
                    )

                    st.altair_chart(chart, use_container_width=True)

                except requests.RequestException as e:
                    st.error(f"Ошибка запроса к API: {e}")
