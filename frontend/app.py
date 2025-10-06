"""Frontend-приложение"""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

# Набор настроек для приложения
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL = st.sidebar.text_input("API URL", DEFAULT_API_URL)
st.title("MS Predictor — Multi SMILES Demo")

# Основная логика работы frontend
smiles_input = st.text_area("Введите SMILES (по одному в строке)", value="CCO\nCCN\nCCC")
smiles_list = [s.strip() for s in smiles_input.splitlines() if s.strip()]

if st.button("Predict"):

    if not smiles_list:
        st.warning("Введите хотя бы одну SMILES")
    else:
        for smiles in smiles_list:
            st.subheader(f"SMILES: {smiles}")
            with st.spinner(f"Предсказание спектра для {smiles}..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json={"smiles": smiles}, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()

                    pred = data.get("pred_vector", [])
                    if not pred:
                        st.warning("Не удалось предсказать спектр для этой SMILES")
                        continue

                    # фиксируем mz от 0 до 1000
                    mz_bins = list(range(1001))

                    # DataFrame для Altair
                    df = pd.DataFrame(
                        {
                            "mz": mz_bins[: len(pred)],  # если модель вернула меньше
                            "intensity": [float(max(i, 0.0)) for i in pred],
                        }
                    )

                    chart = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(
                            x=alt.X("mz:Q", scale=alt.Scale(domain=[0, 1000]), title="m/z"),
                            y=alt.Y("intensity:Q", title="Intensity"),
                        )
                        .properties(width=800, height=300)
                    )

                    st.altair_chart(chart, use_container_width=True)

                except requests.RequestException as e:
                    st.error(f"Ошибка запроса к API: {e}")
