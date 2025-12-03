# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Predicción: 3 enfermedades", layout="wide")

MODEL_FOLDER = "models"
# Lista automática de modelos con extensión .joblib
def find_models():
    m = {}
    for f in os.listdir(MODEL_FOLDER):
        if f.endswith(".joblib"):
            name = f.replace(".joblib","").replace("_model","")
            m[name] = os.path.join(MODEL_FOLDER, f)
    return m

models_map = find_models()
if not models_map:
    st.error("No se encontraron modelos en carpeta 'models/'. Añade tus .joblib y recarga.")
    st.stop()

st.title("App: Probabilidades de enfermedades (Diabetes / Hipertensión / Obesidad)")
st.write("Carga un CSV con las **features ya preprocesadas** (mismas columnas que se usaron en entrenamiento) o introduce una fila manualmente.")

col1, col2 = st.columns([2,1])
with col1:
    mode = st.radio("Modo", ["Carga CSV (batch)","Manual (single)"])
    if mode == "Carga CSV (batch)":
        uploaded = st.file_uploader("Sube CSV con features (sin columna target)", type=["csv"])
        if uploaded:
            X = pd.read_csv(uploaded)
            st.success(f"CSV cargado: {X.shape}")
            st.dataframe(X.head())
        else:
            X = None
    else:
        st.write("Introduce los valores (las columnas deben coincidir con las del entrenamiento).")
        # mostramos inputs dinámicos si existe un ejemplo en data/
        example_path = "data/example_features.csv"
        if os.path.exists(example_path):
            example = pd.read_csv(example_path).iloc[0:1]
            inputs = {}
            for c in example.columns:
                val = st.text_input(c, value=str(example.iloc[0][c]))
                inputs[c] = [float(val) if val.replace('.','',1).lstrip('-').isdigit() else val]
            X = pd.DataFrame(inputs)
        else:
            st.info("No hay ejemplo disponible - sube un CSV o crea 'data/example_features.csv' para usar el modo manual.")
            X = None

with col2:
    st.subheader("Modelos detectados")
    for k in models_map.keys():
        st.write("-", k)

def load_and_predict(X):
    results = pd.DataFrame(index=X.index)
    for name, path in models_map.items():
        model = joblib.load(path)
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                if probs.shape[1] >= 2:
                    p = probs[:,1]
                else:
                    p = probs[:, -1]
            else:
                p = model.predict(X)
                p = np.array(p).astype(float)
            results[f"p_{name}"] = p
        except Exception as e:
            st.error(f"Error al predecir con {name}: {e}")
            results[f"p_{name}"] = np.nan
    # prob al menos una
    pcols = [c for c in results.columns if c.startswith("p_")]
    if pcols:
        results["p_al_menos_una"] = 1 - np.prod(1 - results[pcols], axis=1)
    return results

if st.button("Predecir") :
    if X is None:
        st.error("No hay datos de entrada.")
    else:
        with st.spinner("Prediciendo..."):
            res = load_and_predict(X)
        df_show = pd.concat([X.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
        st.dataframe(df_show.head(200))
        st.download_button("Descargar resultados (CSV)", df_show.to_csv(index=False).encode('utf-8'), file_name="predicciones.csv")