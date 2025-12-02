# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Optional: shap (heavy). We'll try to use it if available and if background data exists.
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ----------------------------
# Settings
# ----------------------------
MODEL_FILES = {
    "diabetes": "diabetes_model.joblib",
    "hypertension": "hypertension_model.joblib",
    "obesity": "obesity_model.joblib"
}

FEATURES_FILE = "feature_names.json"   # optional: list of features saved at training time
LOG_FILE = "prediction_logs.csv"
BACKGROUND_SAMPLE = "background_sample.csv"  # optional small sample for SHAP

st.set_page_config(page_title="Health Risk Scoring", layout="wide")

# ----------------------------
# Helpers: model loading
# ----------------------------
@st.cache_resource
def load_models(model_files):
    models = {}
    for k, f in model_files.items():
        if os.path.exists(f):
            models[k] = joblib.load(f)
        else:
            models[k] = None
    return models

@st.cache_data
def load_feature_names():
    if os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, "r") as fh:
            return json.load(fh)
    return None

def ensure_columns_match(df, expected_cols):
    """Return (ok, missing, extra)"""
    if expected_cols is None:
        return None, [], list(df.columns)
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]
    ok = len(missing) == 0
    return ok, missing, extra

def predict_df(model_pipeline, df):
    """Given the saved Pipeline (prep + clf), return probabilities and preds"""
    proba = model_pipeline.predict_proba(df)[:, 1]
    pred = model_pipeline.predict(df)
    return pred, proba

def save_log(df_log):
    header = not os.path.exists(LOG_FILE)
    df_log.to_csv(LOG_FILE, mode="a", header=header, index=False)

# ----------------------------
# Main UI
# ----------------------------
st.title("Sistema de Puntaje de Riesgo — Demo Streamlit")
st.markdown("Carga un archivo CSV con las mismas columnas usadas en entrenamiento o sube un caso (1 fila). "
            "La columna target no debe estar en el CSV. Asegúrate de que los nombres de columnas coincidan.")

models = load_models(MODEL_FILES)
feature_names = load_feature_names()

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Configuración")
    disease = st.selectbox("Seleccionar enfermedad", ["hypertension", "diabetes", "obesity"])
    mode = st.radio("Modo", ["Batch (CSV)", "Single case (CSV row)"])
    st.write("Model available:", "Sí" if models[disease] is not None else "No — carga el .joblib en este directorio")

    # threshold (optional) - you can adjust based on cost analysis
    default_threshold = 0.5
    threshold = st.slider("Umbral (probabilidad) para clasificar como positivo", 0.0, 1.0, default_threshold, 0.01)

with col2:
    st.header("Input / Scoring")

    uploaded_file = st.file_uploader("Subir CSV (X features) — para Batch o Single (1 fila)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file)
            st.success(f"Archivo cargado con {df_in.shape[0]} filas y {df_in.shape[1]} columnas.")
            st.dataframe(df_in.head())
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            st.stop()
    else:
        df_in = None

    # If no feature list found, show hint
    if feature_names is None:
        st.info("No se encontró feature_names.json. Asegúrate de que tu CSV tenga las mismas columnas que usaste para entrenar.")
    else:
        st.write(f"Se esperan {len(feature_names)} features. Ej: {', '.join(feature_names[:10])} ...")

    # Button to score
    if st.button("Ejecutar scoring"):

        if models[disease] is None:
            st.error("Modelo no disponible. Coloca el archivo .joblib correspondiente en el directorio.")
        elif df_in is None:
            st.error("Sube un CSV primero.")
        else:
            # Check columns
            ok, missing, extra = ensure_columns_match(df_in, feature_names)
            if feature_names is not None and not ok:
                st.warning(f"Columnas faltantes: {missing}. Columnas extra en CSV: {extra}. Asegúrate de que coincidan.")
                # but allow to proceed? we'll stop to be safe
                st.stop()

            # Predict
            model_pipeline = models[disease]
            try:
                preds, probas = predict_df(model_pipeline, df_in)
            except Exception as e:
                st.error(f"Error durante predict: {e}")
                st.stop()

            df_out = df_in.copy()
            df_out["pred"] = preds
            df_out["proba"] = probas
            df_out["pred_threshold"] = (df_out["proba"] >= threshold).astype(int)

            st.success(f"Scoring completado — filas: {df_out.shape[0]}")
            st.dataframe(df_out[["proba", "pred", "pred_threshold"]].head(20))

            # Download
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar resultados (CSV)", data=csv, file_name=f"predictions_{disease}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

            # Log predictions
            log_cols = ["timestamp", "disease", "proba", "pred", "pred_threshold"]
            df_log = pd.DataFrame({
                "timestamp": datetime.now().isoformat(),
                "disease": disease,
                "proba": df_out["proba"],
                "pred": df_out["pred"],
                "pred_threshold": df_out["pred_threshold"]
            })
            # Expand to rows
            df_log = df_log.explode(["proba", "pred", "pred_threshold"])
            save_log(df_log.reset_index(drop=True))
            st.info(f"Resultados guardados en {LOG_FILE}")

            # SHAP explanation (if available)
            if SHAP_AVAILABLE:
                st.subheader("Explicación local (SHAP) — primeras filas")
                # Need background data to initialize explainer reliably
                background = None
                if os.path.exists(BACKGROUND_SAMPLE):
                    try:
                        background = pd.read_csv(BACKGROUND_SAMPLE)
                    except Exception:
                        background = None

                # prepare preprocessed data if pipeline has prep
                try:
                    prep = model_pipeline.named_steps.get("prep", None)
                    clf = model_pipeline.named_steps.get("clf", None)
                except Exception:
                    prep, clf = None, None

                if clf is None:
                    st.write("No se encontró el clasificador dentro del pipeline para SHAP.")
                else:
                    # get the first row (or N rows)
                    row = df_in.iloc[0:1]
                    try:
                        if prep is not None:
                            X_for_shap = prep.transform(row)
                            if background is not None:
                                bg_for_shap = prep.transform(background)
                                explainer = shap.TreeExplainer(clf, bg_for_shap)
                            else:
                                # small background: take row itself (not ideal)
                                explainer = shap.TreeExplainer(clf)
                            shap_values = explainer.shap_values(X_for_shap)
                            # plot summary as matplotlib
                            fig, ax = plt.subplots(figsize=(6,3))
                            shap.summary_plot(shap_values, X_for_shap, show=False)
                            st.pyplot(fig)
                        else:
                            st.write("El pipeline no tiene 'prep' definido; SHAP requiere preprocesado coherente.")
                    except Exception as e:
                        st.write("SHAP no pudo generarse:", e)
            else:
                st.info("SHAP no disponible. Instala 'shap' para explicaciones globales/locales.")

# Footer
st.markdown("---")
st.markdown("**Notas:** 1) El CSV debe incluir las columnas de features en el mismo orden/nombre usados en entrenamiento. "
            "2) No subas datos sensibles en entornos no seguros.")
