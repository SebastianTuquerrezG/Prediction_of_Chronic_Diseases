# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Predicción: Diabetes / Hipertensión / Obesidad", layout="wide")

MODEL_FOLDER = "models"

# ----- Definiciones de columnas por modelo -----
DIABETES_COLS = [
 "HighChol","CholCheck","BMI","Smoker","HeartDiseaseorAttack","PhysActivity",
 "Fruits","Veggies","HvyAlcoholConsump","GenHlth","MentHlth","PhysHlth","DiffWalk",
 "Stroke","HighBP","Age_mapped","sex_female","sex_male","BMI_flag_high"
]

HIPERTENSION_COLS = [
 "age","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak",
 "slope","ca","thal","sex_female","sex_male"
]

OBESIDAD_COLS = [
 "Age","Height","Weight","family_history_with_overweight","FAVC","FCVC","NCP",
 "CAEC","SMOKE","CH2O","SCC","FAF","TUE","CALC",
 "Gender_std_female","Gender_std_male",
 "MTRANS_std_automobile","MTRANS_std_bike","MTRANS_std_motorbike",
 "MTRANS_std_public_transportation","MTRANS_std_walking"
]

# Map models to joblib filenames
MODEL_FILES = {
    "diabetes": os.path.join(MODEL_FOLDER, "diabetes_model.joblib"),
    "hipertension": os.path.join(MODEL_FOLDER, "hypertension_model.joblib"),
    "obesidad": os.path.join(MODEL_FOLDER, "obesity_model.joblib"),
}

# Load pipelines
models = {}
for k, path in MODEL_FILES.items():
    if os.path.exists(path):
        try:
            models[k] = joblib.load(path)
        except Exception as e:
            st.error(f"Error cargando {k} desde {path}: {e}")
    else:
        st.warning(f"No se encontró el archivo para {k}: {path}")

st.title("Interfaz manual por modelo — Predicción de 3 enfermedades")
st.write("Introduce valores por campo para cada modelo. Los pipelines ya contienen sus transformadores y esperan las columnas que se muestran. Para campos one-hot (sexo, transporte) proporciona la opción legible y la app creará las columnas necesarias.")

tab_general, tab_diab, tab_hip, tab_ob = st.tabs(["General / Batch","Diabetes","Hipertensión","Obesidad"])

# Helper para checkboxes MTRANS
MTRANS_OPTIONS = ["automobile","bike","motorbike","public_transportation","walking"]

def ensure_columns(df, required_cols, fill_value=0):
    """Asegura que df contenga todas las required_cols (agrega columnas faltantes con fill_value)."""
    for c in required_cols:
        if c not in df.columns:
            df[c] = fill_value
    return df[required_cols]

# ---------------- TAB: General / Batch ----------------
with tab_general:
    st.header("Predicción por lotes (CSV) o resumen")
    uploaded = st.file_uploader("Subir CSV (batch) — columnas ya preprocesadas según sea necesario", type=["csv"], key="batch_uploader")
    selected_models = st.multiselect("Seleccionar modelos a usar (batch)", options=list(models.keys()), default=list(models.keys()), key="batch_select_models")
    if st.button("Predecir CSV (batch)", key="batch_predict_btn"):
        if uploaded is None:
            st.error("Sube un CSV para predicción por lote.")
        else:
            X_all = pd.read_csv(uploaded)
            out_dfs = []
            for mname in selected_models:
                if mname not in models:
                    st.warning(f"Modelo {mname} no cargado — se omite.")
                    continue
                req_cols = DIABETES_COLS if mname=="diabetes" else (HIPERTENSION_COLS if mname=="hipertension" else OBESIDAD_COLS)
                missing = [c for c in req_cols if c not in X_all.columns]
                if missing:
                    st.error(f"Para el modelo {mname} faltan columnas en el CSV: {missing}. No se puede predecir este modelo.")
                    continue
                X_use = X_all[req_cols].copy()
                try:
                    proba = models[mname].predict_proba(X_use)
                    p = proba[:,1] if proba.shape[1] >= 2 else proba[:,-1]
                    out_dfs.append(pd.Series(p, name=f"p_{mname}"))
                except Exception as e:
                    st.error(f"Error prediciendo con {mname}: {e}")
            if out_dfs:
                df_out = pd.concat(out_dfs, axis=1)
                df_out["p_al_menos_una"] = 1 - np.prod(1 - df_out.values, axis=1)
                res_df = pd.concat([X_all.reset_index(drop=True), df_out.reset_index(drop=True)], axis=1)
                st.success("Predicción completada — primeras filas:")
                st.dataframe(res_df.head(100))
                st.download_button("Descargar resultados (CSV)", res_df.to_csv(index=False).encode('utf-8'), file_name="predicciones_batch.csv", key="download_batch")
            else:
                st.error("No se pudieron predecir con ninguno de los modelos seleccionados.")

# ---------------- TAB: Diabetes ----------------
with tab_diab:
    st.header("Diabetes — ingresar valores manualmente")
    cols = st.columns(3)
    with cols[0]:
        Age_mapped = st.number_input("Age_mapped (edad mapeada numericamente)", min_value=0.0, max_value=120.0, value=40.0, step=1.0, key="diab_Age_mapped")
        HighChol = st.selectbox("HighChol (1=sí,0=no)", options=[1,0], index=1, key="diab_HighChol")
        CholCheck = st.selectbox("CholCheck (1=yes cholesterol check in 5 years)", options=[1,0], index=1, key="diab_CholCheck")
        BMI = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=200.0, value=25.0, step=0.1, key="diab_BMI")
    with cols[1]:
        Smoker = st.selectbox("Smoker (1=si,0=no)", options=[1,0], index=1, key="diab_Smoker")
        HeartDiseaseorAttack = st.selectbox("HeartDiseaseorAttack (1=si,0=no)", options=[1,0], index=0, key="diab_HeartDiseaseorAttack")
        PhysActivity = st.selectbox("PhysActivity (1=si,0=no)", options=[1,0], index=1, key="diab_PhysActivity")
        Fruits = st.selectbox("Fruits (1=si,0=no)", options=[1,0], index=1, key="diab_Fruits")
        Veggies = st.selectbox("Veggies (1=si,0=no)", options=[1,0], index=1, key="diab_Veggies")
    with cols[2]:
        HvyAlcoholConsump = st.selectbox("HvyAlcoholConsump (1=si,0=no)", options=[1,0], index=0, key="diab_HvyAlcoholConsump")
        GenHlth = st.slider("GenHlth (1-excelente .. 5-pobre)", min_value=1, max_value=5, value=3, key="diab_GenHlth")
        MentHlth = st.number_input("MentHlth (días mala salud mental en último mes)", min_value=0, max_value=30, value=0, key="diab_MentHlth")
        PhysHlth = st.number_input("PhysHlth (días mala salud física en último mes)", min_value=0, max_value=30, value=0, key="diab_PhysHlth")
        DiffWalk = st.selectbox("DiffWalk (1=si,0=no)", options=[1,0], index=0, key="diab_DiffWalk")
    col4, col5 = st.columns(2)
    with col4:
        Stroke = st.selectbox("Stroke (1=si,0=no)", options=[1,0], index=0, key="diab_Stroke")
        HighBP = st.selectbox("HighBP (1=si,0=no)", options=[1,0], index=0, key="diab_HighBP")
    with col5:
        sex = st.selectbox("Sexo", options=["male","female"], index=0, key="diab_sex")
        # encode sex to two columns
        sex_female = 1 if sex=="female" else 0
        sex_male = 1 if sex=="male" else 0

    BMI_flag_high = 1 if BMI > 60 else 0

    if st.button("Predecir Diabetes"):
        # crear df con columnas guardadas por pipeline
        row = {
            "HighChol": HighChol, "CholCheck": CholCheck, "BMI": BMI, "Smoker": Smoker,
            "HeartDiseaseorAttack": HeartDiseaseorAttack, "PhysActivity": PhysActivity,
            "Fruits": Fruits, "Veggies": Veggies, "HvyAlcoholConsump": HvyAlcoholConsump,
            "GenHlth": GenHlth, "MentHlth": MentHlth, "PhysHlth": PhysHlth,
            "DiffWalk": DiffWalk, "Stroke": Stroke, "HighBP": HighBP,
            "Age_mapped": Age_mapped, "sex_female": sex_female, "sex_male": sex_male,
            "BMI_flag_high": BMI_flag_high
        }
        X = pd.DataFrame([row])
        # ensure columns order
        X = ensure_columns(X, DIABETES_COLS)
        try:
            model = models.get("diabetes")
            proba = model.predict_proba(X)
            p = proba[:,1] if proba.shape[1] >= 2 else proba[:,-1]
            st.metric("Probabilidad Diabetes", f"{p[0]:.4f}")
        except Exception as e:
            st.error(f"Error al predecir Diabetes: {e}")

# ---------------- TAB: Hipertensión ----------------
with tab_hip:
    st.header("Hipertensión — ingresar valores manualmente")
    cols = st.columns(3)
    with cols[0]:
        age = st.number_input("age (años)", min_value=0, max_value=120, value=50, step=1, key="hip_age")
        cp = st.selectbox("cp (Chest pain type) 0=asymp 1=typical 2=atypical 3=non-anginal", options=[0,1,2,3], index=0, key="hip_cp")
        trestbps = st.number_input("trestbps (mm Hg)", min_value=0, max_value=300, value=120, key="hip_trestbps")
        chol = st.number_input("chol (mg/dl)", min_value=0, max_value=1000, value=200, key="hip_chol")
    with cols[1]:
        fbs = st.selectbox("fbs (1 if fasting blood sugar > 120 mg/dl)", options=[1,0], index=1, key="hip_fbs")
        restecg = st.selectbox("restecg (0,1,2)", options=[0,1,2], index=0, key="hip_restecg")
        thalach = st.number_input("thalach (max heart rate achieved)", min_value=0, max_value=300, value=150, key="hip_thalach")
    with cols[2]:
        exang = st.selectbox("exang (exercise induced angina) 1=yes 0=no", options=[1,0], index=0, key="hip_exang")
        oldpeak = st.number_input("oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="hip_oldpeak")
        slope = st.selectbox("slope (0 upsloping,1 flat,2 downsloping)", options=[0,1,2], index=1, key="hip_slope")
        ca = st.selectbox("ca (number of major vessels 0-3)", options=[0,1,2,3], index=0, key="hip_ca")
        thal = st.selectbox("thal (3 normal,6 fixed,7 reversible)", options=[3,6,7], index=0, key="hip_thal")
    sex_hip = st.selectbox("Sexo", options=["male","female"], index=0, key="hip_sex")
    sex_female_hip = 1 if sex_hip=="female" else 0
    sex_male_hip = 1 if sex_hip=="male" else 0

    if st.button("Predecir Hipertensión"):
        row = {
            "age": age, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
            "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
            "slope": slope, "ca": ca, "thal": thal, "sex_female": sex_female_hip, "sex_male": sex_male_hip
        }
        X = pd.DataFrame([row])
        X = ensure_columns(X, HIPERTENSION_COLS)
        try:
            model = models.get("hipertension")
            proba = model.predict_proba(X)
            p = proba[:,1] if proba.shape[1] >= 2 else proba[:,-1]
            st.metric("Probabilidad Hipertensión", f"{p[0]:.4f}")
        except Exception as e:
            st.error(f"Error al predecir Hipertensión: {e}")

# ---------------- TAB: Obesidad ----------------
with tab_ob:
    st.header("Obesidad — ingresar valores manualmente")
    c1, c2, c3 = st.columns(3)
    with c1:
        Age_o = st.number_input("Age", min_value=0, max_value=120, value=30, step=1, key="ob_Age")
        Height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170, key="ob_Height")
        Weight = st.number_input("Weight (kg)", min_value=10, max_value=400, value=70, key="ob_Weight")
        family_history_with_overweight = st.selectbox("family_history_with_overweight (1=si,0=no)", options=[1,0], index=0, key="ob_family_history")
    with c2:
        FAVC = st.selectbox("FAVC (frequent high-caloric food) (1=si,0=no)", options=[1,0], index=0, key="ob_FAVC")
        FCVC = st.selectbox("FCVC (freq veg) 1-3", options=[1,2,3], index=1, key="ob_FCVC")
        NCP = st.selectbox("NCP (n° main meals) 1-4", options=[1,2,3,4], index=2, key="ob_NCP")
        CAEC = st.selectbox("CAEC (snacking) 1-3", options=[1,2,3], index=0, key="ob_CAEC")
    with c3:
        SMOKE = st.selectbox("SMOKE (1=si,0=no)", options=[1,0], index=0, key="ob_SMOKE")
        CH2O = st.selectbox("CH2O (water consumption 1-3)", options=[1,2,3], index=1, key="ob_CH2O")
        SCC = st.selectbox("SCC (caloric beverages 1=yes,0=no)", options=[1,0], index=0, key="ob_SCC")
        FAF = st.selectbox("FAF (physical activity 0-3)", options=[0,1,2,3], index=1, key="ob_FAF")
        TUE = st.selectbox("TUE (time devices 0-2)", options=[0,1,2], index=1, key="ob_TUE")
        CALC = st.selectbox("CALC (alcohol 0-2)", options=[0,1,2], index=0, key="ob_CALC")

    Gender = st.selectbox("Gender", options=["male","female"], index=0, key="ob_Gender")
    Gender_std_female = 1 if Gender=="female" else 0
    Gender_std_male = 1 if Gender=="male" else 0

    st.markdown("**Modo transporte** — selecciona los que aplica (se crean columnas one-hot):")
    mtrans_checks = {}
    col_trans = st.columns(2)
    for i, opt in enumerate(MTRANS_OPTIONS):
        mtrans_checks[opt] = st.checkbox(opt, value=False, key=f"mtrans_{opt}")

    if st.button("Predecir Obesidad"):
        row = {
            "Age": Age_o, "Height": Height, "Weight": Weight,
            "family_history_with_overweight": family_history_with_overweight,
            "FAVC": FAVC, "FCVC": FCVC, "NCP": NCP, "CAEC": CAEC,
            "SMOKE": SMOKE, "CH2O": CH2O, "SCC": SCC, "FAF": FAF,
            "TUE": TUE, "CALC": CALC,
            "Gender_std_female": Gender_std_female, "Gender_std_male": Gender_std_male,
            "MTRANS_std_automobile": 1 if mtrans_checks["automobile"] else 0,
            "MTRANS_std_bike": 1 if mtrans_checks["bike"] else 0,
            "MTRANS_std_motorbike": 1 if mtrans_checks["motorbike"] else 0,
            "MTRANS_std_public_transportation": 1 if mtrans_checks["public_transportation"] else 0,
            "MTRANS_std_walking": 1 if mtrans_checks["walking"] else 0
        }
        X = pd.DataFrame([row])
        X = ensure_columns(X, OBESIDAD_COLS)
        try:
            model = models.get("obesidad")
            proba = model.predict_proba(X)
            p = proba[:,1] if proba.shape[1] >= 2 else proba[:,-1]
            st.metric("Probabilidad Obesidad", f"{p[0]:.4f}")
        except Exception as e:
            st.error(f"Error al predecir Obesidad: {e}")

# ---------------- Footer: combinar manuales si el usuario quiere ----------------
st.markdown("---")
st.header("Combinar predicciones (si ingresaste manualmente para los 3 modelos)")
st.write("Si has calculado ya las 3 probabilidades manualmente y quieres la probabilidad combinada, ingresa los 3 valores (0-1) aquí.")
p1 = st.number_input("p_diabetes (0-1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
p2 = st.number_input("p_hipertension (0-1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
p3 = st.number_input("p_obesidad (0-1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
if st.button("Calcular p_al_menos_una"):
    p_comb = 1 - (1-p1)*(1-p2)*(1-p3)
    st.metric("Probabilidad al menos una", f"{p_comb:.4f}")
    st.write("Nota: fórmula asume independencia entre eventos. Interpreta con cautela si los outputs del pipeline están correlacionados.")