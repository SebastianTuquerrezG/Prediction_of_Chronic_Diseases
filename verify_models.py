# verify_models.py
import joblib
import pandas as pd
import numpy as np
import os

# Configura aquí los archivos que tengas
models_info = {
    "diabetes": {
        "joblib": "models/diabetes_model.joblib",
        "train_csv": "data/diabetes_train.csv"   # opcional, para prueba
    },
    "hipertension": {
        "joblib": "models/hypertension_model.joblib",
        "train_csv": "data/hypertension_train.csv"
    },
    "obesidad": {
        "joblib": "models/obesity_model.joblib",
        "train_csv": "data/obesidad_train.csv"
    },
}

def load_model(path):
    print("Cargando", path)
    obj = joblib.load(path)
    print(" tipo:", type(obj))
    # si es sklearn Pipeline:
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(obj, Pipeline):
            print(" -> Es Pipeline con pasos:", obj.named_steps.keys())
    except Exception:
        pass
    return obj

def try_predict_proba(model, X):
    # intenta predict_proba, si no existe intenta predict
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        print(" predict_proba shape:", getattr(probs, "shape", None))
        # si es binario, tomar col 1
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:,1]
        # multiclass fallback: tomar última columna
        return probs[:, -1]
    else:
        print(" predict_proba no disponible, usando predict como fallback")
        preds = model.predict(X)
        # si son 0/1, devolver como float
        return np.array(preds).astype(float)

for key, info in models_info.items():
    jpath = info["joblib"]
    if not os.path.exists(jpath):
        print(f"AVISO: no existe {jpath} para {key}")
        continue
    model = load_model(jpath)

    # si existe CSV de prueba, úsalo; si no, intentamos crear ejemplo
    csvp = info.get("train_csv")
    if csvp and os.path.exists(csvp):
        df = pd.read_csv(csvp)
        # Quitar columna target si existe
        if "target" in df.columns:
            X = df.drop(columns=["target"]).iloc[:5]
        else:
            # si hay una columna objetivo con otro nombre, 
            # aquí solo tomamos las primeras 5 columnas
            X = df.iloc[:5, :]

        print(f" Probando con {X.shape[0]} filas y {X.shape[1]} columnas")
        try:
            p = try_predict_proba(model, X)
            print(" Probabilidades (primeras filas):", p[:5])
        except Exception as e:
            print(" ERROR al predecir:", e)
    else:
        print(" No hay CSV de prueba disponible; intentando predecir con ejemplo dummy si el modelo es Pipeline")
        # si el modelo es Pipeline y conoce columnas, a veces acepta DataFrame vacío: intentamos
        try:
            dummy = pd.DataFrame([ [0]*1 ], columns=[0])
            p = try_predict_proba(model, dummy)
            print(" Resultado dummy:", p)
        except Exception as e:
            print(" No fue posible probar automáticamente:", e)

print("FIN verificación.")
