# build_preprocessors.py
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

models_info = {
    "diabetes": "data/diabetes_train.csv",
    "hipertension": "data/hipertension_train.csv",
    "obesidad": "data/obesidad_train.csv",
}

os.makedirs("models", exist_ok=True)

for name, path in models_info.items():
    print(f"Procesando {name} desde {path} ...")
    df = pd.read_csv(path)
    # Asumimos que la columna objetivo no está en X (si está, eliminarla)
    # Si tu CSV tiene columna "target" cambia / elimina:
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    # detect columns
    num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    print(" num_cols:", num_cols)
    print(" cat_cols:", cat_cols)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    preproc.fit(df)
    joblib.dump({
        "preprocessor": preproc,
        "num_features": num_cols,
        "cat_features": cat_cols,
    }, f"models/{name}_preproc.joblib")
    print(f"Guardado models/{name}_preproc.joblib")
