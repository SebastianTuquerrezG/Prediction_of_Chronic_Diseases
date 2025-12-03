import joblib
import os

model_paths = {
    "diabetes": "models/diabetes_model.joblib",
    "hipertension": "models/hypertension_model.joblib",
    "obesidad": "models/obesity_model.joblib",
}

for name, path in model_paths.items():
    print("="*50)
    print("MODELO:", name)

    if not os.path.exists(path):
        print("❌ No existe:", path)
        continue

    model = joblib.load(path)
    print("Tipo de objeto:", type(model))

    if hasattr(model, "feature_names_in_"):
        print("✅ Este modelo SÍ guarda las columnas:")
        print(list(model.feature_names_in_))
    else:
        print("⚠️ Este modelo NO guarda feature_names_in_")
