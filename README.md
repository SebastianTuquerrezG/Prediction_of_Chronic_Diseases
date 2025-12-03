# Health Risk Prediction — Diabetes, Hypertension & Obesity  
**Proyecto:** Detección y priorización de pacientes en riesgo mediante modelos de machine learning  
**Metodología:** CRISP-DM  
**Autor:** Joan Sebastián Tuquerrez Gomez

---

## 🩺 Resumen del proyecto  
Este repositorio contiene el desarrollo completo de un sistema de **predicción de riesgo** para tres enfermedades crónicas: **Diabetes, Hipertensión y Obesidad**.  \
A partir de vgariables clinicas y habitos de vida.
El objetivo es ofrecer una herramienta basada en machine learning que permita a instituciones de salud **priorizar pacientes**, apoyar decisiones clínicas y promover intervenciones tempranas.

La metodología empleada fue **CRISP-DM**, abordando todas sus fases:  
Entendimiento del negocio → Entendimiento de datos → Preparacion de datos → modelado → evaluación → despliegue.

---

## 📂 Contenido del repositorio  
Este repo incluye:

- Scripts y notebooks de:
  - Limpieza y preparación de datos  
  - Exploración y análisis estadístico  
  - Entrenamiento y comparación de modelos  
  - Evaluación (ROC, F1, precision-recall, bootstrap CI, calibración)
- Modelos entrenados listos para uso (`*.joblib`)
- Prototipo de despliegue usando Streamlit (`app.py`)
- Datos procesados (o ruta a ellos)
- Reportes y documentación generada

---

## 🎯 Objetivo del negocio  
- **Problema:** Identificar pacientes con riesgo aumentado de Diabetes, Hipertensión u Obesidad usando información clínica básica.  
- **Meta:** Permitir una detección temprana y una priorización más eficiente de pacientes.  
- **Requerimientos clave:**  
  - Probabilidades interpretables  
  - Explicabilidad por predicción (e.g., SHAP)  
  - Alto recall para minimizar falsos negativos  
  - Prevención de data leakage  

---

## 🗂 Dataset  
Se trabajó con tres conjuntos de datos (uno por enfermedad):

- Datasets ya divididos en:
  - `X_train_*`
  - `X_test_*`
  - `y_train_*`
  - `y_test_*`
- Archivos en formato CSV con features clínicos y hábitos de vida.

### Principales pasos de preparación:

✔ Imputación  
✔ Encoding  
✔ Normalización / tratamiento de outliers  
✔ **Eliminación de variables con fuga (Obesidad)**  
✔ **SMOTE para balanceo (Hipertensión)**  
✔ Creación de pipelines reproducibles  

---

## 🤖 Modelado  
Se probaron diferentes algoritmos:

- **Logistic Regression** (baseline)  
- **Decision Tree** (interpretable)  
- **XGBoost** (modelo final seleccionado)

### Hiperparámetros del modelo final (XGBoost)
```text
n_estimators=150
learning_rate=0.07
max_depth=2
subsample=0.5
colsample_bytree=0.5
reg_alpha=3
reg_lambda=8
min_child_weight=6
eval_metric='logloss'
base_score=0.5
