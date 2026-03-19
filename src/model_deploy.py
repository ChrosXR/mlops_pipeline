"""
model_deploy.py
API de despliegue del modelo con FastAPI
"""

import pandas as pd
import numpy as np
import os
import joblib

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# =============================
# PATHS
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.txt")

# =============================
# CARGA DEL MODELO
# =============================
print("Cargando modelo...")
model = joblib.load(MODEL_PATH)
print("Modelo cargado correctamente")

# =============================
# THRESHOLD
# =============================
try:
    with open(THRESHOLD_PATH, "r") as f:
        THRESHOLD = float(f.read())
except:
    THRESHOLD = 0.5

# =============================
# FASTAPI
# =============================
app = FastAPI(
    title="API - Predicción de Pago de Créditos",
    description="Modelo MLOps para evaluar riesgo crediticio",
    version="1.1"
)

# =============================
# SCHEMA CON EJEMPLO
# =============================
class Cliente(BaseModel):
    edad_cliente: float
    salario_cliente: float
    cuota_pactada: float
    total_otros_prestamos: float
    tendencia_ingresos: str

    class Config:
        schema_extra = {
            "example": {
                "edad_cliente": 35,
                "salario_cliente": 2500,
                "cuota_pactada": 300,
                "total_otros_prestamos": 500,
                "tendencia_ingresos": "Estable"
            }
        }

class InputData(BaseModel):
    data: List[Cliente]

# =============================
# FEATURE ENGINEERING
# =============================
def preparar_features_api(df):
    df = df.copy()
    eps = 1e-6

    df['ratio_cuota_ingreso'] = df['cuota_pactada'] / (df['salario_cliente'] + eps)
    df['ratio_deuda_ingreso'] = df['total_otros_prestamos'] / (df['salario_cliente'] + eps)

    mapping = {
        'creciente': 'Creciente',
        'estable': 'Estable',
        'decreciente': 'Decreciente'
    }

    if "tendencia_ingresos" in df.columns:
        df["tendencia_ingresos"] = (
            df["tendencia_ingresos"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )

    return df

# =============================
# HEALTH CHECK
# =============================
@app.get("/")
def home():
    return {
        "status": "ok",
        "mensaje": "API de scoring crediticio activa"
    }

# =============================
# PREDICT (MEJORADO)
# =============================
@app.post("/predict")
def predict(input_data: InputData):
    try:
        df = pd.DataFrame([item.dict() for item in input_data.data])

        df = preparar_features_api(df)

        # Alinear columnas
        expected_cols = model.feature_names_in_

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_cols]

        # Predicción
        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)

        # Interpretación de negocio
        resultado = []
        for i in range(len(preds)):

            if preds[i] == 1:
                decision = "Cliente probablemente pagará a tiempo"
                riesgo = "Bajo"
            else:
                decision = "Cliente con riesgo de NO pagar"
                riesgo = "Alto"

            resultado.append({
                "prediccion": int(preds[i]),
                "probabilidad_pago": float(probs[i]),
                "riesgo": riesgo,
                "interpretacion": decision
            })

        return {
            "threshold_usado": THRESHOLD,
            "resultados": resultado
        }

    except Exception as e:
        return {"error": str(e)}

# =============================
# EXPLAIN (MEJORADO)
# =============================
@app.post("/explain")
def explain(input_data: InputData):
    try:
        df = pd.DataFrame([item.dict() for item in input_data.data])
        df = preparar_features_api(df)

        expected_cols = model.feature_names_in_

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_cols]

        model_step = model.named_steps["model"]

        # Detectar tipo de modelo
        if hasattr(model_step, "coef_"):
            importance_values = model_step.coef_[0]
        elif hasattr(model_step, "feature_importances_"):
            importance_values = model_step.feature_importances_
        else:
            return {"error": "Modelo no soporta interpretabilidad"}

        importance = sorted(
            zip(expected_cols, importance_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]

        return {
            "mensaje": "Top variables que más influyen en la predicción",
            "top_features": [
                {
                    "feature": f,
                    "impacto": float(v),
                    "interpretacion": "Aumenta riesgo" if v < 0 else "Disminuye riesgo"
                }
                for f, v in importance
            ]
        }

    except Exception as e:
        return {"error": str(e)}