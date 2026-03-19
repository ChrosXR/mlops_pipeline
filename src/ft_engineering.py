"""
ft_engineering.py
Preparación de datos SIN leakage (versión final)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preparar_features(df):

    df = df.copy()

    # =============================
    # Limpieza de nulos
    # =============================
    nulos_reales = ['', ' ', '-', 'nan', 'NaN', 'NULL', 'None', 'N/A']
    df.replace(nulos_reales, np.nan, inplace=True)

    # =============================
    # Target
    # =============================
    df = df.dropna(subset=["Pago_atiempo"])
    df["Pago_atiempo"] = df["Pago_atiempo"].astype(int)

    # =============================
    # Eliminar leakage
    # =============================
    df.drop(columns=[
        'puntaje',
        'saldo_mora', 'saldo_mora_codeudor',
        'saldo_total', 'saldo_principal'
    ], errors='ignore', inplace=True)

    # =============================
    # Normalización categórica
    # =============================
    if "tendencia_ingresos" in df.columns:
        mapping = {
            'creciente': 'Creciente',
            'estable': 'Estable',
            'decreciente': 'Decreciente'
        }
        df["tendencia_ingresos"] = (
            df["tendencia_ingresos"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )

    # =============================
    # Feature Engineering
    # =============================
    eps = 1e-6

    df['ratio_cuota_ingreso'] = df['cuota_pactada'] / (df['salario_cliente'] + eps)
    df['ratio_deuda_ingreso'] = df['total_otros_prestamos'] / (df['salario_cliente'] + eps)

    # =============================
    # Fecha controlada
    # =============================
    if 'fecha_prestamo' in df.columns:
        fecha_corte = pd.Timestamp("2025-01-01")
        df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'], errors='coerce')
        df['antiguedad_prestamo'] = (fecha_corte - df['fecha_prestamo']).dt.days
        df.drop(columns=['fecha_prestamo'], inplace=True)

    # =============================
    # Missing flags
    # =============================
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[f'{col}_missing'] = df[col].isna().astype(int)

    # =============================
    # Split
    # =============================
    X = df.drop(columns=['Pago_atiempo'])
    y = df['Pago_atiempo']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test