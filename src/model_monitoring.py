"""
model_monitoring.py
Monitoreo de Data Drift
"""

import pandas as pd
import numpy as np

from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

from cargar_datos import cargar_datos
from ft_engineering import preparar_features


# =============================
# PSI (Population Stability Index)
# =============================
def calcular_psi(expected, actual, bins=10):

    # Convertimos a series y eliminamos NaN
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    # Validación: evitar columnas vacías
    if len(expected) < 10 or len(actual) < 10:
        return np.nan

    # Definir cortes por cuantiles (baseline)
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.quantile(expected, quantiles)

    # Contar frecuencias
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convertir a proporciones
    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    # Calcular PSI (con epsilon para evitar log(0))
    psi = np.sum(
        (actual_perc - expected_perc) *
        np.log((actual_perc + 1e-6) / (expected_perc + 1e-6))
    )

    return psi


# =============================
# KS Test
# =============================
def calcular_ks(expected, actual):

    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    # Validación
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    stat, _ = ks_2samp(expected, actual)
    return stat


# =============================
# Jensen-Shannon Divergence
# =============================
def calcular_js(expected, actual, bins=10):

    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    # Validación
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Histogramas
    expected_hist, bins = np.histogram(expected, bins=bins, density=True)
    actual_hist, _ = np.histogram(actual, bins=bins, density=True)

    # Evitar ceros
    expected_hist += 1e-6
    actual_hist += 1e-6

    return jensenshannon(expected_hist, actual_hist)


# =============================
# Chi-cuadrado robusto (categóricas)
# =============================
def calcular_chi2(expected, actual):

    # Conteo de categorías
    expected_counts = expected.value_counts()
    actual_counts = actual.value_counts()

    # Unificar categorías
    all_cats = set(expected_counts.index).union(set(actual_counts.index))

    expected_arr = np.array([expected_counts.get(cat, 0) for cat in all_cats])
    actual_arr = np.array([actual_counts.get(cat, 0) for cat in all_cats])

    # Normalización para evitar errores de suma
    expected_arr = expected_arr / expected_arr.sum()
    actual_arr = actual_arr / actual_arr.sum()

    # Cálculo manual de chi2 (más estable)
    chi2 = np.sum((actual_arr - expected_arr) ** 2 / (expected_arr + 1e-6))

    return chi2


# =============================
# Clasificación de riesgo
# =============================
def evaluar_riesgo(psi):

    if pd.isna(psi):
        return "⚪ Sin datos"

    if psi < 0.1:
        return "🟢 Bajo"
    elif psi < 0.25:
        return "🟡 Medio"
    else:
        return "🔴 Alto"


# =============================
# Función principal de monitoreo
# =============================
def monitorear_drift():

    print("📥 Cargando datos...")
    df = cargar_datos()

    # Aplicar feature engineering
    X_train, X_test, y_train, y_test = preparar_features(df)

    # Simulación:
    # histórico = entrenamiento
    # actual = nuevos datos
    df_hist = X_train.copy()
    df_actual = X_test.copy()

    resultados = []

    print("\n🔍 Calculando Data Drift...\n")

    for col in df_hist.columns:

        print(f"Procesando variable: {col}")

        # Validar que exista en ambos datasets
        if col not in df_actual.columns:
            continue

        # Validar columnas vacías o NaN
        if df_hist[col].dropna().empty or df_actual[col].dropna().empty:
            print(f"[WARNING] Columna ignorada por NaN: {col}")
            continue

        try:

            # =============================
            # Variables numéricas
            # =============================
            if pd.api.types.is_numeric_dtype(df_hist[col]):

                psi = calcular_psi(df_hist[col], df_actual[col])
                ks = calcular_ks(df_hist[col], df_actual[col])
                js = calcular_js(df_hist[col], df_actual[col])

                resultados.append({
                    "variable": col,
                    "tipo": "numérica",
                    "PSI": psi,
                    "KS": ks,
                    "JS": js,
                    "Chi2": np.nan,
                    "riesgo": evaluar_riesgo(psi)
                })

            # =============================
            # Variables categóricas
            # =============================
            else:

                chi2 = calcular_chi2(df_hist[col], df_actual[col])

                resultados.append({
                    "variable": col,
                    "tipo": "categórica",
                    "PSI": np.nan,
                    "KS": np.nan,
                    "JS": np.nan,
                    "Chi2": chi2,
                    "riesgo": "🟡 Revisar"
                })

        except Exception as e:
            print(f"[ERROR] {col}: {e}")

    # Convertir resultados a DataFrame
    df_resultados = pd.DataFrame(resultados)

    print("\n📊 RESULTADOS DE DRIFT:")
    print(
        df_resultados
        #.sort_values(by="PSI", ascending=False)
    )

    return df_resultados


# =============================
# Ejecución
# =============================
if __name__ == "__main__":
    monitorear_drift()