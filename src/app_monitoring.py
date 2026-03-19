"""
app_monitoring.py
Visualización del monitoreo de Data Drift
"""

# =============================
# LIBRERÍAS
# =============================

import streamlit as st  # Framework web
import matplotlib.pyplot as plt  # Gráficos

# Importamos la lógica de monitoreo
from model_monitoring import monitorear_drift


# =============================
# CONFIGURACIÓN DE LA APP
# =============================
st.set_page_config(
    page_title="Monitoreo Data Drift",
    layout="wide"
)

# Título principal
st.title("Monitoreo de Data Drift")


# =============================
# EJECUTAR MONITOREO
# =============================
df = monitorear_drift()  # Llamamos función del backend


# =============================
# TABLA DE RESULTADOS
# =============================
st.subheader("Métricas de Drift")
st.dataframe(df)  # Mostrar tabla interactiva


# =============================
# GRÁFICO PSI
# =============================
st.subheader("Variables con mayor Drift")

# Filtrar solo variables numéricas
df_num = df[df["tipo"] == "numérica"]

# Ordenar por PSI descendente
df_num = df_num.sort_values(by="PSI", ascending=False).head(10)

# Crear figura
fig, ax = plt.subplots()

# Gráfico horizontal
ax.barh(df_num["variable"], df_num["PSI"])

# Invertir eje para mejor lectura
ax.invert_yaxis()

# Mostrar gráfico
st.pyplot(fig)


# =============================
# ALERTAS
# =============================
st.subheader("Alertas automáticas")

# Iterar sobre variables con mayor drift
for _, row in df_num.iterrows():

    # Evaluar nivel de riesgo
    if row["PSI"] > 0.25:
        st.error(f"{row['variable']} → Drift ALTO (reentrenar modelo)")
    elif row["PSI"] > 0.1:
        st.warning(f"{row['variable']} → Drift MEDIO (monitorear)")
    else:
        st.success(f"{row['variable']} → Estable")
    
