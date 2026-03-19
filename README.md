# Proyecto MLOps - Predicción de Pago de Créditos

HENRY - Modulo 5: F.N.C.D.P - Proyecto Integrador


## 1. Descripción del proyecto

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning capaz de predecir si un cliente pagará su crédito a tiempo (`Pago_atiempo`), utilizando información histórica de créditos.

El desarrollo se realizó bajo un enfoque de MLOps, asegurando:

- Reproducibilidad  
- Escalabilidad  
- Trazabilidad  
- Facilidad de despliegue en producción  

El proyecto sigue una estructura estricta de carpetas requerida por pipelines automatizados (Jenkins), garantizando compatibilidad con entornos productivos.

---

## 2. Contexto de negocio

La empresa financiera busca:

- Reducir el riesgo de impago  
- Mejorar la aprobación de créditos  
- Optimizar la toma de decisiones  

El modelo permite identificar clientes con mayor probabilidad de incumplimiento, priorizando la detección de riesgo (clase 0).

---

## 3. Estructura del proyecto

mlops_pipeline/
│
├── src/
│ ├── cargar_datos.ipynb
│ ├── comprension_eda.ipynb
│ ├── ft_engineering.py
│ ├── model_training_evaluation.py
│ ├── model_deploy.py
│ └── model_monitoring.py
│
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
└── README.md


---

## 4. Flujo del pipeline

### 4.1 Carga de datos

Se implementa un script (`cargar_datos.py`) que:

- Obtiene rutas dinámicamente  
- Carga el dataset desde el proyecto  
- Garantiza portabilidad  

---

### 4.2 Análisis exploratorio (EDA)

En el notebook `comprension_eda.ipynb` se realizó:

- Exploración inicial del dataset  
- Limpieza y unificación de nulos  
- Tipado de variables  
- Análisis univariado, bivariado y multivariado  

#### Hallazgos clave

- Dataset con fuerte componente financiero  
- Presencia de outliers y distribuciones sesgadas  
- Variable objetivo desbalanceada  
- Variables relevantes:
  - `puntaje_datacredito`
  - `saldo_mora`
  - `huella_consulta`

---

### 4.3 Feature Engineering

En `ft_engineering.py` se implementó:

#### Procesos clave

- Eliminación de data leakage:
  - `saldo_mora`
  - `saldo_total`
  - `saldo_principal`
- Creación de variables:
  - `ratio_cuota_ingreso`
  - `ratio_deuda_ingreso`
- Generación de variables derivadas:
  - `antiguedad_prestamo`
- Creación de indicadores de valores faltantes  
- Normalización de variables categóricas  

#### División de datos

- Train: 70%  
- Test: 30%  
- Muestreo estratificado  

---

### 4.4 Entrenamiento y evaluación

Archivo: `model_training_evaluation.py`

#### Modelos implementados

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Decision Tree  
- Hist Gradient Boosting  

#### Técnicas aplicadas

- Pipeline completo con:
  - Imputación  
  - Escalamiento  
  - One Hot Encoding  
- Validación cruzada (Stratified K-Fold)  
- Optimización de threshold  

#### Métricas evaluadas

- ROC-AUC  
- Recall (especialmente clase 0)  
- F1-score  
- Curva ROC  

#### Enfoque de negocio

Se prioriza:

- Maximizar el Recall de la clase 0 (clientes que no pagan)  
- Minimizar el riesgo financiero  

---

## 5. Monitoreo del modelo (Data Drift)

Archivo: `model_monitoring.py`

Se implementaron métricas:

- PSI (Population Stability Index)  
- KS (Kolmogorov-Smirnov)  
- Jensen-Shannon  
- Chi-cuadrado (variables categóricas)  

---

### 5.1 Resultados de Drift

- Variables numéricas con drift bajo (PSI < 0.1)  
- Variables categóricas que requieren monitoreo:
  - `tipo_laboral`  
  - `tendencia_ingresos`  

#### Interpretación

- El modelo es estable  
- No requiere reentrenamiento inmediato  
- Puede ser utilizado en producción  

---

### 5.2 Aplicación de monitoreo

Se desarrolló una aplicación en Streamlit (`app_monitoring.py`) que permite:

- Visualizar métricas de drift  
- Identificar variables con mayor cambio  
- Generar alertas automáticas  

---

## 6. Despliegue del modelo

Archivo: `model_deploy.py`

Este módulo implementa el despliegue del modelo en producción mediante una API REST construida con **FastAPI**, permitiendo consumir el modelo desde cualquier aplicación externa.

---

#### Objetivo

Disponibilizar el modelo entrenado como un servicio accesible vía HTTP para realizar predicciones en tiempo real o por lotes.

---

#### Componentes principales

##### Carga del modelo

El script carga automáticamente:

- Modelo entrenado (`model.pkl`)
- Threshold optimizado (`threshold.txt`)

```python
model = joblib.load(MODEL_PATH)

#En caso de no encontrar el threshold, se utiliza un valor por defecto de 0.5.
```
##### API con FastAPI

Se define una API con los siguientes endpoints:

- / → Health check (verifica que la API está activa)

- /predict → Genera predicciones

- /explain → Explica las variables más importantes del modelo

##### Esquema de entrada

Se utiliza Pydantic para validar los datos de entrada:

```json
{
  "data": [
    {
      "edad_cliente": 35,
      "salario_cliente": 2500,
      "cuota_pactada": 300,
      "total_otros_prestamos": 500,
      "tendencia_ingresos": "Estable"
    }
  ]
}
```

##### Feature Engineering en producción

Se replica la lógica de transformación utilizada en entrenamiento:

- Creación de variables:

    - ratio_cuota_ingreso

    - ratio_deuda_ingreso

- Normalización de categorías (tendencia_ingresos)

- Manejo de valores faltantes implícito

Esto garantiza consistencia entre entrenamiento y predicción.

##### Predicción (/predict)

El endpoint:

1. Convierte el input en DataFrame

2. Aplica feature engineering

3. Alinea las columnas con el modelo (feature_names_in_)

4. Genera probabilidades (predict_proba)

5. Aplica threshold optimizado

6. Devuelve resultados interpretables

Output:

```json
{
  "threshold_usado": 0.42,
  "resultados": [
    {
      "prediccion": 1,
      "probabilidad_pago": 0.87,
      "riesgo": "Bajo",
      "interpretacion": "Cliente probablemente pagará a tiempo"
    }
  ]
}
```

##### Interpretabilidad (/explain)

Se implementa un endpoint que permite entender el modelo:

- Detecta automáticamente el tipo de modelo:

    - Modelos lineales (coef_)

    - Modelos de árboles (feature_importances_)

- Retorna las 10 variables más influyentes

Esto aporta transparencia y explicabilidad al sistema.

##### Soporte para predicción por lotes

El diseño permite enviar múltiples clientes en una sola solicitud:

```json
{
  "data": [{...}, {...}, {...}]
}
```

Facilitando integración con sistemas batch o pipelines externos.

##### Contenerización (Docker)

El despliegue se realiza mediante una imagen Docker que incluye:

- Código fuente del proyecto

- Dependencias (requirements.txt)

- Modelo entrenado (model.pkl)

- Threshold (threshold.txt)

- Servidor de aplicación (Uvicorn)

Esto permite:

- Portabilidad entre entornos

- Fácil despliegue en cloud

- Integración con CI/CD (Jenkins)

##### Ejecución de la API

```bash
uvicorn src.model_deploy:app --reload
```

Luego acceder a:
- Documentación interactiva:
    http://localhost:8000/docs
---

#### Valor en MLOps

Este módulo representa la etapa de Serving dentro del ciclo MLOps:

- Permite pasar de modelo experimental a producto

- Garantiza reproducibilidad en inferencia

- Facilita integración con sistemas reales

- Añade interpretabilidad y monitoreo potencial

---

## 7. Principales hallazgos

- Alta presencia de outliers en variables financieras  
- Distribuciones no normales en múltiples variables  
- Desbalance en la variable objetivo  
- Variables financieras con alto poder predictivo  
- Mejora del modelo mediante variables derivadas (ratios)  
- Estabilidad del modelo en datos nuevos (sin drift significativo)  

---

## 8. Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## 9. Ejecución del proyecto
```bash
# Entrenamiento del modelo
python src/model_training_evaluation.py
# Monitoreo de drift
python src/model_monitoring.py
# Aplicación de monitoreo
streamlit run src/app_monitoring.py
```

---

## 10. Enfoque MLOps implementado

Este proyecto aplica buenas prácticas de MLOps:

- Separación de componentes (datos, features, modelos, monitoreo)

- Pipelines reproducibles

- Control de data leakage

- Evaluación robusta de modelos

- Monitoreo continuo en producción

- Compatibilidad con procesos CI/CD

---

## 11. Autor

Christian Pascual
Científico de Datos Junior Advanced
