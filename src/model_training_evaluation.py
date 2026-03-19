"""
model_training_evaluation.py
Versión final MLOps con guardado de modelo + threshold óptimo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Métricas
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score,
    recall_score
)

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ft_engineering import preparar_features
from cargar_datos import cargar_datos


# =============================
# Pipeline
# =============================
def build_model(model, num_cols, cat_cols):

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    return Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])


# =============================
# Evaluación con threshold tuning
# =============================
def evaluate_thresholds(y_true, probs):

    thresholds = np.arange(0.01, 0.5, 0.01)

    best_threshold = 0.5
    best_score = 0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1_0 = f1_score(y_true, preds, pos_label=0)

        if f1_0 > best_score:
            best_score = f1_0
            best_threshold = t

    return best_threshold, best_score


# =============================
# Evaluación principal
# =============================
def evaluate_models():

    df = cargar_datos()

    X_train, X_test, y_train, y_test = preparar_features(df)

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # =============================
    # Modelos
    # =============================
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            class_weight={0: 20, 1: 1},
            C=0.5
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingClassifier(),

        "DecisionTree": DecisionTreeClassifier(
            max_depth=8,
            class_weight='balanced',
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=10,
            learning_rate=0.05,
            max_iter=200
        )
    }

    results = []
    pipelines = {}  #  guardar pipelines

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():

        print(f"\n==============================")
        print(f"Evaluando: {name}")
        print(f"==============================")

        pipeline = build_model(model, num_cols, cat_cols)

        # =============================
        # Cross-validation
        # =============================
        probs_cv = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            method='predict_proba'
        )[:, 1]

        auc_cv = roc_auc_score(y_train, probs_cv)

        # =============================
        # Fit final
        # =============================
        pipeline.fit(X_train, y_train)

        #  Guardamos el pipeline
        pipelines[name] = pipeline

        probs_test = pipeline.predict_proba(X_test)[:, 1]

        # =============================
        # Threshold óptimo
        # =============================
        best_threshold, best_score = evaluate_thresholds(y_test, probs_test)

        preds_test = (probs_test >= best_threshold).astype(int)

        report = classification_report(y_test, preds_test, output_dict=True)

        recall_0 = recall_score(y_test, preds_test, pos_label=0)
        auc_test = roc_auc_score(y_test, probs_test)

        results.append({
            "Modelo": name,
            "AUC": auc_test,
            "Recall_Clase_0": recall_0,
            "F1_Clase_0": report['0']['f1-score'],
            "F1_Clase_1": report['1']['f1-score'],
            "Threshold": best_threshold
        })

        # =============================
        # ROC
        # =============================
        fpr, tpr, _ = roc_curve(y_test, probs_test)
        plt.plot(fpr, tpr, label=name)

    plt.plot([0, 1], [0, 1], '--')
    plt.title("ROC Curve Comparativa")
    plt.legend()
    plt.show()

    # =============================
    # Resultados finales
    # =============================
    df_results = pd.DataFrame(results)

    print("\n==============================")
    print("RESULTADOS FINALES")
    print("==============================")
    print(df_results)

    # =============================
    # Mejor modelo
    # =============================
    best_model = df_results.sort_values(by="Recall_Clase_0", ascending=False).iloc[0]

    best_model_name = best_model["Modelo"]
    best_pipeline = pipelines[best_model_name]
    best_threshold = best_model["Threshold"]

    print("\nMEJOR MODELO:")
    print(best_model)

    # =============================
    # Guardado
    # =============================
    
    joblib.dump(best_pipeline, "model.pkl")

    with open("threshold.txt", "w") as f:
        f.write(str(best_threshold))

    print(f"\nModelo guardado como model.pkl ({best_model_name})")
    print(f"Threshold guardado: {best_threshold:.2f}")
    

# =============================
# Ejecutar
# =============================
if __name__ == "__main__":
    evaluate_models()