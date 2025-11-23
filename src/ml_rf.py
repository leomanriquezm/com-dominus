#src/ml_rf.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import matplotlib.pyplot as plt

def train_random_forest(
    df_comunas: pd.DataFrame,
    features: list,
    target: str = "dominancia"
):
    """
    Entrenar un RF con GridSearchCV y validación cruzada estratificada (configurado para 20/80).
    Devuelve:
      -best_model 
      -metrics (acc_train, acc_test, f1_train, f1_test)
      -X_train (como array) para SHAP (importante para usar SHAP)
    """
    X = df_comunas[features]
    y = df_comunas[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        "rf__n_estimators": [5, 10, 15, 25, 50, 100, 200],
        "rf__max_depth": [2, 5, 10, 15, 20, 25],
        "rf__min_samples_split": [2, 5, 7, 9],
        "rf__min_samples_leaf": [1, 2],
        "rf__max_features": ["sqrt", "log2"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    f1_train = f1_score(y_train, y_train_pred, average="macro")
    f1_test = f1_score(y_test, y_test_pred, average="macro")

    print("Mejores hiperparámetros:", grid_search.best_params_)
    print("Accuracy TRAIN:", acc_train, " - TEST:", acc_test)
    print("F1 TRAIN:", f1_train, " - TEST:", f1_test)
    print("\n=== Classification report (TEST) ===")
    print(classification_report(y_test, y_test_pred))
    print("\n=== Matriz de confusión (TEST) ===")
    print(confusion_matrix(y_test, y_test_pred))

    return best_model, (acc_train, acc_test, f1_train, f1_test), X_train.values

def plot_feature_importance(best_model, features, output_path="importancia_RF.png"):
    """
    Guarda figura con importancias de variables del RF.
    """
    rf = best_model.named_steps["rf"]
    importances = rf.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(8, 5))
    plt.barh(np.array(features)[indices], importances[indices])
    plt.xlabel("Importancia relativa")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
