#src/interpretability.py

import shap
import pandas as pd
import matplotlib.pyplot as plt

def compute_shap(
    best_model,
    X_train_array,
    features,
    output_path="shap_RF.png"
):
    """
    Calcula valores SHAP para el modelo RF y guarda un beeswarm plot (recordar formato de X).
    """
    X_train_df = pd.DataFrame(X_train_array, columns=features)
    rf = best_model.named_steps["rf"]

    explainer = shap.Explainer(rf, X_train_df)
    shap_values = explainer(X_train_df)

    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(shap_values[:, :, 1], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
