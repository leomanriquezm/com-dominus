#src/dominance_labeling.py

import pandas as pd

def label_dominance(comunas_gdf, all_dominating_nodes):
    """
    Agrega columna 'dominancia' (1/0) al GeoDataFrame de comunas (etiqueta)
    """
    comunas_gdf = comunas_gdf.copy()
    comunas_gdf["dominancia"] = comunas_gdf["Comuna"].apply(
        lambda c: 1 if c in all_dominating_nodes else 0
    )
    return comunas_gdf

def build_ml_dataframe(path_excel: str, comunas_gdf) -> pd.DataFrame:
    """
    Carga df_comunas desde Excel y agrega la variable 'dominancia' desde el GeoDataFrame de comunas.
    """
    df_comunas = pd.read_excel(path_excel)
    df_comunas["dominancia"] = comunas_gdf["dominancia"].values
    return df_comunas
