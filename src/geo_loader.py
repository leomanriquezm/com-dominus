#src/geo_loader.py

import geopandas as gpd
from .config import SHAPEFILE_PATH, CRS_METRIC, EXCLUDE_COMUNAS

def load_comunas(shp_path: str = SHAPEFILE_PATH,
                 crs: str = CRS_METRIC) -> gpd.GeoDataFrame:
    """
    Carga el shapefile (.shp. Recordar que deben estar en la misma carpeta todos los archivos de geo) de comunas, lo lleva a CRS métrico y arregla geometrías.
    """
    comunas = gpd.read_file(shp_path)
    comunas = comunas.to_crs(crs)

    #Arreglar geometrías (buffer(0) limpia pequeños problemas)
    comunas["geometry"] = comunas.buffer(0)

    #Me quedo con columnas relevantes
    comunas = comunas[["Comuna", "Region", "geometry"]]

    #Excluir comunas insulares
    comunas = comunas[~comunas["Comuna"].isin(EXCLUDE_COMUNAS)]

    return comunas
