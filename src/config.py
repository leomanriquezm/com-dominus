#src/config.py

DATA_DIR = "data"

#Rutas de entrada
SHAPEFILE_PATH = f"{DATA_DIR}/comunas.shp"
EXCEL_COMUNAS_PATH = f"{DATA_DIR}/comunas0.xlsx"

#Columnas de atributos para modelos de ML
FEATURES = ["empresas", "trabajadores", "pobreza", "campamentos", "h_campamentos"]

#CRS métrico
CRS_METRIC = "EPSG:3857"

#Comunas a excluir
EXCLUDE_COMUNAS = ["Isla de Pascua", "Juan Fernández"]

#Umbral de distancia para considerar comunas vecinas (en metros)
UMBRAL_VECINDAD = 25
