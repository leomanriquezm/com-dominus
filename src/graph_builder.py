#src/graph_builder.py

import networkx as nx
from .config import UMBRAL_VECINDAD

def build_adjacency_graph(comunas_gdf, umbral_distancia: float = UMBRAL_VECINDAD) -> nx.Graph:
    """
    Construye el grafo de vecindad entre comunas.
    """
    G = nx.Graph()

    # Añadir nodos
    for _, row in comunas_gdf.iterrows():
        G.add_node(row["Comuna"])

    # Spatial index
    sindex = comunas_gdf.sindex

    for i, row_i in comunas_gdf.iterrows():
        geom_i = row_i.geometry
        nombre_i = row_i["Comuna"]

        posibles = list(sindex.intersection(geom_i.bounds))

        for j in posibles:
            if j <= i:
                continue

            row_j = comunas_gdf.iloc[j]
            geom_j = row_j.geometry
            nombre_j = row_j["Comuna"]

            if geom_i.touches(geom_j):
                G.add_edge(nombre_i, nombre_j)
            else:
                dist = geom_i.distance(geom_j)
                if dist <= umbral_distancia:
                    G.add_edge(nombre_i, nombre_j)

    return G
