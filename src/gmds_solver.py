#src/gmds_solver.py

import pulp
import networkx as nx
import pandas as pd

def solve_gmds_by_region(comunas_gdf: pd.DataFrame,
                         G: nx.Graph):
    """
    Resuelve el problema GMDS por región.
    Devuelve:
      -dominating_by_region: dict {region: [lista de comunas dominantes]}
      -all_dominating_nodes: set con todas las comunas dominantes a nivel nacional
    """
    regiones = sorted(comunas_gdf["Region"].unique())
    dominating_by_region = {}
    all_dominating_nodes = set()

    for region in regiones:
        comunas_region = comunas_gdf.loc[
            comunas_gdf["Region"] == region, "Comuna"
        ].tolist()

        G_r = G.subgraph(comunas_region).copy()

        # Problema de optimización
        prob = pulp.LpProblem(f"MinDominatingSet_{region}", pulp.LpMinimize)

        x = {
            v: pulp.LpVariable(f"x_{str(v).replace(' ', '_')}",
                               lowBound=0, upBound=1, cat=pulp.LpBinary)
            for v in G_r.nodes()
        }

        # Objetivo: minimizar tamaño del conjunto dominante
        prob += pulp.lpSum(x[v] for v in G_r.nodes()), "Minimize_number_of_dominating_vertices"

        # Restricciones de dominancia
        for u in G_r.nodes():
            vecinos = list(G_r.neighbors(u))
            prob += x[u] + pulp.lpSum(x[v] for v in vecinos) >= 1, f"Domina_{region}_{u}"

        # Resolver
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Extraer solución
        dominating_set_region = [v for v in G_r.nodes() if pulp.value(x[v]) > 0.5]

        dominating_by_region[region] = dominating_set_region
        all_dominating_nodes.update(dominating_set_region)

    return dominating_by_region, all_dominating_nodes
