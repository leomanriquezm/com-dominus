####################################################################
####################################################################
#Com-Dominus: Agente para Comunas Dominantes en Redes Territoriales#
####################################################################
####################################################################

Com-Dominus es un agente inteligente que:
1. Construye una red de vecindad entre comunas chilenas (por fronteras).
2. Resuelve el problema de Dominancia Mínima Global (GMDS) por región.
3. Etiqueta comunas dominantes.
4. Entrena un modelo de Random Forest para explicar la dominancia.
5. Genera salidas visuales e interpretables (mapa, importancia de variables, SHAP).

---

## Estructura del proyecto

com-dominus/
│
├── data/
│   ├── comunas.shp
│   ├── comunas0.xlsx
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── geo_loader.py
│   ├── graph_builder.py
│   ├── gmds_solver.py
│   ├── dominance_labeling.py
│   ├── ml_rf.py
│   ├── visualization.py
│   ├── interpretability.py
│   └── agent.py
│
└── main.py

###############
###############
#Para ejecutar#
###############
###############

Desde CMD:
1. cd "ruta..."
2. py main.py




