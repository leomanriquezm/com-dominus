#main.py
from src.agent import ComDominusAgent

if __name__ == "__main__":
    features = ["empresas","trabajadores","pobreza","campamentos","h_campamentos"]

    agent = ComDominusAgent(
        shp_path="data/comunas.shp",
        excel_path="data/comunas0.xlsx",
        features=features
    )

    agent.build_network()
    agent.solve_gmds()
    agent.prepare_ml_data()
    agent.train_rf()
    agent.plot_outputs()
