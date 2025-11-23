#src/agent.py
from .geo_loader import load_comunas
from .graph_builder import build_adjacency_graph
from .gmds_solver import solve_gmds_by_region
from .dominance_labeling import label_dominance, build_ml_dataframe
from .ml_rf import train_random_forest, plot_feature_importance
from .visualization import plot_biobio_graph

class ComDominusAgent:
    def __init__(self, shp_path, excel_path, features):
        self.shp_path = shp_path
        self.excel_path = excel_path
        self.features = features

        self.comunas = None
        self.G = None
        self.dominating_by_region = None
        self.all_dominating_nodes = None
        self.df_comunas = None
        self.best_model = None
        self.metrics = None

    def build_network(self):
        self.comunas = load_comunas(self.shp_path)
        self.G = build_adjacency_graph(self.comunas)

    def solve_gmds(self):
        (self.dominating_by_region,
         self.all_dominating_nodes) = solve_gmds_by_region(self.comunas, self.G)
        self.comunas = label_dominance(self.comunas, self.all_dominating_nodes)

    def prepare_ml_data(self):
        self.df_comunas = build_ml_dataframe(self.excel_path, self.comunas)

    def train_rf(self):
        self.best_model, self.metrics = train_random_forest(
            self.df_comunas, self.features
        )

    def plot_outputs(self):
        plot_biobio_graph(self.comunas, self.G,
                          dominating_nodes=self.all_dominating_nodes)
        plot_feature_importance(self.best_model, self.features)
