#src/visualization.py
import matplotlib.pyplot as plt
import networkx as nx

def plot_biobio_graph(comunas_gdf, G, output_path="biobio_dominancia.png",
                      dominating_nodes=None):
    comunas_biobio = comunas_gdf[comunas_gdf["Region"] == "Región del Bío-Bío"].copy()
    sub_nodes = list(comunas_biobio["Comuna"])
    G_biobio = G.subgraph(sub_nodes)

    pos_biobio = {
        row["Comuna"]: (row.geometry.centroid.x, row.geometry.centroid.y)
        for _, row in comunas_biobio.iterrows()
    }

    if dominating_nodes is None:
        node_colors = "lightgray"
        node_sizes = 35
    else:
        node_colors = [
            "red" if comuna in dominating_nodes else "gray"
            for comuna in G_biobio.nodes()
        ]
        node_sizes = [
            80 if comuna in dominating_nodes else 30
            for comuna in G_biobio.nodes()
        ]

    plt.figure(figsize=(10, 12))
    ax = comunas_biobio.plot(
        color="whitesmoke",
        edgecolor="lightgray",
        figsize=(10, 12)
    )

    nx.draw_networkx_edges(
        G_biobio, pos_biobio,
        ax=ax,
        edge_color="black",
        alpha=0.4,
        width=0.8
    )

    nx.draw_networkx_nodes(
        G_biobio, pos_biobio,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9
    )

    nx.draw_networkx_labels(
        G_biobio, pos_biobio,
        ax=ax,
        font_size=6
    )

    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
