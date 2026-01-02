import pandas as pd
import networkx as nx

def load_graph(nodes_path, edges_path, invert_weights=False):
    """
    Load a graph from nodes and edges CSV files.

    Parameters:
    - invert_weights: if True, weight = 1 / TF-IDF (for shortest paths)
    """

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    G = nx.Graph()

    # Add nodes
    for _, row in nodes.iterrows():
        G.add_node(row["Id"], node_type=row["Type"])

    # Add edges
    for _, row in edges.iterrows():
        weight = (
            1 / row["Weight"] if invert_weights and row["Weight"] > 0
            else row["Weight"]
        )
        G.add_edge(row["Source"], row["Target"], weight=weight)

    return G, nodes