import pandas as pd
import networkx as nx
import os

# -------------------------------
# CONFIGURATION
# -------------------------------

NODES_PATH = "gephi_graph/nodes.csv"
EDGES_PATH = "gephi_graph/edges.csv"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD NODES AND EDGES
# =========================

nodes = pd.read_csv(NODES_PATH)
edges = pd.read_csv(EDGES_PATH)

# Creation graphe non directionnel
G = nx.Graph()

# Add nodes with type attribute
for _, row in nodes.iterrows():   #on parcourt le tableau ligne par ligne sans tenir compte de l'index
    G.add_node(row["Id"], node_type=row["Type"]) #on crée un node (nom = ID et attribut = Type)

# Add weighted edges
for _, row in edges.iterrows():
    G.add_edge(  # ajoute un lien entre source et target avec son poids
        row["Source"],
        row["Target"],
        weight=row["Weight"]
    )


# =========================
# DEGREE CENTRALITY (RAW DEGREE)
# =========================

'''degree = dict(G.degree())

df_degree = pd.DataFrame.from_dict(
    degree,
    orient="index",
    columns=["degree"]
).reset_index().rename(columns={"index": "node"})'''


df_degree = pd.DataFrame(
    G.degree(),
    columns=["node", "degree"]
)
# =========================
# WEIGHTED DEGREE (STRENGTH)
# =========================

'''weighted_degree = dict(G.degree(weight="weight"))

df_degree["weighted_degree"] = df_degree["node"].map(weighted_degree)'''

'''df_weighted = pd.DataFrame(
    G.degree(weight="weight"),
    columns=["node", "weighted_degree"]
)'''

df_weighted_degree = pd.DataFrame(
    G.degree(weight="weight"),
    columns=["node", "weighted_degree"]
)

# =========================
# ADD NODE TYPE
# =========================

df_degree = df_degree.merge(  #Regarde la colonne node dans le premier tableau et cherche la correspondance dans la colonne Id du deuxième tableau.
    nodes[["Id", "Type"]],
    left_on="node",
    right_on="Id",
    how="left"  #Même si on ne trouve pas de correspondant, on garde quand même l'info
).drop(columns="Id") #apres fusion, cette colonne devient un doublon donc on la supprime

df_degree.rename(columns={"Type": "node_type"}, inplace=True) #renome colonne type par node type

df_results = df_degree.merge(
    df_weighted_degree,
    on="node"
)






print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# =========================
# SAVE RESULTS
# =========================

'''df_degree.sort_values(
    by="weighted_degree",
    ascending=False
).to_csv("results/degree_centrality.csv", index=False)'''

output_path = os.path.join(OUTPUT_DIR, "degree_centrality.csv")
df_results.to_csv(output_path, index=False)


print("✅ Degree centrality computed.")

print(f"Degree centrality saved to {output_path}")

