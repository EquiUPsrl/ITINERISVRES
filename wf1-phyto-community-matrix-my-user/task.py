from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import os
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--filtered_file', action='store', type=str, required=True, dest='filtered_file')


args = arg_parser.parse_args()
print(args)

id = args.id

filtered_file = args.filtered_file.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF1/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF1/' + 'data'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

datain = filtered_file


cluster = ['treatment', 'season']

taxlev = "scientificname_accepted"

param = "density"

display = "site"   # here we cluster sites

method = "braycurtis"


action_name = "community_matrix"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]

    p = act.set_index("parameter")["value"]
    
    taxlev = p.get("taxlev", taxlev)
    param = p.get("param", param)
    display = p.get("display", display)
    method = p.get("method", method)
    
    if "cluster" in p:
        cluster = [x for x in p["cluster"].split(",")]

dataset = pd.read_csv(datain, sep=";", engine="python", decimal=".")

dataset.columns = (
    dataset.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.lower()
)

cluster_lower = [c.lower() for c in cluster]
taxlev_lower = taxlev.lower()
param_lower = param.lower()

missing_cols = [c for c in cluster_lower + [taxlev_lower, param_lower] if c not in dataset.columns]
if missing_cols:
    raise KeyError(f"Missing required columns in dataset: {missing_cols}")

if len(cluster_lower) > 1:
    site_id = dataset[cluster_lower].astype(str).agg(".".join, axis=1)
elif len(cluster_lower) == 1 and cluster_lower[0] != "whole":
    site_id = dataset[cluster_lower[0]].astype(str)
else:
    site_id = pd.Series("all", index=dataset.index)

if param_lower not in dataset.columns:
    dataset[param_lower] = 1.0

matz = (
    dataset
    .groupby([site_id, dataset[taxlev_lower]])[param_lower]
    .sum()
    .unstack(fill_value=0.0)
)

matrix_path = os.path.join(output_dir, f"CommunityMatrix_{param_lower}.csv")
matz.to_csv(matrix_path, sep=";", decimal=".", index=True)
print("Community matrix saved to:", matrix_path)

if display == "species":
    mat_for_analysis = matz.T
else:
    mat_for_analysis = matz

labels = mat_for_analysis.index.astype(str)

dist_vec = pdist(mat_for_analysis.values, metric=method)
linkage_data = linkage(dist_vec, method="complete")

plt.figure(figsize=(7, 5))
dendrogram(
    linkage_data,
    labels=labels,
    leaf_rotation=90,
    leaf_font_size=10
)

plt.xlabel(display.capitalize())
plt.ylabel(f"{method} distance")
plt.title(f"Hierarchical clustering – {display} ({method})")

plt.tight_layout()
file_graph = os.path.join(output_dir, f"CommunityAnalysis_dendrogram_{display}.png")
plt.savefig(file_graph, dpi=300, bbox_inches="tight")
plt.show()
print("Community dendrogram saved to:", file_graph)

"""

dissimilarity_matrix = squareform(dist_vec)

mds = MDS(
    n_components=2,
    metric=False,
    dissimilarity="precomputed",
    random_state=42,
    max_iter=3000,
    n_init=10
)
coords = mds.fit_transform(dissimilarity_matrix)

plt.figure(figsize=(7, 5))
plt.scatter(coords[:, 0], coords[:, 1])

for i, lab in enumerate(labels):
    plt.text(coords[i, 0], coords[i, 1], lab, fontsize=8, ha="center", va="center")

if display == "site":
    plt.xlabel("NMDS 1 (sites)")
    plt.ylabel("NMDS 2 (sites)")
else:
    plt.xlabel("NMDS 1 (species)")
    plt.ylabel("NMDS 2 (species)")

plt.title(f"NMDS ({method} distance) – {display}")

plt.tight_layout()
file_graph_nmds = os.path.join(output_dir, f"CommunityAnalysis_nmds_{display}.png")
plt.savefig(file_graph_nmds, dpi=300, bbox_inches="tight")
plt.show()
print("NMDS plot saved to:", file_graph_nmds)

"""

file_matrix_path = open("/tmp/matrix_path_" + id + ".json", "w")
file_matrix_path.write(json.dumps(matrix_path))
file_matrix_path.close()
