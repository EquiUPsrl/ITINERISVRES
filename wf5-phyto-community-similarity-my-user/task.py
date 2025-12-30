from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from collections import Counter
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--filtered_file', action='store', type=str, required=True, dest='filtered_file')

arg_parser.add_argument('--sample_metrics_file', action='store', type=str, required=True, dest='sample_metrics_file')


args = arg_parser.parse_args()
print(args)

id = args.id

filtered_file = args.filtered_file.replace('"','')
sample_metrics_file = args.sample_metrics_file.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'
conf_temp_path = conf_temp_path = '/tmp/data/WF5/' + 'tmp'

warnings.filterwarnings("ignore", category=ValueWarning)


sns.set(style="white")


input_file = filtered_file
sample_metric_file = sample_metrics_file
output_dir = conf_output_path
TMP_DIR = conf_temp_path
CSV_DIR    = os.path.join(output_dir, "csv")
PLOTS_DIR  = os.path.join(output_dir, "plots")

os.makedirs(CSV_DIR, exist_ok=True)

df_species = pd.read_csv(input_file, low_memory=False)
sample_metrics = pd.read_csv(sample_metric_file, low_memory=False)

print(df_species.columns)

print("\n=== Community matrix & dissimilarities ===")


sample_id_cols = [
    col for col in ["country", "locality", "year", "month",
                    "parentEventID", "eventID", "season"]
    if col in df_species.columns
]

df_species["sample_id"] = df_species[sample_id_cols].astype(str).agg("_".join, axis=1)


comm = (
    df_species
    .pivot_table(
        index="sample_id",
        columns="acceptedNameUsage",
        values="density",
        aggfunc="sum",
        fill_value=0
    )
)

comm_log = np.log10(comm + 1)

print("Community matrix shape (samples × species):", comm_log.shape)

meta_cols = ["country", "locality", "season", "year", "month"]
meta_cols = [c for c in meta_cols if c in sample_metrics.columns]

sample_metrics["sample_id"] = sample_metrics[sample_id_cols].astype(str).agg("_".join, axis=1)
meta = (
    sample_metrics
    .drop_duplicates(subset=["sample_id"])
    .set_index("sample_id")[meta_cols]
    .reindex(comm_log.index)
)

print("Metadata matched for communities:", meta.shape[0], "samples")

bray_dist = pdist(comm_log.values, metric="braycurtis")
bray_dist_mat = squareform(bray_dist)

bray_df = pd.DataFrame(bray_dist_mat, index=comm_log.index, columns=comm_log.index)
bray_path = os.path.join(CSV_DIR, "bray_curtis_distance_matrix.csv")
bray_df.to_csv(bray_path)
print("Saved Bray–Curtis distance matrix:", bray_path)


def morisita_horn_similarity(u, v):
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    N1, N2 = u.sum(), v.sum()
    if N1 == 0 or N2 == 0:
        return 0.0
    lambda1 = np.sum(u ** 2) / (N1 ** 2)
    lambda2 = np.sum(v ** 2) / (N2 ** 2)
    denom = (lambda1 + lambda2) * N1 * N2
    if denom == 0:
        return 0.0
    return (2 * np.sum(u * v)) / denom



loc_comm = (
    df_species
    .groupby(["locality", "acceptedNameUsage"], dropna=False)["density"]
    .sum()
    .unstack(fill_value=0)
)

loc_comm_log = np.log10(loc_comm + 1)
locs = loc_comm_log.index.tolist()
X_loc = loc_comm_log.values
n_loc = len(locs)

mh_loc = np.zeros((n_loc, n_loc))
for i in range(n_loc):
    for j in range(n_loc):
        mh_loc[i, j] = morisita_horn_similarity(X_loc[i], X_loc[j])

mh_loc_df = pd.DataFrame(mh_loc, index=locs, columns=locs)
mh_loc_path = os.path.join(CSV_DIR, "morisita_horn_locality.csv")
mh_loc_df.to_csv(mh_loc_path)
print("Saved Morisita–Horn (locality) matrix:", mh_loc_path)

loc_season_comm = (
    df_species
    .groupby(["locality", "season", "acceptedNameUsage"], dropna=False)["density"]
    .sum()
    .unstack(fill_value=0)
)

loc_season_comm_log = np.log10(loc_season_comm + 1)
ls_index = loc_season_comm_log.index
X_ls = loc_season_comm_log.values
n_ls = len(ls_index)

mh_ls = np.zeros((n_ls, n_ls))
for i in range(n_ls):
    for j in range(n_ls):
        mh_ls[i, j] = morisita_horn_similarity(X_ls[i], X_ls[j])

mh_ls_df = pd.DataFrame(mh_ls, index=ls_index, columns=ls_index)
mh_ls_path = os.path.join(CSV_DIR, "morisita_horn_locality_season.csv")
mh_ls_df.to_csv(mh_ls_path)
print("Saved Morisita–Horn (locality × season) matrix:", mh_ls_path)



print("\n=== nMDS ordination ===")


mds = MDS(
    n_components=2,
    metric=False,
    dissimilarity="precomputed",
    random_state=0,
    n_init=20,
    max_iter=300
)

mds_coords = mds.fit_transform(bray_df.values)
stress = mds.stress_
print(f"nMDS stress: {stress:.4f}")

mds_df = pd.DataFrame(mds_coords, index=bray_df.index, columns=["MDS1", "MDS2"])
mds_df = mds_df.join(meta)

mds_path = os.path.join(CSV_DIR, "nmds_coordinates_samples.csv")
mds_df.to_csv(mds_path)
print("Saved nMDS coordinates:", mds_path)

centroids = (
    mds_df
    .groupby(["locality", "season"], dropna=False)[["MDS1", "MDS2"]]
    .mean()
    .reset_index()
)

centroids_path = os.path.join(CSV_DIR, "nmds_centroids_locality_season.csv")
centroids.to_csv(centroids_path, index=False)
print("Saved nMDS centroids:", centroids_path)


sns.set(style="white", rc={"axes.grid": False})

plt.figure(figsize=(10, 8))

localities = mds_df["locality"].unique()
palette_loc = sns.color_palette("tab20", len(localities))
color_map = dict(zip(localities, palette_loc))

shape_map = {"Spring": "s", "Autumn": "^"}  # square / triangle

for _, row in mds_df.iterrows():
    plt.scatter(
        row["MDS1"],
        row["MDS2"],
        s=45,
        c=[color_map[row["locality"]]],
        marker=shape_map.get(row["season"], "o"),
        alpha=0.55,                 # lighter points
        edgecolor="black",
        linewidth=0.3
    )

for _, row in centroids.iterrows():
    plt.scatter(
        row["MDS1"],
        row["MDS2"],
        s=200,
        c=[color_map[row["locality"]]],
        marker=shape_map.get(row["season"], "o"),
        edgecolor="black",
        linewidth=1.2
    )

plt.title(f"nMDS ordination (Bray–Curtis); stress = {stress:.4f}", fontsize=14)
plt.xlabel("MDS1")
plt.ylabel("MDS2")


loc_patches = [mpatches.Patch(color=color_map[l], label=l) for l in localities]

season_markers = [
    plt.Line2D([0], [0], marker="s", color="black", linestyle="",
               label="Spring"),
    plt.Line2D([0], [0], marker="^", color="black", linestyle="",
               label="Autumn")
]

legend1 = plt.legend(handles=loc_patches, title="Locality",
                     bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
plt.gca().add_artist(legend1)

plt.legend(handles=season_markers, title="Season",
           bbox_to_anchor=(1.02, 0.6), loc="upper left", fontsize=10)

plt.tight_layout()
fig_path = os.path.join(PLOTS_DIR, "nmds_simple_locality_color_season_shape.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved clean simple nMDS figure:", fig_path)



print("\n=== Morisita–Horn heatmaps & SIMPER ===")

sns.set(style="white")
plt.figure(figsize=(12, 10))

mask = np.triu(np.ones_like(mh_loc_df, dtype=bool), k=1)

sns.heatmap(
    mh_loc_df.astype(float),   # <--- forcing numeric ensures annot works
    cmap=sns.light_palette("blue", as_cmap=True),  # soft colors
    annot=True,
    fmt=".2f",
    mask=mask,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Morisita–Horn similarity"},
    square=True
)

plt.title("Morisita–Horn similarity — Localities")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mh_locality_soft.png"), dpi=300)
plt.close()


plt.figure(figsize=(14, 12))

mask = np.triu(np.ones_like(mh_ls_df, dtype=bool), k=1)

sns.heatmap(
    mh_ls_df.astype(float),
    cmap=sns.light_palette("green", as_cmap=True),
    annot=True,
    fmt=".2f",
    mask=mask,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Morisita–Horn similarity"},
    square=True
)

plt.title("Morisita–Horn similarity — Locality × Season")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "mh_loc_season_soft.png"), dpi=300)
plt.close()


def simper(comm_df, group_series, n_top=20):
    """
    Simple SIMPER implementation.
    comm_df: samples × species (log-transformed abundances)
    group_series: grouping factor (e.g. locality)
    Returns a dict of DataFrames, one per pair of groups.
    """
    results = {}
    species = comm_df.columns

    groups = group_series.astype(str)
    unique_groups = groups.unique()

    for i in range(len(unique_groups)):
        for j in range(i + 1, len(unique_groups)):
            g1 = unique_groups[i]
            g2 = unique_groups[j]

            idx1 = groups[groups == g1].index
            idx2 = groups[groups == g2].index

            if len(idx1) == 0 or len(idx2) == 0:
                continue

            mean1 = comm_df.loc[idx1].mean(axis=0)
            mean2 = comm_df.loc[idx2].mean(axis=0)

            diff = np.abs(mean1 - mean2)
            total_diff = diff.sum()
            if total_diff == 0:
                continue

            contrib = diff / total_diff * 100.0
            df_pair = pd.DataFrame({
                "species": species,
                "mean_abund_group1": mean1.values,
                "mean_abund_group2": mean2.values,
                "abs_diff": diff.values,
                "percent_contribution": contrib.values,
            })
            df_pair = df_pair.sort_values("percent_contribution", ascending=False)
            df_pair["cumulative_contribution"] = df_pair["percent_contribution"].cumsum()

            results[(g1, g2)] = df_pair.head(n_top)

    return results


simper_results = simper(comm_log, meta["locality"], n_top=30)

simper_dir = os.path.join(output_dir, "simper")
os.makedirs(simper_dir, exist_ok=True)

for (g1, g2), df_pair in simper_results.items():
    fname = f"simper_{g1}_vs_{g2}.csv".replace(" ", "_")
    path = os.path.join(simper_dir, fname)
    df_pair.to_csv(path, index=False)

print("SIMPER results saved in folder:", simper_dir)




print("\n=== SIMPER summary ===")

simper_dir = os.path.join(output_dir, "simper")
simper_files = sorted(glob.glob(os.path.join(simper_dir, "simper_*.csv")))

print(f"Found {len(simper_files)} SIMPER files in:", simper_dir)

summary_rows = []
taxa_counter = Counter()

for fpath in simper_files:
    df = pd.read_csv(fpath)
    if df.empty:
        continue

    top5 = df.head(5).copy()

    if "cumulative_contribution" in top5.columns:
        top5_cum = float(top5["cumulative_contribution"].iloc[-1])
    else:
        top5_cum = float(top5["percent_contribution"].sum())

    fname = os.path.basename(fpath)
    base = fname.replace("simper_", "").replace(".csv", "")
    if "_vs_" in base:
        g1, g2 = base.split("_vs_", 1)
    else:
        g1, g2 = base, ""

    summary_rows.append(
        {
            "file": fname,
            "lagoon1": g1,
            "lagoon2": g2,
            "top5_cumulative_contribution": top5_cum,
        }
    )

    for sp in top5["species"]:
        taxa_counter[sp] = taxa_counter[sp] + 1

simper_summary = pd.DataFrame(summary_rows)

simper_summary_path = os.path.join(CSV_DIR, "simper_summary_top5_cumulative.csv")
simper_summary.to_csv(simper_summary_path, index=False)
print("Saved SIMPER summary table to:", simper_summary_path)

if not simper_summary.empty:
    vals = simper_summary["top5_cumulative_contribution"]
    print("\nTop-5 cumulative contribution (%) across all pairs:")
    print("  N pairs   :", len(vals))
    print("  Mean      :", vals.mean())
    print("  Median    :", vals.median())
    print("  Min       :", vals.min())
    print("  Max       :", vals.max())

    simper_stats = pd.DataFrame(
        {
            "N_pairs": [len(vals)],
            "mean_top5": [vals.mean()],
            "median_top5": [vals.median()],
            "min_top5": [vals.min()],
            "max_top5": [vals.max()],
        }
    )
    simper_stats_path = os.path.join(CSV_DIR, "simper_summary_top5_stats.csv")
    simper_stats.to_csv(simper_stats_path, index=False)
    print("Saved SIMPER stats to:", simper_stats_path)

taxa_rows = []
for sp, count in taxa_counter.most_common():
    taxa_rows.append({"species": sp, "top5_frequency": count})

taxa_summary = pd.DataFrame(taxa_rows)
taxa_summary_path = os.path.join(CSV_DIR, "simper_top_taxa_frequency.csv")
taxa_summary.to_csv(taxa_summary_path, index=False)
print("Saved SIMPER taxa-frequency table to:", taxa_summary_path)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
