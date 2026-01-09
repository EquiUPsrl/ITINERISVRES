from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.special import gammaln

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--matrix_path', action='store', type=str, required=True, dest='matrix_path')


args = arg_parser.parse_args()
print(args)

id = args.id

matrix_path = args.matrix_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF1/' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

mat = pd.read_csv(matrix_path, sep=";", decimal=".", index_col=0)

labels = mat.index.astype(str)

dist_vec = pdist(mat.values, metric=method)
dist_matrix = squareform(dist_vec)

def permanova_manual(distance_matrix, groups, n_perm=999):
    groups = np.array(groups)
    N = len(groups)

    grand_mean = distance_matrix[np.triu_indices(N, 1)].mean()

    ss_total = np.sum((distance_matrix - grand_mean)**2) / 2

    ss_between = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) > 1:
            gm = distance_matrix[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
            ss_between += len(idx) * (gm - grand_mean)**2

    n_groups = len(np.unique(groups))
    df_between = n_groups - 1
    df_within = N - n_groups

    if df_within <= 0:
        return np.nan, np.nan

    ss_within = ss_total - ss_between

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    F_stat = ms_between / ms_within

    perm_F = []
    for _ in range(n_perm):
        perm_groups = np.random.permutation(groups)
        ss_between_perm = 0

        for g in np.unique(perm_groups):
            idx = np.where(perm_groups == g)[0]
            if len(idx) > 1:
                gm = distance_matrix[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
                ss_between_perm += len(idx) * (gm - grand_mean)**2

        ms_between_perm = ss_between_perm / df_between
        perm_F.append(ms_between_perm / ms_within)

    p_val = (np.sum(np.array(perm_F) >= F_stat) + 1) / (n_perm + 1)
    return F_stat, p_val

def permdisp_manual(distance_matrix, groups, n_perm=999):
    groups = np.array(groups)
    dist_matrix = np.array(distance_matrix)
    N = len(groups)

    group_centroids = {}
    group_mean_distances = {}

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        sub = dist_matrix[np.ix_(idx, idx)]
        centroid = sub.mean(axis=0)
        group_centroids[g] = centroid
        group_mean_distances[g] = sub.mean()

    obs = np.var(list(group_mean_distances.values()))

    perm_stats = []
    for _ in range(n_perm):
        perm_groups = np.random.permutation(groups)
        perm_distances = []

        for g in np.unique(perm_groups):
            idx = np.where(perm_groups == g)[0]
            sub = dist_matrix[np.ix_(idx, idx)]
            perm_distances.append(sub.mean())

        perm_stats.append(np.var(perm_distances))

    p_value = (np.sum(np.array(perm_stats) >= obs) + 1) / (n_perm + 1)
    return obs, p_value

meta = pd.DataFrame({"site": labels})
meta["treatment"] = meta["site"].str.split(".", expand=True)[0]
meta["season"] = meta["site"].str.split(".", expand=True)[1]

results = {}

results["treatment"] = permanova_manual(dist_matrix, meta["treatment"])
results["season"] = permanova_manual(dist_matrix, meta["season"])
results["treatment+season"] = permanova_manual(
    dist_matrix,
    meta["treatment"] + "_" + meta["season"]
)

out_perm = os.path.join(output_dir, "PERMANOVA_results.txt")
with open(out_perm, "w") as f:
    for k, v in results.items():
        f.write(f"{k}: F={v[0]:.4f}, p={v[1]:.4f}\n")

print("PERMANOVA results saved to:", out_perm)

disp_results = {}
disp_results["treatment"] = permdisp_manual(dist_matrix, meta["treatment"])
disp_results["season"] = permdisp_manual(dist_matrix, meta["season"])

out_disp = os.path.join(output_dir, "PERMDISP_results.txt")
with open(out_disp, "w") as f:
    for k, v in disp_results.items():
        f.write(f"{k}: variance={v[0]:.4f}, p={v[1]:.4f}\n")

print("PERMDISP results saved to:", out_disp)

pairwise_out = os.path.join(output_dir, "Pairwise_PERMANOVA.txt")
with open(pairwise_out, "w") as f:

    treatments = meta["treatment"].unique()
    f.write("PAIRWISE TREATMENT\n")
    for a, b in itertools.combinations(treatments, 2):
        idx = meta.index[(meta["treatment"] == a) | (meta["treatment"] == b)]
        dist_sub = dist_matrix[np.ix_(idx, idx)]
        groups_sub = meta.loc[idx, "treatment"]
        F, p = permanova_manual(dist_sub, groups_sub)
        f.write(f"{a} vs {b}: F={F:.4f}, p={p:.4f}\n")

    seasons = meta["season"].unique()
    f.write("\nPAIRWISE SEASON\n")
    for a, b in itertools.combinations(seasons, 2):
        idx = meta.index[(meta["season"] == a) | (meta["season"] == b)]
        dist_sub = dist_matrix[np.ix_(idx, idx)]
        groups_sub = meta.loc[idx, "season"]
        F, p = permanova_manual(dist_sub, groups_sub)
        f.write(f"{a} vs {b}: F={F:.4f}, p={p:.4f}\n")

print("Pairwise PERMANOVA saved to:", pairwise_out)




matz = pd.read_csv(matrix_path, sep=";", decimal=".", index_col=0)

mat_values = matz.values
site_labels = matz.index.astype(str)


def log_comb(a, b):
    """log( C(a, b) ) in a numerically stable manner."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)

def rarefaction_curve(counts, max_depth=None, n_points=100):
    """
    Individual-based rarefaction curve for a site.
    counts: abundances per species (1D array)
    max_depth: maximum depth (n)
    n_points: number of points along the curve
    """
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]

    N = counts.sum()
    if N <= 0:
        return np.array([]), np.array([])

    if max_depth is None or max_depth > N:
        max_depth = int(N)

    if n_points >= max_depth:
        n_vals = np.arange(1, max_depth + 1)
    else:
        n_vals = np.unique(np.linspace(1, max_depth, n_points, dtype=int))

    expected_S = np.zeros_like(n_vals, dtype=float)

    for i, n in enumerate(n_vals):
        log_denom = log_comb(N, n)

        valid = (N - counts) >= n
        p_absent = np.zeros_like(counts, dtype=float)

        if np.any(valid):
            log_num = log_comb(N - counts[valid], n)
            p_absent[valid] = np.exp(log_num - log_denom)

        expected_S[i] = np.sum(1.0 - p_absent)

    return n_vals, expected_S


total_individuals = mat_values.sum(axis=1)
positive_sites = total_individuals > 0
if not np.any(positive_sites):
    raise ValueError("No sites with individuals > 0.")

min_N = int(total_individuals[positive_sites].min())
print("Min N between sites:", min_N)

max_depth_global = min(min_N, 5000)
print("Maximum depth used for curves:", max_depth_global)

all_curves = {}
all_nvals = None

for idx, label in enumerate(site_labels):
    counts = mat_values[idx, :]
    N = counts.sum()
    if N <= 0:
        print(f"Site {label}: no individual, jump.")
        continue

    n_vals, exp_S = rarefaction_curve(counts, max_depth=max_depth_global, n_points=100)
    all_curves[label] = exp_S
    if all_nvals is None:
        all_nvals = n_vals


n_vals = all_nvals
rare_df = pd.DataFrame(index=n_vals)
rare_df.index.name = "n_individuals"

for label, curve in all_curves.items():
    rare_df[label] = curve

rare_csv_path = os.path.join(output_dir, "RarefactionCurves_sites.csv")
rare_df.to_csv(rare_csv_path, sep=";", decimal=".")
print("Rarefaction curves saved in:", rare_csv_path)

plt.figure(figsize=(8, 6))
for label, curve in all_curves.items():
    plt.plot(n_vals, curve, label=label)

plt.xlabel("Number of individuals sampled (n)")
plt.ylabel("Expected richness (E[Species])")
plt.title("Species rarefaction curves (individual-based)")
plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plot_path = os.path.join(output_dir, "RarefactionCurves_sites.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
print("Chart saved in:", plot_path)



treatments = [lab.split(".")[0] for lab in site_labels]

treat_dict = {}
for t in np.unique(treatments):
    rows = [i for i, lab in enumerate(site_labels) if lab.startswith(t)]
    pooled_counts = mat_values[rows, :].sum(axis=0)
    treat_dict[t] = pooled_counts

pooled_curves = {}
for t, counts in treat_dict.items():
    n_vals, expS = rarefaction_curve(counts, max_depth=max_depth_global, n_points=100)
    pooled_curves[t] = expS

plt.figure(figsize=(8,6))
for t, curve in pooled_curves.items():
    plt.plot(n_vals, curve, label=t)

plt.xlabel("Number of individuals sampled (n)")
plt.ylabel("Expected richness (E[Species])")
plt.title("Pooled rarefaction per Treatment")
plt.legend()
plt.tight_layout()
plt.show()

