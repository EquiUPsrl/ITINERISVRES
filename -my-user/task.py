from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from skbio.stats.distance import DistanceMatrix
from skbio.stats.distance import mantel
import statsmodels.api as sm

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--abiotic_file', action='store', type=str, required=True, dest='abiotic_file')

arg_parser.add_argument('--filtered_file', action='store', type=str, required=True, dest='filtered_file')

arg_parser.add_argument('--sample_metrics_file', action='store', type=str, required=True, dest='sample_metrics_file')


args = arg_parser.parse_args()
print(args)

id = args.id

abiotic_file = args.abiotic_file.replace('"','')
filtered_file = args.filtered_file.replace('"','')
sample_metrics_file = args.sample_metrics_file.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'
conf_temp_path = conf_temp_path = '/tmp/data/WF5/' + 'tmp'

print("\n=== Environment–community relationships ===")



sns.set(style="white", rc={"axes.grid": False})

try:
    SKBIO_MANTEL = True
except ImportError:
    SKBIO_MANTEL = False
    print("scikit-bio not available: Mantel test will be approximated by Pearson correlation on distance vectors.")


output_dir = conf_output_path
TMP_DIR = conf_temp_path
CSV_DIR    = os.path.join(output_dir, "csv")
PLOTS_DIR  = os.path.join(output_dir, "plots")


ABIOTIC_FILE = abiotic_file
SPECIES_FILE = filtered_file
SAMPLE_METRICS_FILE = sample_metrics_file

abiotic = pd.read_csv(ABIOTIC_FILE, sep=";", low_memory=False)
df_species = pd.read_csv(SPECIES_FILE, low_memory=False)
sample_metrics = pd.read_csv(SAMPLE_METRICS_FILE, low_memory=False)

abiotic.columns = abiotic.columns.str.strip()

abiotic = abiotic[[c for c in abiotic.columns if not c.startswith("Unnamed")]]

print("Abiotic data loaded:", abiotic.shape)
print("First columns:", abiotic.columns[:15].tolist())

if "season" not in abiotic.columns:
    def assign_season_from_month(m):
        try:
            m = int(m)
        except Exception:
            return np.nan
        if m == 4:
            return "Spring"
        elif m == 10:
            return "Autumn"
        else:
            return "Other"

    abiotic["season"] = abiotic["month"].apply(assign_season_from_month)

id_cols_env = [c for c in ["country", "locality", "year", "month",
                           "parentEventID", "season"] if c in abiotic.columns]
print("ID columns in abiotic data:", id_cols_env)



env_vars = []
for c in abiotic.columns:
    if c in id_cols_env:
        continue
    col_numeric = pd.to_numeric(abiotic[c], errors="coerce")
    if col_numeric.notna().sum() >= 0.5 * len(col_numeric):
        abiotic[c] = col_numeric
        env_vars.append(c)

print("Number of environmental variables (numeric):", len(env_vars))
print("Environmental variables used (first 20):", env_vars[:20])



group_cols_env = [c for c in ["locality", "season"] if c in abiotic.columns]
if len(group_cols_env) == 0:
    raise ValueError("No 'locality'/'season' columns found in abiotic data for aggregation.")

abiotic_env_mean = (
    abiotic
    .groupby(group_cols_env, dropna=False)[env_vars]
    .mean()
    .reset_index()
)

print("\nAbiotic means at locality × season level (head):")
print(abiotic_env_mean.head())



if "season" not in df_species.columns and "season" in sample_metrics.columns:
    merge_cols = ["country", "locality", "year", "month", "parentEventID", "eventID"]
    common_merge_cols = [c for c in merge_cols
                         if c in df_species.columns and c in sample_metrics.columns]
    df_species = df_species.merge(
        sample_metrics[common_merge_cols + ["season"]].drop_duplicates(),
        on=common_merge_cols,
        how="left"
    )

comm_ls = (
    df_species
    .groupby(["locality", "season", "acceptedNameUsage"])["density"]
    .sum()
    .reset_index()
    .pivot_table(
        index=["locality", "season"],
        columns="acceptedNameUsage",
        values="density",
        fill_value=0
    )
)

comm_ls_log = np.log10(comm_ls + 1)

print("\nCommunity (locality × season) matrix shape:", comm_ls_log.shape)



abiotic_env_mean = abiotic_env_mean.set_index(group_cols_env)
comm_ls_log = comm_ls_log.sort_index()

common_index = comm_ls_log.index.intersection(abiotic_env_mean.index)

comm_ls_valid = comm_ls_log.loc[common_index]
abiotic_env_valid = abiotic_env_mean.loc[common_index]

print("Valid locality × season with both biology & abiotic:", len(common_index))



bray_ls = pdist(comm_ls_valid.values, metric="braycurtis")
bray_ls_mat = squareform(bray_ls)

bray_ls_df = pd.DataFrame(bray_ls_mat, index=common_index, columns=common_index)
bray_ls_path = os.path.join(CSV_DIR, "bray_curtis_locality_season_envsubset.csv")
bray_ls_df.to_csv(bray_ls_path)
print("Saved Bray–Curtis (locality × season, env subset) to:", bray_ls_path)

env_data = np.sqrt(abiotic_env_valid[env_vars])
env_data = env_data.replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy="mean")
env_mat = imputer.fit_transform(env_data)

if np.isnan(env_mat).any():
    raise ValueError("NaNs remain in env_mat after imputation.")

env_dist = pdist(env_mat, metric="euclidean")
env_dist_mat = squareform(env_dist)

env_dist_df = pd.DataFrame(env_dist_mat, index=common_index, columns=common_index)
env_dist_path = os.path.join(CSV_DIR, "euclidean_envdist_locality_season.csv")
env_dist_df.to_csv(env_dist_path)
print("Saved Euclidean environmental distance matrix to:", env_dist_path)



if len(common_index) >= 3:
    if SKBIO_MANTEL:
        dm_bray = DistanceMatrix(bray_ls_mat, ids=[str(i) for i in range(bray_ls_mat.shape[0])])
        dm_env = DistanceMatrix(env_dist_mat, ids=[str(i) for i in range(env_dist_mat.shape[0])])

        mantel_res = mantel(dm_bray, dm_env, method="pearson", permutations=999)
        mantel_r, mantel_p, mantel_perm = mantel_res
        print("\nMantel test (Bray–Curtis vs abiotic Euclidean):")
        print(f"  r = {mantel_r:.4f}, p = {mantel_p:.6f}, permutations = {mantel_perm}")

        with open(os.path.join(CSV_DIR, "mantel_bray_env_locality_season.txt"), "w") as f:
            f.write("Mantel test (Bray–Curtis vs abiotic Euclidean)\n")
            f.write(f"r = {mantel_r:.4f}\n")
            f.write(f"p = {mantel_p:.6f}\n")
            f.write(f"permutations = {mantel_perm}\n")
    else:
        iu = np.triu_indices_from(bray_ls_mat, k=1)
        bc_vec = bray_ls_mat[iu]
        env_vec = env_dist_mat[iu]
        r, p = pearsonr(bc_vec, env_vec)
        print("\nMantel-like Pearson correlation (no permutations):")
        print(f"  r = {r:.4f}, p = {p:.6f}")

        with open(os.path.join(CSV_DIR, "mantel_like_bray_env_locality_season.txt"), "w") as f:
            f.write("Mantel-like Pearson correlation (Bray–Curtis vs abiotic Euclidean)\n")
            f.write(f"r = {r:.4f}\n")
            f.write(f"p = {p:.6f}\n")
else:
    print("Not enough locality × season combinations for Mantel test.")



bio_env = (
    sample_metrics
    .groupby(["locality", "season"], dropna=False)
    .agg(
        mean_density=("cell_density", "mean"),
        mean_richness=("taxa_richness", "mean")
    )
    .reset_index()
    .set_index(["locality", "season"])
)

bio_env = bio_env.loc[common_index]

bio_env_full = bio_env.join(abiotic_env_valid[env_vars])
bio_env_full["log10_mean_density"] = np.log10(bio_env_full["mean_density"] + 1)



print("\nUnivariate regressions vs environment")

uni_density_rows = []
uni_richness_rows = []

for v in env_vars:
    x = bio_env_full[v].values

    if np.isnan(x).sum() > 0.5 * len(x):
        continue

    mask_d = ~np.isnan(x) & ~np.isnan(bio_env_full["log10_mean_density"].values)
    if mask_d.sum() >= 5:
        X_d = sm.add_constant(x[mask_d])
        y_d = bio_env_full["log10_mean_density"].values[mask_d]
        model_d = sm.OLS(y_d, X_d).fit()
        uni_density_rows.append({
            "variable": v,
            "n": int(mask_d.sum()),
            "slope": model_d.params[1],
            "intercept": model_d.params[0],
            "p_value": model_d.pvalues[1],
            "r_squared": model_d.rsquared
        })

    mask_r = ~np.isnan(x) & ~np.isnan(bio_env_full["mean_richness"].values)
    if mask_r.sum() >= 5:
        X_r = sm.add_constant(x[mask_r])
        y_r = bio_env_full["mean_richness"].values[mask_r]
        model_r = sm.OLS(y_r, X_r).fit()
        uni_richness_rows.append({
            "variable": v,
            "n": int(mask_r.sum()),
            "slope": model_r.params[1],
            "intercept": model_r.params[0],
            "p_value": model_r.pvalues[1],
            "r_squared": model_r.rsquared
        })

uni_density_df = pd.DataFrame(uni_density_rows).sort_values("p_value")
uni_richness_df = pd.DataFrame(uni_richness_rows).sort_values("p_value")

uni_density_path = os.path.join(CSV_DIR, "univariate_env_log10density.csv")
uni_richness_path = os.path.join(CSV_DIR, "univariate_env_richness.csv")
uni_density_df.to_csv(uni_density_path, index=False)
uni_richness_df.to_csv(uni_richness_path, index=False)

print("Saved univariate regressions:")
print("  log10 density vs env ->", uni_density_path)
print("  richness vs env      ->", uni_richness_path)



def forward_stepwise(df, response, candidate_vars, max_vars=6, p_enter=0.05):
    """
    Simple forward stepwise selection based on p-values.
    """
    selected = []
    remaining = candidate_vars.copy()
    best_model = None

    while len(selected) < max_vars and len(remaining) > 0:
        best_p = None
        best_var = None

        for v in remaining:
            X_vars = selected + [v]
            X = sm.add_constant(df[X_vars].values)
            y = df[response].values
            model = sm.OLS(y, X, missing="drop").fit()
            p_v = model.pvalues[-1]  # p-value of last-added variable
            if best_p is None or p_v < best_p:
                best_p = p_v
                best_var = v
                best_model = model

        if best_p is not None and best_p < p_enter:
            selected.append(best_var)
            remaining.remove(best_var)
        else:
            break

    return selected, best_model

step_df = bio_env_full.copy()
step_df = step_df.dropna(subset=["log10_mean_density", "mean_richness"])

candidate_vars = [v for v in env_vars if step_df[v].notna().sum() >= 0.8 * len(step_df)]
print("\nNumber of candidate env variables for stepwise:", len(candidate_vars))

if len(candidate_vars) > 0 and len(step_df) >= 10:
    sel_dens, model_dens = forward_stepwise(step_df, "log10_mean_density",
                                            candidate_vars, max_vars=6, p_enter=0.05)
    print("\nStepwise multiple regression (log10 density):")
    print("Selected variables:", sel_dens)
    if model_dens is not None:
        print(model_dens.summary())
        with open(os.path.join(CSV_DIR, "stepwise_log10density_summary.txt"), "w") as f:
            f.write(str(model_dens.summary()))

    sel_rich, model_rich = forward_stepwise(step_df, "mean_richness",
                                            candidate_vars, max_vars=6, p_enter=0.05)
    print("\nStepwise multiple regression (richness):")
    print("Selected variables:", sel_rich)
    if model_rich is not None:
        print(model_rich.summary())
        with open(os.path.join(CSV_DIR, "stepwise_richness_summary.txt"), "w") as f:
            f.write(str(model_rich.summary()))
else:
    print("Not enough candidate variables or observations for stepwise regression.")



print("\nCorrelating Bray–Curtis with environmental distances")

iu = np.triu_indices_from(bray_ls_mat, k=1)
bc_vec = bray_ls_mat[iu]

rows_envdist = []

vars_env = [
    "decimalLatitude",
    "decimalLongitude",
    "Area (km2)",
    "Outlet length (km)",
    "Outlet width (km)",
    "Sinuosity index",
    "Depth",
    "Salinity Gradient"
]

vars_env = [v for v in vars_env if v in abiotic_env_valid.columns]

for v in vars_env:
    X = abiotic_env_valid[[v]].values.astype(float)

    col = X[:, 0]
    if np.isnan(col).any():
        col[np.isnan(col)] = np.nanmean(col)
        X[:, 0] = col

    dist_v = pdist(X, metric="euclidean")

    r, p = pearsonr(bc_vec, dist_v)

    rows_envdist.append({
        "environmental_variable": v,
        "pearson_r": r,
        "p_value": p
    })

df_envdist = pd.DataFrame(rows_envdist).sort_values("p_value")

out_path = os.path.join(CSV_DIR, "bray_vs_environment_distances.csv")
df_envdist.to_csv(out_path, index=False)
print("Saved Bray–Curtis vs environment distances to:", out_path)


print("\nBuilding Table 4 (Bray–Curtis vs individual env distances)")

iu = np.triu_indices_from(bray_ls_mat, k=1)
bc_vec = bray_ls_mat[iu]

table4_rows = []

vars_for_table4 = [
    "decimalLatitude",
    "decimalLongitude",
    "Area (km2)",
    "Outlet length (km)",
    "Outlet width (km)",
    "Sinuosity index",
    "Depth",
    "Salinity Gradient"
]

vars_for_table4 = [v for v in vars_for_table4 if v in abiotic_env_valid.columns]

for v in vars_for_table4:
    X = abiotic_env_valid[[v]].values.astype(float)

    col = X[:, 0]
    if np.isnan(col).any():
        col[np.isnan(col)] = np.nanmean(col)
        X[:, 0] = col

    dist_v = pdist(X, metric="euclidean")

    r, p = pearsonr(bc_vec, dist_v)

    table4_rows.append({
        "variable": v,
        "r": r,
        "p": p
    })

table4 = pd.DataFrame(table4_rows).sort_values("p")
table4_path = os.path.join(CSV_DIR, "table4_bray_envdist_correlations.csv")
table4.to_csv(table4_path, index=False)
print("Saved Table 4 (Bray–Curtis vs individual env distances) to:", table4_path)

print("\nHierarchical clustering and dendrograms")

labels = [f"{loc}_{season}" for (loc, season) in common_index]

Z_env = linkage(env_dist, method="average")

plt.figure(figsize=(12, 6))
dendrogram(Z_env, labels=labels, leaf_rotation=90, leaf_font_size=8)
plt.ylabel("Euclidean distance (abiotic)")
plt.title("Hierarchical clustering of ecosystems based on abiotic variables")
plt.tight_layout()
dendro_env_path = os.path.join(PLOTS_DIR, "dendrogram_abiotic_locality_season.png")
plt.savefig(dendro_env_path, dpi=300)
plt.close()
print("Saved abiotic dendrogram to:", dendro_env_path)

Z_bio = linkage(bray_ls, method="average")

plt.figure(figsize=(12, 6))
dendrogram(Z_bio, labels=labels, leaf_rotation=90, leaf_font_size=8)
plt.ylabel("Bray–Curtis dissimilarity")
plt.title("Hierarchical clustering of ecosystems based on phytoplankton composition")
plt.tight_layout()
dendro_bio_path = os.path.join(PLOTS_DIR, "dendrogram_biotic_locality_season.png")
plt.savefig(dendro_bio_path, dpi=300)
plt.close()
print("Saved biotic dendrogram to:", dendro_bio_path)



print("\nPlotting multi-panel Bray–Curtis vs environmental distances")


iu = np.triu_indices_from(bray_ls_mat, k=1)
bc_vec = bray_ls_mat[iu]

var_map = {
    "decimalLongitude": "decimalLongitude",
    "decimalLatitude": "decimalLatitude",
    "Outlet length": "Outlet length (km)",
    "Area": "Area (km2)",
    "Outlet width": "Outlet width (km)",
    "Depth": "Depth",
    "Salinity": "Salinity Gradient",   # or "Salinity" if you really have that column
    "Sinuosity index": "Sinuosity index",
}

labels = list(var_map.keys())
n_vars = len(labels)
n_cols = 4
n_rows = 2

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4 * n_cols, 3.2 * n_rows),
    sharey=True
)
axes = np.atleast_2d(axes)

for idx, label in enumerate(labels):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]

    colname = var_map[label]
    if colname not in abiotic_env_valid.columns:
        ax.text(0.5, 0.5, f"{label}\nnot found",
                ha="center", va="center", fontsize=8)
        ax.axis("off")
        print(f"Warning: column '{colname}' not found for label '{label}'.")
        continue

    X = abiotic_env_valid[[colname]].values.astype(float)
    col_data = X[:, 0]
    if np.isnan(col_data).any():
        col_data[np.isnan(col_data)] = np.nanmean(col_data)
        X[:, 0] = col_data

    dist_v = pdist(X, metric="euclidean")

    r, p = pearsonr(bc_vec, dist_v)

    ax.scatter(dist_v, bc_vec, alpha=0.5, s=10)
    X_reg = sm.add_constant(dist_v)
    model_reg = sm.OLS(bc_vec, X_reg).fit()
    x_line = np.linspace(dist_v.min(), dist_v.max(), 100)
    y_line = model_reg.params[0] + model_reg.params[1] * x_line
    ax.plot(x_line, y_line, color="black", linewidth=1)

    ax.set_title(label, fontsize=9)
    text_str = f"r = {r:.2f}\np = {p:.1e}"
    ax.text(
        0.05, 0.95, text_str,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    if row == n_rows - 1:
        ax.set_xlabel("Distance", fontsize=9)
    if col == 0:
        ax.set_ylabel("Bray–Curtis", fontsize=9)

plt.tight_layout()
fig_path = os.path.join(PLOTS_DIR, "bray_vs_envdist_multipanel_8vars.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print("Saved multi-panel distance–distance plot (8 vars) to:", fig_path)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
