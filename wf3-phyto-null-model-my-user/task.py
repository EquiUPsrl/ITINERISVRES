from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import gaussian_kde
import os
import pandas as pd
from sklearn.linear_model import PoissonRegressor
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--bio_file_filtered', action='store', type=str, required=True, dest='bio_file_filtered')

arg_parser.add_argument('--biotic_size_class', action='store', type=str, required=True, dest='biotic_size_class')

arg_parser.add_argument('--cca2_site_scores_path', action='store', type=str, required=True, dest='cca2_site_scores_path')


args = arg_parser.parse_args()
print(args)

id = args.id

bio_file_filtered = args.bio_file_filtered.replace('"','')
biotic_size_class = args.biotic_size_class.replace('"','')
cca2_site_scores_path = args.cca2_site_scores_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

output_dir = os.path.join(conf_output_path, "NULL_MODEL")
os.makedirs(output_dir, exist_ok=True)


bio_file         = bio_file_filtered
site_scores_file = cca2_site_scores_path   # site scores from CCA model in R
sizeclass_file   = biotic_size_class  # sizeClass per species

EXCLUDE_SCI = "Phytoflagellate undetermined 1"      # optional




bio = pd.read_csv(
    bio_file,
    sep=",",
    na_values=["", "NA", "NaN", "na", "-"],
    encoding="utf-8"
)

if "ID" in bio.columns:
    bio.set_index("ID", inplace=True)

bio = bio.apply(pd.to_numeric, errors="coerce")

print("bio shape (campioni x specie):", bio.shape)




sites = pd.read_csv(site_scores_file)

if "SampleID" in sites.columns:
    sites.set_index("SampleID", inplace=True)

candidates_axis1 = ["CCA1_std", "CCA1", "Axis1", "CCA1_raw"]
axis1_col = next((c for c in candidates_axis1 if c in sites.columns), None)

if axis1_col is None:
    raise ValueError(
        "I can't find the column for the canonical axis 1 in"
        f"{site_scores_file}. Columns present: {sites.columns.tolist()}"
    )

common_ids = bio.index.intersection(sites.index)
bio   = bio.loc[common_ids].copy()
sites = sites.loc[common_ids].copy()

bio = bio.loc[bio.sum(axis=1) > 0]
sites = sites.loc[bio.index]

axis1 = sites[axis1_col].values  # np.array
print("Samples used for RA3-like:", bio.shape[0])



sc = pd.read_csv(sizeclass_file, sep=",", encoding="utf-8")

if "scientificName" not in sc.columns or "sizeClass" not in sc.columns:
    raise ValueError(
        f"I can't find 'scientificName' or 'sizeClass' in {sizeclass_file}. "
        f"Columns present: {sc.columns.tolist()}"
    )

sc_simple = sc[["scientificName", "sizeClass"]].dropna().copy()
sc_simple["scientificName"] = sc_simple["scientificName"].astype(str)

mask_excl_sc = sc_simple["scientificName"].str.contains(
    EXCLUDE_SCI, case=False, na=False
)
sc_simple = sc_simple[~mask_excl_sc].copy()

def normalize_name(s):
    return s.lower().replace(" ", "").replace(".", "")

sc_simple["norm_name"] = sc_simple["scientificName"].apply(normalize_name)

sc_group = (
    sc_simple.groupby("norm_name")
    .agg({
        "scientificName": lambda x: x.iloc[0],
        "sizeClass":     lambda x: x.value_counts().index[0]
    })
    .reset_index()
)

bio_cols = pd.DataFrame({"bio_name": bio.columns})
bio_cols["norm_name"] = bio_cols["bio_name"].astype(str).apply(normalize_name)

merge_sc = pd.merge(
    bio_cols,
    sc_group[["norm_name", "scientificName", "sizeClass"]],
    on="norm_name",
    how="inner"
)

print("Species in bio:", bio.shape[1])
print("Species with assigned sizeClass:", merge_sc["bio_name"].nunique())

taxa_info = merge_sc.set_index("bio_name")[["sizeClass", "scientificName"]].copy()
taxa_info.columns = ["SizeClass", "Taxa"]

bio = bio.loc[:, taxa_info.index]
species_names = bio.columns.tolist()
Y = bio.values  # matrix sites x species

size_classes = taxa_info["SizeClass"].astype(str).reindex(species_names).values

mask_phyto = taxa_info["Taxa"].str.contains(EXCLUDE_SCI, case=False, na=False)
if mask_phyto.any():
    to_drop = taxa_info.index[mask_phyto]
    bio = bio.drop(columns=to_drop)
    species_names = bio.columns.tolist()
    Y = bio.values
    size_classes = taxa_info.loc[species_names, "SizeClass"].astype(str).values
    taxa_info = taxa_info.loc[species_names]



poly_global = PolynomialFeatures(degree=2, include_bias=False)

def fit_poisson_quadratic_sklearn(x, y):
    """
    GLM Poisson with polynomial(x, degree=2).
    Returns model or None if problems occur.
    """
    mask = ~np.isnan(y) & ~np.isnan(x)
    x_valid = x[mask]
    y_valid = y[mask]

    if y_valid.size == 0 or np.all(y_valid == 0):
        return None

    w = np.sqrt(y_valid + 1.0)

    X_poly = poly_global.fit_transform(x_valid.reshape(-1, 1))

    model = PoissonRegressor(alpha=0.0, max_iter=1000)
    try:
        model.fit(X_poly, y_valid, sample_weight=w)
    except Exception:
        return None

    return model


def extract_opt_tol_from_model(model, window=(-3.0, 3.0)):
    """
    Extracts optimum/tolerance from GLM:
    log(mu) = b0 + b1 x + b2 x^2
    """
    if model is None:
        return dict(optimum=np.nan, tolerance=np.nan, unimodal=False)

    coefs = model.coef_
    if coefs.size < 2:
        return dict(optimum=np.nan, tolerance=np.nan, unimodal=False)

    b1, b2 = coefs[0], coefs[1]

    if np.isnan(b2) or b2 >= 0:
        return dict(optimum=np.nan, tolerance=np.nan, unimodal=False)

    opt = -b1 / (2.0 * b2)
    tol = np.sqrt(-1.0 / (2.0 * b2))

    w_min, w_max = window
    unimodal = (w_min <= opt <= w_max)

    return dict(optimum=opt, tolerance=tol, unimodal=unimodal)




def compute_niche_glm(Y_mat, axis1, species_names, size_classes, taxa_info,
                      window=(-3.0, 3.0)):
    """
    For each species:
     - GLM Quadratic Poisson on Axis 1
     - Optimum, Tolerance
     - xmin, xmax
     - Total Abundance
    """
    n_sites, n_species = Y_mat.shape
    records = []

    for j in range(n_species):
        y = Y_mat[:, j]
        total = np.nansum(y)
        if total <= 0:
            continue

        model = fit_poisson_quadratic_sklearn(axis1, y)
        ot = extract_opt_tol_from_model(model, window=window)
        if (not ot["unimodal"]) or np.isnan(ot["optimum"]) or np.isnan(ot["tolerance"]):
            continue

        mu  = ot["optimum"]
        tol = ot["tolerance"]
        xmin = mu - tol / 2.0
        xmax = mu + tol / 2.0

        sp_name = species_names[j]
        cls = str(size_classes[j])
        sci = taxa_info.loc[sp_name, "Taxa"]

        records.append({
            "bio_name": sp_name,
            "scientificName": sci,
            "sizeClass": cls,
            "optimum": mu,
            "tolerance": tol,
            "xmin": xmin,
            "xmax": xmax,
            "total_abundance": total
        })

    return pd.DataFrame(records)




def overlap(row_i, row_j):
    xi_min, xi_max = row_i["xmin"], row_i["xmax"]
    xj_min, xj_max = row_j["xmin"], row_j["xmax"]
    inter = max(0.0, min(xi_max, xj_max) - max(xi_min, xj_min))
    width_i = xi_max - xi_min
    return inter / width_i if width_i > 0 else 0.0




niche_obs = compute_niche_glm(Y, axis1, species_names, size_classes, taxa_info)
print("Species with valid GLM niche (observed):", len(niche_obs))

class_counts = niche_obs.groupby("sizeClass")["scientificName"].nunique()
valid_classes = class_counts[class_counts >= 3].index.tolist()
niche_obs = niche_obs[niche_obs["sizeClass"].isin(valid_classes)].copy()
print("Size classes used:", valid_classes)

observed_overlaps = []

for cls, g in niche_obs.groupby("sizeClass"):
    g = g.sort_values("total_abundance", ascending=False)
    if len(g) >= 2:
        r1 = g.iloc[0]
        r2 = g.iloc[1]
        observed_overlaps.append(overlap(r1, r2))

obs_global = np.mean(observed_overlaps)
print(f"Global observed overlap (average top-2 per size class) = {obs_global:.4f}")



N_SIM = 500  # set iteration numbers
simulated_globals = []

for sim in range(N_SIM):
    Y_sim = Y.copy()
    for j in range(Y_sim.shape[1]):
        Y_sim[:, j] = np.random.permutation(Y_sim[:, j])

    niche_sim = compute_niche_glm(
        Y_sim, axis1, species_names, size_classes, taxa_info,
        window=(-3.0, 3.0)
    )

    niche_sim = niche_sim[niche_sim["sizeClass"].isin(valid_classes)].copy()

    sim_overlaps = []
    for cls, g in niche_sim.groupby("sizeClass"):
        g = g.sort_values("total_abundance", ascending=False)
        if len(g) >= 2:
            r1 = g.iloc[0]
            r2 = g.iloc[1]
            sim_overlaps.append(overlap(r1, r2))

    if len(sim_overlaps) == 0:
        continue

    simulated_globals.append(np.mean(sim_overlaps))

simulated_globals = np.array(simulated_globals)
print("Valid simulations:", len(simulated_globals))




mean_sim = simulated_globals.mean()
sd_sim   = simulated_globals.std(ddof=1)

count_le = np.sum(simulated_globals <= obs_global)
p_left = (count_le + 1) / (len(simulated_globals) + 1)

count_ge = np.sum(simulated_globals >= obs_global)
p_right = (count_ge + 1) / (len(simulated_globals) + 1)

SES = (obs_global - mean_sim) / sd_sim

print("\n====== GLOBAL STATISTICS (RA3-like GLM, random Y) ======")
print(f"Observed mean overlap   = {obs_global:.4f}")
print(f"Simulated mean overlap  = {mean_sim:.4f}")
print(f"Simulated SD overlap    = {sd_sim:.4f}")
print(f"SES (obs - mean_sim)/sd = {SES:.4f}")
print(f"p-value (sim <= obs)    = {p_left:.4f}")
print(f"p-value (sim >= obs)    = {p_right:.4f}")
print("==========================================================\n")

pd.DataFrame({
    "Observed_global_overlap": [obs_global],
    "Simulated_mean": [mean_sim],
    "Simulated_sd": [sd_sim],
    "SES": [SES],
    "p_value_left_tail": [p_left],
    "p_value_right_tail": [p_right],
    "n_size_classes": [len(valid_classes)],
    "n_simulations": [len(simulated_globals)]
}).to_csv(os.path.join(output_dir, "axis1_nullmodel_RA3like_GLM_full_stats.csv"), index=False)




plt.figure(figsize=(9, 5))
kde = gaussian_kde(simulated_globals)
x = np.linspace(simulated_globals.min(), simulated_globals.max(), 400)
plt.plot(x, kde(x), label="Null model (KDE)", color="blue")

plt.axvline(mean_sim, color="blue", linestyle="--",
            label=f"Media simulata = {mean_sim:.3f}")
plt.axvline(obs_global, color="red", linewidth=2,
            label=f"Osservato = {obs_global:.3f}")

plt.title(f"Axis 1 – RA3-like GLM-based null model\n"
          f"SES = {SES:.2f}, p_left = {p_left:.4f}")
plt.xlabel("Medium overlap (top-2 species / size class)")
plt.ylabel("Kernel density")
plt.legend()
plt.tight_layout()

fig_name = os.path.join(output_dir, "null_model_kde_chart.png")
plt.savefig(fig_name, dpi=300)
print(f"Figure saved as: {fig_name}")

plt.show()



def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

x_ecdf, y_ecdf = ecdf(simulated_globals)
obs_ecdf_y = (np.sum(simulated_globals <= obs_global) + 1) / (N_SIM + 1)

plt.figure(figsize=(9, 5))
plt.step(x_ecdf, y_ecdf, where='post', color="black", label="ECDF simulations")
plt.axvline(obs_global, color="red", lw=2, label=f"Observed = {obs_global:.3f}")

plt.xlabel("Medium Overlap (top-2 species/class)")
plt.ylabel("Cumulative probability")
plt.title("ECDF – Null model RA3-like (Axis 1)\nGLM-based niches")
plt.grid(alpha=.3)
plt.legend()
plt.tight_layout()

fig_name = os.path.join(output_dir, "null_model_ecdf_chart.png")
plt.savefig(fig_name, dpi=300)
print(f"Figure saved as: {fig_name}")

plt.show()

output_dir = conf_output_path

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
