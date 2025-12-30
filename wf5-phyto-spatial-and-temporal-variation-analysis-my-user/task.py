from scipy.stats import pearsonr
import os
import pandas as pd
import numpy as np
from scipy.stats import levene
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro
import statsmodels.formula.api as smf

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


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'
conf_temp_path = conf_temp_path = '/tmp/data/WF5/' + 'tmp'

input_file = filtered_file
OUTPUT_DIR = conf_output_path
TMP_DIR = conf_temp_path
CSV_DIR    = os.path.join(OUTPUT_DIR, "csv")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(CSV_DIR, exist_ok=True)

df_species = pd.read_csv(input_file, low_memory=False)


sample_id_cols = [
    col for col in [
        "country", "locality", "year", "month",
        "parentEventID", "eventID", "season"
    ]
    if col in df_species.columns
]

print("\nSample ID columns:", sample_id_cols)

def compute_diversity(group, include_groups=False, **kwargs):
    """
    Compute Shannon, Simpson (1 - D) and Pielou evenness for a single sample
    using species-level abundances in column 'density'.
    """
    abund = group["density"].values.astype(float)
    N = abund.sum()

    if N <= 0:
        return pd.Series(
            {
                "shannon": np.nan,
                "simpson": np.nan,
                "pielou_evenness": np.nan,
            }
        )

    p = abund / N

    H = -np.sum(p * np.log(p))

    D = np.sum(p ** 2)
    simpson_1D = 1.0 - D

    S = (abund > 0).sum()
    if S > 1:
        J = H / np.log(S)
    else:
        J = np.nan

    return pd.Series(
        {
            "shannon": H,
            "simpson": simpson_1D,
            "pielou_evenness": J,
        }
    )

agg_sample = {
    "cell_density": ("density", "sum"),
    "totalBiovolume": ("totalBiovolume", "sum"),
    "totalBiomass": ("totalBiomass", "sum"),
    "totalCarbonContent": ("totalCarbonContent", "sum"),
    "taxa_richness": ("acceptedNameUsage", "nunique"),
}

for tax in ["genus", "family", "order", "class", "phylum"]:
    if tax in df_species.columns:
        agg_sample[f"{tax}_richness"] = (tax, "nunique")

sample_basic = (
    df_species
    .groupby(sample_id_cols, dropna=False)
    .agg(**agg_sample)
    .reset_index()
)

if "shape" in df_species.columns:
    shape_richness = (
        df_species
        .groupby(sample_id_cols, dropna=False)["shape"]
        .nunique()
        .reset_index(name="shape_richness")
    )
else:
    shape_richness = sample_basic[sample_id_cols].copy()
    shape_richness["shape_richness"] = np.nan

diversity_metrics = (
    df_species
    .groupby(sample_id_cols, dropna=False)
    .apply(compute_diversity, include_groups=False)
    .reset_index()
)

sample_metrics = (
    sample_basic
    .merge(diversity_metrics, on=sample_id_cols, how="left")
    .merge(shape_richness, on=sample_id_cols, how="left")
)

sample_metrics["log10_cell_density"] = np.log10(sample_metrics["cell_density"] + 1)

sample_metrics_file = os.path.join(CSV_DIR, "sample_metrics.csv")
sample_metrics.to_csv(sample_metrics_file, index=False)
print("\nSample-level metrics saved to:", sample_metrics_file)
print("Sample-level metrics (head):")
print(sample_metrics.head())




anova_loc_df = sample_metrics.dropna(
    subset=["log10_cell_density", "taxa_richness", "locality", "season"]
).copy()
anova_loc_df["locality"] = anova_loc_df["locality"].astype("category")
anova_loc_df["season"]   = anova_loc_df["season"].astype("category")

print("\nANOVA (locality × season) dataset size:", len(anova_loc_df))

if "country" in sample_metrics.columns:
    anova_country_df = sample_metrics.dropna(
        subset=["log10_cell_density", "taxa_richness", "country", "season"]
    ).copy()
    anova_country_df["country"] = anova_country_df["country"].astype("category")
    anova_country_df["season"]  = anova_country_df["season"].astype("category")
    print("ANOVA (country × season) dataset size:", len(anova_country_df))
else:
    anova_country_df = None
    print("Column 'country' not found in sample_metrics; skipping country × season ANOVA.")

if {"country", "locality", "season"}.issubset(sample_metrics.columns):
    anova_nested_df = sample_metrics.dropna(
        subset=["log10_cell_density", "taxa_richness", "country", "locality", "season"]
    ).copy()
    anova_nested_df["country"]  = anova_nested_df["country"].astype("category")
    anova_nested_df["locality"] = anova_nested_df["locality"].astype("category")
    anova_nested_df["season"]   = anova_nested_df["season"].astype("category")
    print("Nested ANOVA dataset size:", len(anova_nested_df))
else:
    anova_nested_df = None
    print("Nested ANOVA not possible: missing country/locality/season.")



levene_results = []

groups_season = [
    g["log10_cell_density"].values
    for _, g in anova_loc_df.groupby("season", observed=False)
    if len(g) > 1
]
if len(groups_season) >= 2:
    stat, p = levene(*groups_season)
    print(f"\nLevene test (log10_cell_density ~ season): stat={stat:.4f}, p={p:.6f}")
    levene_results.append(
        {"test": "Levene_density_season", "stat": stat, "pvalue": p}
    )

groups_locality = [
    g["log10_cell_density"].values
    for _, g in anova_loc_df.groupby("locality", observed=False)
    if len(g) > 1
]
if len(groups_locality) >= 2:
    stat, p = levene(*groups_locality)
    print(f"Levene test (log10_cell_density ~ locality): stat={stat:.4f}, p={p:.6f}")
    levene_results.append(
        {"test": "Levene_density_locality", "stat": stat, "pvalue": p}
    )



model_density_loc_season = smf.ols(
    "log10_cell_density ~ C(locality) * C(season)",
    data=anova_loc_df
).fit()
anova_density_loc_season = sm.stats.anova_lm(model_density_loc_season, typ=2)
print("\nTwo-way ANOVA for log10_cell_density (locality × season):")
print(anova_density_loc_season)

model_richness_loc_season = smf.ols(
    "taxa_richness ~ C(locality) * C(season)",
    data=anova_loc_df
).fit()
anova_richness_loc_season = sm.stats.anova_lm(model_richness_loc_season, typ=2)
print("\nTwo-way ANOVA for taxa_richness (locality × season):")
print(anova_richness_loc_season)



if anova_country_df is not None and len(anova_country_df) > 0:
    model_density_country_season = smf.ols(
        "log10_cell_density ~ C(country) * C(season)",
        data=anova_country_df
    ).fit()
    anova_density_country_season = sm.stats.anova_lm(model_density_country_season, typ=2)
    print("\nTwo-way ANOVA for log10_cell_density (country × season):")
    print(anova_density_country_season)

    model_richness_country_season = smf.ols(
        "taxa_richness ~ C(country) * C(season)",
        data=anova_country_df
    ).fit()
    anova_richness_country_season = sm.stats.anova_lm(model_richness_country_season, typ=2)
    print("\nTwo-way ANOVA for taxa_richness (country × season):")
    print(anova_richness_country_season)
else:
    model_density_country_season = None
    model_richness_country_season = None
    anova_density_country_season = None
    anova_richness_country_season = None
    print("\nSkipping country × season ANOVA (no data).")



if anova_nested_df is not None and len(anova_nested_df) > 0:
    model_density_nested = smf.ols(
        "log10_cell_density ~ C(season) + C(country) / C(locality)",
        data=anova_nested_df
    ).fit()
    anova_density_nested = sm.stats.anova_lm(model_density_nested, typ=2)
    print("\nNested ANOVA for log10_cell_density (country/locality + season):")
    print(anova_density_nested)

    model_richness_nested = smf.ols(
        "taxa_richness ~ C(season) + C(country) / C(locality)",
        data=anova_nested_df
    ).fit()
    anova_richness_nested = sm.stats.anova_lm(model_richness_nested, typ=2)
    print("\nNested ANOVA for taxa_richness (country/locality + season):")
    print(anova_richness_nested)
else:
    model_density_nested = None
    model_richness_nested = None
    anova_density_nested = None
    anova_richness_nested = None
    print("\nSkipping nested ANOVA (no data).")



shapiro_results = []

def shapiro_on_model(model, name):
    if model is None:
        return
    resid = model.resid
    if len(resid) >= 3:
        stat, p = shapiro(resid)
        print(f"\nShapiro–Wilk for residuals of {name}: stat={stat:.4f}, p={p:.6f}")
        shapiro_results.append(
            {"model": name, "stat": stat, "pvalue": p}
        )

shapiro_on_model(model_density_loc_season, "log10_density_locality_season")
shapiro_on_model(model_richness_loc_season, "taxa_richness_locality_season")
shapiro_on_model(model_density_country_season, "log10_density_country_season")
shapiro_on_model(model_richness_country_season, "taxa_richness_country_season")
shapiro_on_model(model_density_nested, "log10_density_nested")
shapiro_on_model(model_richness_nested, "taxa_richness_nested")



STATS_DIR = os.path.join(OUTPUT_DIR, "statistics")
os.makedirs(STATS_DIR, exist_ok=True)
print("\nStatistics output folder:", STATS_DIR)

levene_df = pd.DataFrame(levene_results)
levene_df.to_csv(os.path.join(STATS_DIR, "levene_tests.csv"), index=False)
print("Saved: levene_tests.csv")

shapiro_df = pd.DataFrame(shapiro_results)
shapiro_df.to_csv(os.path.join(STATS_DIR, "shapiro_tests.csv"), index=False)
print("Saved: shapiro_tests.csv")

anova_density_loc_season.to_csv(
    os.path.join(STATS_DIR, "anova_log10density_locality_season.csv")
)
anova_richness_loc_season.to_csv(
    os.path.join(STATS_DIR, "anova_taxarichness_locality_season.csv")
)

if anova_density_country_season is not None:
    anova_density_country_season.to_csv(
        os.path.join(STATS_DIR, "anova_log10density_country_season.csv")
    )
if anova_richness_country_season is not None:
    anova_richness_country_season.to_csv(
        os.path.join(STATS_DIR, "anova_taxarichness_country_season.csv")
    )

if anova_density_nested is not None:
    anova_density_nested.to_csv(
        os.path.join(STATS_DIR, "anova_log10density_nested_country_locality_season.csv")
    )
if anova_richness_nested is not None:
    anova_richness_nested.to_csv(
        os.path.join(STATS_DIR, "anova_taxarichness_nested_country_locality_season.csv")
    )

print("ANOVA tables saved.")


with open(os.path.join(STATS_DIR, "model_log10density_locality_season_summary.txt"), "w") as f:
    f.write(model_density_loc_season.summary().as_text())

with open(os.path.join(STATS_DIR, "model_taxarichness_locality_season_summary.txt"), "w") as f:
    f.write(model_richness_loc_season.summary().as_text())

if model_density_country_season is not None:
    with open(os.path.join(STATS_DIR, "model_log10density_country_season_summary.txt"), "w") as f:
        f.write(model_density_country_season.summary().as_text())

if model_richness_country_season is not None:
    with open(os.path.join(STATS_DIR, "model_taxarichness_country_season_summary.txt"), "w") as f:
        f.write(model_richness_country_season.summary().as_text())

if model_density_nested is not None:
    with open(os.path.join(STATS_DIR, "model_log10density_nested_summary.txt"), "w") as f:
        f.write(model_density_nested.summary().as_text())

if model_richness_nested is not None:
    with open(os.path.join(STATS_DIR, "model_taxarichness_nested_summary.txt"), "w") as f:
        f.write(model_richness_nested.summary().as_text())

print("OLS model summaries saved.")



DESC_DIR = os.path.join(OUTPUT_DIR, "descriptive")
os.makedirs(DESC_DIR, exist_ok=True)

print("\nDescriptive output folder:", DESC_DIR)


desc_stats = (
    sample_metrics
    .groupby(["locality", "season"], dropna=False)
    .agg(
        n_samples=("cell_density", "count"),
        mean_density=("cell_density", "mean"),
        sd_density=("cell_density", "std"),
        min_density=("cell_density", "min"),
        max_density=("cell_density", "max"),
        mean_richness=("taxa_richness", "mean"),
        sd_richness=("taxa_richness", "std"),
        min_richness=("taxa_richness", "min"),
        max_richness=("taxa_richness", "max"),
    )
    .reset_index()
)

desc_path = os.path.join(DESC_DIR, "descriptive_locality_season.csv")
desc_stats.to_csv(desc_path, index=False)
print("Saved descriptive statistics:", desc_path)



density_sorted = desc_stats.sort_values("mean_density")

lowest_density = density_sorted.iloc[0]
highest_density = density_sorted.iloc[-1]


richness_sorted = desc_stats.sort_values("mean_richness")

lowest_richness = richness_sorted.iloc[0]
highest_richness = richness_sorted.iloc[-1]


summary_path = os.path.join(DESC_DIR, "locality_summary.txt")
with open(summary_path, "w") as f:
    f.write("=== Locality × Season Summary ===\n\n")

    f.write("Lowest mean cell density:\n")
    f.write(str(lowest_density) + "\n\n")

    f.write("Highest mean cell density:\n")
    f.write(str(highest_density) + "\n\n")

    f.write("Lowest mean taxon richness:\n")
    f.write(str(lowest_richness) + "\n\n")

    f.write("Highest mean taxon richness:\n")
    f.write(str(highest_richness) + "\n\n")

print("Saved locality summary:", summary_path)




sns.set(style="white", rc={"axes.grid": False})

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=sample_metrics,
    x="locality",
    y="log10_cell_density",
    hue="season"
)
plt.xticks(rotation=45, ha="right")
plt.ylabel(" density [cells L$^{-1}$]")
plt.tight_layout()
fig_path = os.path.join(PLOTS_DIR, "boxplot_log10_density_locality_season.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print("Saved figure:", fig_path)


plt.figure(figsize=(10, 6))
sns.boxplot(
    data=sample_metrics,
    x="locality",
    y="taxa_richness",
    hue="season"
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Species richness")
plt.tight_layout()
fig_path = os.path.join(PLOTS_DIR, "boxplot_richness_locality_season.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print("Saved figure:", fig_path)








print("\n=== SECTION 9: Density–richness relationships ===")

STATS_DIR = os.path.join(OUTPUT_DIR, "statistics")
os.makedirs(STATS_DIR, exist_ok=True)



rel_sample = sample_metrics.dropna(
    subset=["log10_cell_density", "taxa_richness", "season"]
).copy()

print("Sample-level data points:", len(rel_sample))

r_samp, p_samp = pearsonr(
    rel_sample["log10_cell_density"],
    rel_sample["taxa_richness"]
)

print(f"Sample-level Pearson r = {r_samp:.3f}, p = {p_samp:.4f}")

model_samp = smf.ols(
    "taxa_richness ~ log10_cell_density",
    data=rel_sample
).fit()

print("\nSample-level regression summary:")
print(model_samp.summary())

sample_stats = pd.DataFrame(
    [{
        "scale": "sample",
        "pearson_r": r_samp,
        "pearson_p": p_samp,
        "slope": model_samp.params.get("log10_cell_density", np.nan),
        "intercept": model_samp.params.get("Intercept", np.nan),
        "r_squared": model_samp.rsquared,
        "n": len(rel_sample)
    }]
)

sample_stats_path = os.path.join(STATS_DIR, "density_richness_sample_level.csv")
sample_stats.to_csv(sample_stats_path, index=False)
print("Saved:", sample_stats_path)



if "desc_stats" not in globals() or \
   not {"mean_density", "mean_richness"}.issubset(desc_stats.columns):

    desc_stats = (
        sample_metrics
        .groupby(["locality", "season"], dropna=False)
        .agg(
            n_samples=("cell_density", "count"),
            mean_density=("cell_density", "mean"),
            mean_richness=("taxa_richness", "mean"),
        )
        .reset_index()
    )

desc_stats["log10_mean_density"] = np.log10(desc_stats["mean_density"] + 1)

rel_ecosys = desc_stats.dropna(
    subset=["log10_mean_density", "mean_richness"]
).copy()

print("Ecosystem-level data points:", len(rel_ecosys))

r_eco, p_eco = pearsonr(
    rel_ecosys["log10_mean_density"],
    rel_ecosys["mean_richness"]
)

print(f"Ecosystem-level Pearson r = {r_eco:.3f}, p = {p_eco:.4f}")

model_eco = smf.ols(
    "mean_richness ~ log10_mean_density",
    data=rel_ecosys
).fit()

print("\nEcosystem-level regression summary:")
print(model_eco.summary())

ecosys_stats = pd.DataFrame(
    [{
        "scale": "ecosystem_locality_season",
        "pearson_r": r_eco,
        "pearson_p": p_eco,
        "slope": model_eco.params.get("log10_mean_density", np.nan),
        "intercept": model_eco.params.get("Intercept", np.nan),
        "r_squared": model_eco.rsquared,
        "n": len(rel_ecosys)
    }]
)

ecosys_stats_path = os.path.join(STATS_DIR, "density_richness_ecosystem_level.csv")
ecosys_stats.to_csv(ecosys_stats_path, index=False)
print("Saved:", ecosys_stats_path)



sns.set(style="white", rc={"axes.grid": False})


plt.figure(figsize=(7, 6))

sns.scatterplot(
    data=rel_sample,
    x="log10_cell_density",
    y="taxa_richness",
    hue="season",
    alpha=0.5
)

x_vals = np.linspace(
    rel_sample["log10_cell_density"].min(),
    rel_sample["log10_cell_density"].max(),
    200
)
slope_s = model_samp.params["log10_cell_density"]
intercept_s = model_samp.params["Intercept"]
y_vals = intercept_s + slope_s * x_vals
plt.plot(x_vals, y_vals, color="black", linewidth=2)

plt.xlabel("log10(density) [cells L$^{-1}$]")
plt.ylabel("Species richness")
plt.title("Density–richness relationship (sample level)")
plt.legend(frameon=False)

plt.ylim(bottom=0)

ax = plt.gca()
text_str = (
    f"y = {slope_s:.2f}x {intercept_s:+.2f}\n"
    f"r = {r_samp:.3f}, n = {len(rel_sample)}, p = {p_samp:.3g}"
)
ax.text(0.05, 0.95, text_str,
        transform=ax.transAxes,
        va="top", ha="left")

plt.tight_layout()
fig_path = os.path.join(PLOTS_DIR, "reg_density_richness_sample_level.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print("Saved figure:", fig_path)



plt.figure(figsize=(7, 6))

sns.scatterplot(
    data=rel_ecosys,
    x="log10_mean_density",
    y="mean_richness",
    hue="season",
    s=80
)

x_vals = np.linspace(
    rel_ecosys["log10_mean_density"].min(),
    rel_ecosys["log10_mean_density"].max(),
    200
)
slope_e = model_eco.params["log10_mean_density"]
intercept_e = model_eco.params["Intercept"]
y_vals = intercept_e + slope_e * x_vals
plt.plot(x_vals, y_vals, color="black", linewidth=2)

for _, row in rel_ecosys.iterrows():
    plt.text(
        row["log10_mean_density"],
        row["mean_richness"],
        row["locality"],
        fontsize=8,
        ha="left",
        va="center"
    )

plt.xlabel("log10(density) [cells L$^{-1}$]")
plt.ylabel("Mean species richness")
plt.title("Density–richness relationship (locality × season)")
plt.legend(frameon=False)

plt.ylim(bottom=0)

ax = plt.gca()
text_str = (
    f"y = {slope_e:.2f}x {intercept_e:+.2f}\n"
    f"r = {r_eco:.3f}, n = {len(rel_ecosys)}, p = {p_eco:.3g}"
)
ax.text(0.05, 0.95, text_str,
        transform=ax.transAxes,
        va="top", ha="left")

plt.tight_layout()
fig_path = os.path.join(PLOTS_DIR, "reg_density_richness_ecosystem_level.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print("Saved figure:", fig_path)

file_sample_metrics_file = open("/tmp/sample_metrics_file_" + id + ".json", "w")
file_sample_metrics_file.write(json.dumps(sample_metrics_file))
file_sample_metrics_file.close()
