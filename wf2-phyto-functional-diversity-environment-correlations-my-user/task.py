import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--abiotic_file', action='store', type=str, required=True, dest='abiotic_file')

arg_parser.add_argument('--fd_panel_path', action='store', type=str, required=True, dest='fd_panel_path')

arg_parser.add_argument('--locations_config', action='store', type=str, required=True, dest='locations_config')


args = arg_parser.parse_args()
print(args)

id = args.id

abiotic_file = args.abiotic_file.replace('"','')
fd_panel_path = args.fd_panel_path.replace('"','')
locations_config = args.locations_config.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF2/' + 'output'

output_dir = conf_output_path
func_output_dir = os.path.join(output_dir, "Functional approach")
env_output_dir = os.path.join(func_output_dir, "abiotic_correlations")

os.makedirs(func_output_dir, exist_ok=True)
os.makedirs(env_output_dir, exist_ok=True)

env = pd.read_csv(
    abiotic_file,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

env["locality"]   = env["locality"].astype(str).str.strip()
env["locationID"] = env["locationID"].astype(str).str.strip()
env["year"]       = pd.to_numeric(env["year"], errors="coerce")


df = pd.read_csv(locations_config, sep=";")

stations_keep = dict(zip(df["locationID"], df["locality"]))

env_sub = env[env["locationID"].isin(stations_keep.keys())].copy()
env_sub["lake"] = env_sub["locationID"].map(stations_keep)

env_numeric = env_sub.select_dtypes(include=["number"]).copy()
env_numeric = env_numeric.drop(columns=["year"], errors="ignore")

env_annual = (
    env_sub.groupby(["lake", "year"])[env_numeric.columns]
    .mean()
    .reset_index()
)

print("\nAnnual environment – first rows:")
print(env_annual.head())


fd_panel = pd.read_csv(fd_panel_path, sep = ";")

fd_env = fd_panel.merge(env_annual, on=["lake", "year"], how="left")

fd_env_file = os.path.join(env_output_dir, "FD_ENV_all_lakes.csv")
fd_env.to_csv(fd_env_file, sep = ";", index=False)

print("\nSaved FD_ENV_all_lakes.csv to:")
print(fd_env_file)
print(fd_env.head())


fd_env = pd.read_csv(fd_env_file, sep = ";")

exclude = {"RaoQ", "FDis", "FunctionalRedundancy", "S", "year"}
env_vars = [c for c in fd_env.columns if c not in exclude and fd_env[c].dtype != 'O']

rows = []

for lake in fd_env["lake"].unique():
    df_lake = fd_env[fd_env["lake"] == lake]

    for metric in ["RaoQ", "FDis", "FunctionalRedundancy"]:
        for var in env_vars:

            x = df_lake[var].values
            y = df_lake[metric].values
            mask = ~np.isnan(x) & ~np.isnan(y)

            if mask.sum() >= 3:
                r, p = pearsonr(x[mask], y[mask])
                rows.append({
                    "lake": lake,
                    "metric": metric,
                    "env_var": var,
                    "r": r,
                    "p": p,
                    "n": int(mask.sum())
                })

corr_df = pd.DataFrame(rows)
corr_df = corr_df.sort_values(["metric", "lake", "p"])

corr_file = os.path.join(env_output_dir, "FD_ENV_correlations_all_lakes.csv")
corr_df.to_csv(corr_file, sep = ";", index=False)

print("\nSaved correlations to:")
print(corr_file)
print(corr_df.head(20))


corr = pd.read_csv(corr_file, sep = ";")

lakes = corr["lake"].unique()

for lake_name in lakes:
    lake_data = corr[corr["lake"] == lake_name].copy()
    
    print(f"Data processing for the lake: {lake_name}")

    env_order = [
        "waterTemperature",
        "transparency",
        "totalPhosphorous",
        "totalNitrogen",
        "orthophosphate",
        "nitrite",
        "nitrate",
        "reactiveSilica",
        "ph",
        "alcalinity",
        "ammonium",
        "dissolvedOxygen",
    ]
    
    metrics_order = ["RaoQ", "FDis", "FunctionalRedundancy"]
    
    lake_data = lake_data[lake_data["env_var"].isin(env_order) & lake_data["metric"].isin(metrics_order)].copy()
    
    lake_data["metric"]  = pd.Categorical(lake_data["metric"],
                                    categories=metrics_order,
                                    ordered=True)
    lake_data["env_var"] = pd.Categorical(lake_data["env_var"],
                                    categories=env_order[::-1],  # from the first at the top to the last at the bottom
                                    ordered=True)
    
    lake_data["x"] = lake_data["metric"].cat.codes
    lake_data["y"] = lake_data["env_var"].cat.codes
    
    r = lake_data["r"].values
    p = lake_data["p"].values
    
    size_min = 80
    size_max = 800
    sizes = size_min + (size_max - size_min) * np.abs(r)
    
    colors = r
    
    fig, ax = plt.subplots(figsize=(4.5, 6.0))
    
    sc = ax.scatter(
        lake_data["x"], lake_data["y"],
        s=sizes,
        c=colors,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        edgecolors="black",
        linewidths=0.4
    )
    
    for xi, yi, pi in zip(lake_data["x"], lake_data["y"], p):
        if pi < 0.05:
            ax.text(
                xi, yi,
                "*",
                ha="center", va="center",
                fontsize=10,
                fontweight="bold",
                color="white"
            )
    
    ax.set_xticks(range(len(metrics_order)))
    ax.set_xticklabels(metrics_order, rotation=0)
    ax.set_yticks(range(len(env_order)))
    ax.set_yticklabels(env_order[::-1])   # same order of categories
    
    ax.set_xlim(-0.5, len(metrics_order) - 0.5)
    ax.set_ylim(-0.5, len(env_order) - 0.5)
    
    ax.set_xlabel("Functional metrics")
    ax.set_ylabel("Environmental variables")
    
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.subplots_adjust(right=0.82)
    
    cbar_ax = fig.add_axes([0.84, 0.18, 0.03, 0.64])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Pearson r")
    
    out_fig = os.path.join(env_output_dir,
                           lake_name + "_FD_ENV_correlations_bubbles_vertical_clean_ext.tiff")
    fig.savefig(out_fig, dpi=600, bbox_inches="tight")
    plt.show()
    
    print("Figure saved in:", out_fig)

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
