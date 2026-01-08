import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--biotic_file', action='store', type=str, required=True, dest='biotic_file')

arg_parser.add_argument('--input_traits', action='store', type=str, required=True, dest='input_traits')

arg_parser.add_argument('--locations_config', action='store', type=str, required=True, dest='locations_config')


args = arg_parser.parse_args()
print(args)

id = args.id

biotic_file = args.biotic_file.replace('"','')
input_traits = args.input_traits.replace('"','')
locations_config = args.locations_config.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF2/' + 'output'

base_output_dir = conf_output_path
os.makedirs(base_output_dir, exist_ok=True)

out_dir = os.path.join(base_output_dir, "Functional approach")
os.makedirs(out_dir, exist_ok=True)


def clean_name(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = x.replace("  ", " ")
    x = x.replace("�", "")
    x = x.replace("Ã", "")
    return x

def gower_distance_matrix(T):
    T = np.asarray(T, dtype=float)
    n = T.shape[0]
    D = np.zeros((n, n), dtype=float)
    rng = np.ptp(T, axis=0)
    rng[rng == 0] = 1.0
    for i in range(n):
        diff = np.abs(T[i] - T) / rng
        D[i] = diff.mean(axis=1)
    return D

def raoQ(p, D):
    p = np.asarray(p, dtype=float)
    D = np.asarray(D, dtype=float)
    return float((p[:, None] * p[None, :] * D).sum())

def fdis(p, T):
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)
    c = np.average(T, axis=0, weights=p)
    d = np.sqrt(((T - c) ** 2).sum(axis=1))
    return float((p * d).sum())

def min_max_scale(x):
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return np.zeros_like(x)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    if x_max == x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)



df = pd.read_csv(locations_config, sep=";")

stations_keep = dict(zip(df["locationID"], df["locality"]))

phyto = pd.read_csv(
    biotic_file,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

site_order = sorted(phyto["locality"].unique())

phyto_sub = phyto[phyto["locationID"].isin(stations_keep.keys())].copy()
phyto_sub["lake"] = phyto_sub["locationID"].map(stations_keep)

phyto_sub["acceptedNameUsage"] = phyto_sub["acceptedNameUsage"].apply(clean_name)
phyto_sub["year"] = pd.to_numeric(phyto_sub["year"], errors="coerce")
phyto_sub["density"] = pd.to_numeric(phyto_sub["density"], errors="coerce")
phyto_sub = phyto_sub.dropna(subset=["year", "density", "acceptedNameUsage", "lake"])

print("Lakes & years:\n", phyto_sub.groupby("lake")["year"].agg(["min", "max", "nunique"]))

comm_long_all = (
    phyto_sub
    .groupby(["lake", "year", "acceptedNameUsage"], as_index=False)["density"]
    .sum()
)

print("\nCommunity LONG – first rows:")
print(comm_long_all.head())

species_all = comm_long_all["acceptedNameUsage"].unique()

traits = pd.read_csv(
    input_traits,   
    sep=";",
    encoding="utf-8",
    low_memory=False
)

traits["acceptedNameUsage"] = traits["acceptedNameUsage"].apply(clean_name)
traits_sub = traits[traits["acceptedNameUsage"].isin(species_all)].copy()

print("\nN species in all lakes:", len(species_all))
print("N species with traits:", traits_sub["acceptedNameUsage"].nunique())

binary_cols = []
for col in traits_sub.columns:
    if col == "acceptedNameUsage":
        continue
    vals = traits_sub[col].dropna().unique()
    if len(vals) > 0 and set(vals).issubset({0, 1}):
        binary_cols.append(col)

desc = traits_sub[binary_cols].describe().T
min_count = 50

informative = desc[
    (desc["count"] >= min_count) &
    (desc["mean"] > 0.05) &
    (desc["mean"] < 0.95)
].index.tolist()

if len(informative) == 0:
    informative = desc[
        (desc["count"] >= min_count) &
        (desc["mean"] > 0.01) &
        (desc["mean"] < 0.99)
    ].index.tolist()

print("\nBinary informative traits (all lakes):")
print(informative)

traits_to_remove = ["marine", "brackish", "terrestrial"]
informative = [t for t in informative if t not in traits_to_remove]

print("\nBinary informative traits (filtered):")
print(informative)

traits_mat = (
    traits_sub[["acceptedNameUsage"] + informative]
    .drop_duplicates(subset=["acceptedNameUsage"])
    .set_index("acceptedNameUsage")
    .astype(float)
)

print("\nTrait matrix (species × traits) shape:", traits_mat.shape)

results = []

for lake, df_lake in comm_long_all.groupby("lake"):
    comm_matrix = df_lake.pivot_table(
        index="year",
        columns="acceptedNameUsage",
        values="density",
        fill_value=0
    )

    for year, row in comm_matrix.iterrows():
        abund = row.values
        spp = row.index.values
        total = abund.sum()
        S = (abund > 0).sum()

        if total <= 0:
            results.append({"lake": lake, "year": year,
                            "RaoQ": np.nan, "FDis": np.nan, "S": S})
            continue

        df_year = pd.DataFrame({
            "acceptedNameUsage": spp,
            "density": abund
        })

        merged = df_year.merge(
            traits_mat.reset_index(),
            on="acceptedNameUsage",
            how="inner"
        )

        if merged.shape[0] < 2:
            results.append({"lake": lake, "year": year,
                            "RaoQ": np.nan, "FDis": np.nan, "S": S})
            continue

        merged["density"] = merged["density"] / merged["density"].sum()
        p = merged["density"].values
        T = merged[informative].values

        D = gower_distance_matrix(T)
        rao_val = raoQ(p, D)
        fdis_val = fdis(p, T)

        results.append({
            "lake": lake,
            "year": year,
            "RaoQ": rao_val,
            "FDis": fdis_val,
            "S": S
        })

fd_panel = pd.DataFrame(results).sort_values(["lake", "year"])
print("\nFD panel – first rows:")
print(fd_panel.head())


fd_panel["FunctionalRedundancy"] = np.nan

for lake in fd_panel["lake"].unique():
    mask = fd_panel["lake"] == lake
    S_scaled = min_max_scale(fd_panel.loc[mask, "S"].values)
    Rao_scaled = min_max_scale(fd_panel.loc[mask, "RaoQ"].values)
    fd_panel.loc[mask, "FunctionalRedundancy"] = S_scaled - Rao_scaled

fd_panel_path = os.path.join(out_dir, "FD_panel_all_lakes.csv")
fd_panel.to_csv(fd_panel_path, sep = ";", index=False)
print("\nFD panel saved to:", fd_panel_path)



def plot_fd_panel(fd_df, metric, ylabel, filename, site_order):
    lakes_order = site_order
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, lake in enumerate(lakes_order):
        ax = axes[i]
        sub = fd_df[fd_df["lake"] == lake].dropna(subset=[metric])

        if sub.empty:
            ax.set_visible(False)
            continue

        x = sub["year"].values
        y = sub[metric].values

        ax.scatter(x, y, color="black")

        if len(sub) >= 3:
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            slope = model.params[1]
            pval = model.pvalues[1]

            xp = np.linspace(x.min(), x.max(), 100)
            Xp = sm.add_constant(xp)
            pred = model.get_prediction(Xp).summary_frame()

            ax.plot(xp, pred["mean"], color="tab:blue")
            ax.fill_between(
                xp,
                pred["mean_ci_lower"],
                pred["mean_ci_upper"],
                color="tab:blue", alpha=0.2
            )

            ax.text(
                0.05, 0.95,
                f"slope = {slope:.02f}\np = {pval:.3f}",
                transform=ax.transAxes,
                ha="left", va="top", fontsize=8
            )

        ax.set_title(lake, fontsize=10)

    for j in range(len(lakes_order), len(axes)):
        axes[j].set_visible(False)

    for ax in axes[-ncols:]:
        ax.set_xlabel("Year")

    fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")
    plt.tight_layout(rect=[0.06, 0.05, 1, 1])

    plt.savefig(os.path.join(out_dir, filename + ".png"), dpi=300)
    plt.savefig(os.path.join(out_dir, filename + ".svg"))
    plt.show()



plot_fd_panel(fd_panel, "RaoQ",
              ylabel="Functional diversity (RaoQ)",
              filename="FD_RaoQ_panel_all_lakes",
             site_order=site_order)

plot_fd_panel(fd_panel, "FDis",
              ylabel="Functional dispersion (FDis)",
              filename="FD_FDis_panel_all_lakes",
             site_order=site_order)

plot_fd_panel(fd_panel, "FunctionalRedundancy",
              ylabel="Functional redundancy (scaled S - scaled RaoQ)",
              filename="FD_Redundancy_panel_all_lakes",
             site_order=site_order)



metrics = ["RaoQ", "FDis", "FunctionalRedundancy"]
stats_rows = []

for metric in metrics:
    for lake in fd_panel["lake"].unique():
        sub = fd_panel[(fd_panel["lake"] == lake) & (~fd_panel[metric].isna())]

        if len(sub) < 3:
            stats_rows.append({
                "lake": lake,
                "metric": metric,
                "slope": np.nan,
                "intercept": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
                "n": len(sub)
            })
            continue

        x = sub["year"].values
        y = sub[metric].values

        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        slope = model.params[1]
        intercept = model.params[0]
        pval = model.pvalues[1]
        r2 = model.rsquared

        stats_rows.append({
            "lake": lake,
            "metric": metric,
            "slope": slope,
            "intercept": intercept,
            "p_value": pval,
            "r_squared": r2,
            "n": len(sub)
        })

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(out_dir, "FD_trend_statistics_all_lakes.csv"), sep = ";", index=False)

print("\nSaved trend stats to: FD_trend_statistics_all_lakes.csv")
display(stats_df)

