import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--biotic_file', action='store', type=str, required=True, dest='biotic_file')


args = arg_parser.parse_args()
print(args)

id = args.id

biotic_file = args.biotic_file.replace('"','')


conf_output_path = conf_output_path = '' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

reg_output_dir = os.path.join(output_dir, "Regressions")
os.makedirs(reg_output_dir, exist_ok=True)

abundance_col = "density"   # <-- density column

phyto = pd.read_csv(
    biotic_file,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

phyto["locality"]   = phyto["locality"].astype(str).str.strip()
phyto["locationID"] = phyto["locationID"].astype(str).str.strip()
phyto["acceptedNameUsage"] = phyto["acceptedNameUsage"].astype(str).str.strip()

phyto["year"] = pd.to_numeric(phyto["year"], errors="coerce")
phyto[abundance_col] = pd.to_numeric(phyto[abundance_col], errors="coerce")

phyto = phyto.dropna(subset=["year", abundance_col])

richness_year = (
    phyto
    .groupby(["locality", "year"])["acceptedNameUsage"]
    .nunique()
    .reset_index(name="Taxa_richness")
)


grouped_taxa = (
    phyto
    .groupby(["locality", "year", "acceptedNameUsage"], as_index=False)[abundance_col]
    .sum()
)

def shannon_index(group):
    counts = group.values
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))

shannon_year = (
    grouped_taxa
    .groupby(["locality", "year"], group_keys=False)[abundance_col]
    .apply(shannon_index)
    .reset_index(name="Shannon")
)


def lake_regression(df, y_col):
    """
    Linear regression y ~ year for each lake.
    """
    rows = []
    for lake, g in df.groupby("locality"):
        if len(g) < 3:
            continue

        X = sm.add_constant(g["year"])   # columns: 'const' and 'year'
        y = g[y_col]

        model = sm.OLS(y, X).fit()

        rows.append({
            "lake": lake,
            "slope": model.params["year"],
            "pvalue": model.pvalues["year"],
            "R2": model.rsquared
        })
    return pd.DataFrame(rows)

reg_richness = lake_regression(richness_year, "Taxa_richness")
reg_shannon  = lake_regression(shannon_year, "Shannon")

stats_rich_dict = reg_richness.set_index("lake").to_dict(orient="index")
stats_shan_dict = reg_shannon.set_index("lake").to_dict(orient="index")

print("\n=== Trend Richness ===")
print(reg_richness)
print("\n=== Trend Shannon ===")
print(reg_shannon)

reg_richness.to_csv(os.path.join(reg_output_dir, "regressions_richness_per_lake.csv"), sep = ";", index=False)
reg_shannon.to_csv(os.path.join(reg_output_dir, "regressions_shannon_per_lake.csv"), sep = ";", index=False)


def plot_panels_with_regression_clean(df, y_col, y_label, filename, stats_dict, out_dir):
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    lakes = sorted(df["locality"].unique())
    n_lakes = len(lakes)

    ncols = 3
    nrows = int(np.ceil(n_lakes / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7, 2.6 * nrows),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    span = year_max - year_min

    if span <= 10:
        step = 2
    elif span <= 16:
        step = 3
    else:
        step = 5
    xticks = np.arange(year_min, year_max + 1, step)

    y_max = df[y_col].max()
    y_top = y_max * 1.12

    for i, lake in enumerate(lakes):
        ax = axes[i]

        ax.set_facecolor("white")
        ax.patch.set_alpha(0)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.grid(False)

        sub = df[df["locality"] == lake].sort_values("year")
        x = sub["year"].values
        y = sub[y_col].values

        ax.scatter(x, y, s=22, color="black")

        if len(sub) >= 3:
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()

            x_pred = np.linspace(year_min, year_max, 200)
            X_pred = sm.add_constant(x_pred)
            pred = model.get_prediction(X_pred)
            y_pred = pred.predicted_mean
            ci_low, ci_high = pred.conf_int().T

            ax.plot(x_pred, y_pred, color="#1f77b4", linewidth=1.4)
            ax.fill_between(
                x_pred, ci_low, ci_high,
                alpha=0.22, color="#1f77b4"
            )

        ax.set_title(lake, pad=4)

        ax.set_xlim(year_min - 0.5, year_max + 0.5)
        ax.set_ylim(0, y_top)

        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(t)) for t in xticks])

        if (i // ncols) != (nrows - 1):
            ax.tick_params(labelbottom=False)

        st = stats_dict.get(lake)
        if st is not None:
            ax.text(
                0.02, 0.95,
                f"slope = {st['slope']:.2f}\np = {st['pvalue']:.3f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=7
            )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.text(0.5, 0.02, "Year", ha="center", fontsize=10)
    fig.text(0.02, 0.5, y_label, va="center", rotation="vertical", fontsize=10)

    fig.tight_layout(rect=(0.06, 0.05, 1, 1))
    fig.savefig(os.path.join(out_dir, filename), dpi=600, bbox_inches="tight")
    plt.show()


plot_panels_with_regression_clean(
    richness_year,
    "Taxa_richness",
    "Taxa richness",
    "Taxa_richness_panels_clean.tiff",
    stats_rich_dict,
    reg_output_dir
)

plot_panels_with_regression_clean(
    shannon_year,
    "Shannon",
    "Shannon diversity index (H')",
    "Shannon_panels_clean.tiff",
    stats_shan_dict,
    reg_output_dir
)

