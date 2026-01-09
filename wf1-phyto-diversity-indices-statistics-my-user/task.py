import pandas as pd
import warnings
import os
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--indices_file', action='store', type=str, required=True, dest='indices_file')


args = arg_parser.parse_args()
print(args)

id = args.id

indices_file = args.indices_file.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF1/' + 'output'

df_stats = pd.read_csv(indices_file, sep=";", decimal=".")
df_stats.columns = df_stats.columns.str.lower().str.strip()

if "treatment" not in df_stats.columns or "season" not in df_stats.columns:
    raise KeyError("Columns 'treatment' and/or 'season' not found in DiversityIndices_Output.csv")

df_stats["treatment"] = df_stats["treatment"].astype("category")
df_stats["season"] = df_stats["season"].astype("category")

indices = ["r", "shannon_h", "pielou_j"]

print("Indices to analyse:", indices)
print("\n============================")
print(" TWO-WAY ANOVA + TUKEY HSD ")
print("============================\n")

for idx in indices:
    if idx not in df_stats.columns:
        print(f"!! Skipping {idx}: column not found in dataframe.")
        continue

    print("\n" + "#" * 80)
    print(f"INDEX: {idx}")
    print("#" * 80 + "\n")

    data = df_stats[["treatment", "season", idx]].dropna().copy()

    print("Number of observations per treatment × season:")
    print(data.groupby(["treatment", "season"]).size(), "\n")

    formula = f"{idx} ~ C(treatment) * C(season)"
    model = smf.ols(formula, data=data).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)
    print("Two-way ANOVA (treatment, season, interaction):")
    print(anova_table, "\n")

    print("Assumption checks:")
    if len(model.resid) >= 3:  # Shapiro requires at least 3 obs
        w_shapiro, p_shapiro = stats.shapiro(model.resid)
        print(f"  Shapiro–Wilk (residuals): W = {w_shapiro:.3f}, p = {p_shapiro:.4f}")
    else:
        print("  Shapiro–Wilk (residuals): not computed (too few observations).")

    groups_treat = [g[idx].values for _, g in data.groupby("treatment")]
    if all(len(g) > 1 for g in groups_treat):
        stat_lev_treat, p_lev_treat = stats.levene(*groups_treat)
        print(f"  Levene by treatment:     W = {stat_lev_treat:.3f}, p = {p_lev_treat:.4f}")
    else:
        print("  Levene by treatment: not computed (some groups have < 2 obs).")

    groups_season = [g[idx].values for _, g in data.groupby("season")]
    if all(len(g) > 1 for g in groups_season):
        stat_lev_season, p_lev_season = stats.levene(*groups_season)
        print(f"  Levene by season:        W = {stat_lev_season:.3f}, p = {p_lev_season:.4f}")
    else:
        print("  Levene by season: not computed (some groups have < 2 obs).")

    print("\nTukey HSD post-hoc tests:")

    try:
        tukey_treat = pairwise_tukeyhsd(
            endog=data[idx],
            groups=data["treatment"],
            alpha=0.05
        )
        print("\n  Tukey by TREATMENT:")
        print(tukey_treat.summary())
    except Exception as e:
        print("\n  Tukey by TREATMENT: could not be computed.")
        print("  Reason:", e)

    try:
        tukey_season = pairwise_tukeyhsd(
            endog=data[idx],
            groups=data["season"],
            alpha=0.05
        )
        print("\n  Tukey by SEASON:")
        print(tukey_season.summary())
    except Exception as e:
        print("\n  Tukey by SEASON: could not be computed.")
        print("  Reason:", e)

    print("\n" + "-" * 80 + "\n")

print("Done.")








warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


output_dir = conf_output_path
plots_dir = os.path.join(output_dir, "Diversity_Plots")
os.makedirs(plots_dir, exist_ok=True)

df = pd.read_csv(indices_file, sep=";", decimal=".")
df.columns = df.columns.str.lower().str.strip()

df["treatment"] = df["treatment"].astype("category")
df["season"] = df["season"].astype("category")

group_treat = "treatment"
group_season = "season"

panel_cols = ["r", "shannon_h", "pielou_j"]
panel_cols = [c for c in panel_cols if c in df.columns]

name_map = {
    "r": "Richness",
    "shannon_h": "Shannon diversity",
    "pielou_j": "Pielou evenness"
}

print("Panel columns used:", panel_cols)

sns.set_theme(style="white")
plt.rcParams["axes.grid"] = False

treat_order = list(df[group_treat].cat.categories)
season_order = ["Autumn", "Spring", "Summer", "Winter"]
season_order = [s for s in season_order if s in df[group_season].unique()]

letters_richness_season = {
    "Autumn": "ab",
    "Spring": "b",
    "Summer": "ab",
    "Winter": "a"
}


if panel_cols:
    fig, axes = plt.subplots(2, len(panel_cols), figsize=(5 * len(panel_cols), 8))
    axes = axes.reshape(2, len(panel_cols))

    for j, col in enumerate(panel_cols):
        ax = axes[0, j]
        sns.violinplot(
            data=df,
            x=group_treat,
            y=col,
            hue=group_treat,
            order=treat_order,
            dodge=False,
            inner="box",
            palette="Set2",
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x=group_treat,
            y=col,
            order=treat_order,
            color="black",
            size=3,
            alpha=0.6,
            ax=ax,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        sns.despine(ax=ax)
        ax.set_title(f"{name_map[col]} — Treatment", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel(name_map[col])

    for j, col in enumerate(panel_cols):
        ax = axes[1, j]
        sns.violinplot(
            data=df,
            x=group_season,
            y=col,
            hue=group_season,
            order=season_order,
            dodge=False,
            inner="box",
            palette="Set3",
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x=group_season,
            y=col,
            order=season_order,
            color="black",
            size=3,
            alpha=0.6,
            ax=ax,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        sns.despine(ax=ax)
        ax.set_title(f"{name_map[col]} — Season", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel(name_map[col])

        if col == "r":
            y_min = df[col].min()
            y_max = df[col].max()
            y_offset = 0.08 * (y_max - y_min if y_max > y_min else 1.0)

            for x_pos, season in enumerate(season_order):
                if season in letters_richness_season:
                    letter = letters_richness_season[season]
                    group_vals = df.loc[df[group_season] == season, col]
                    if not group_vals.empty:
                        y_pos = group_vals.max() + y_offset
                    else:
                        y_pos = y_max + y_offset
                    ax.text(
                        x_pos,
                        y_pos,
                        letter,
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold"
                    )

    fig.tight_layout()
    out_fig = os.path.join(plots_dir, "panel_violins_treat_season_paper.png")
    fig.savefig(out_fig, dpi=300)
    plt.show()

    print("Paper-style violin panel saved to:", out_fig)
else:
    print("No panel columns found for plotting.")







warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


output_dir = conf_output_path
plots_dir = os.path.join(output_dir, "Diversity_Plots")
os.makedirs(plots_dir, exist_ok=True)

df = pd.read_csv(indices_file, sep=";", decimal=".")
df.columns = df.columns.str.lower().str.strip()

df["treatment"] = df["treatment"].astype("category")
df["season"] = df["season"].astype("category")

group_treat = "treatment"
group_season = "season"

panel_cols = ["r", "shannon_h", "pielou_j"]
panel_cols = [c for c in panel_cols if c in df.columns]

name_map = {
    "r": "Richness",
    "shannon_h": "Shannon diversity",
    "pielou_j": "Pielou evenness",
}

letters_richness_season = {
    "Autumn": "ab",
    "Spring": "b",
    "Summer": "ab",
    "Winter": "a",
}

season_order = ["Autumn", "Spring", "Summer", "Winter"]
season_order = [s for s in season_order if s in df["season"].unique()]
treat_order = list(df["treatment"].cat.categories)

sns.set_theme(style="white")
plt.rcParams["axes.grid"] = False

fig, axes = plt.subplots(2, len(panel_cols), figsize=(5 * len(panel_cols), 8))
axes = axes.reshape(2, len(panel_cols))

for j, col in enumerate(panel_cols):
    ax = axes[0, j]

    sns.boxplot(
        data=df,
        x=group_treat,
        y=col,
        order=treat_order,
        hue=group_treat,
        dodge=False,
        palette="Set2",
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x=group_treat,
        y=col,
        order=treat_order,
        color="black",
        size=3,
        alpha=0.7,
        ax=ax,
    )

    if ax.get_legend():
        ax.get_legend().remove()

    sns.despine(ax=ax)
    ax.set_title(f"{name_map[col]} — Treatment", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel(name_map[col])

for j, col in enumerate(panel_cols):
    ax = axes[1, j]

    sns.boxplot(
        data=df,
        x=group_season,
        y=col,
        order=season_order,
        hue=group_season,
        dodge=False,
        palette="Set3",
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x=group_season,
        y=col,
        order=season_order,
        color="black",
        size=3,
        alpha=0.7,
        ax=ax,
    )

    if ax.get_legend():
        ax.get_legend().remove()

    sns.despine(ax=ax)
    ax.set_title(f"{name_map[col]} — Season", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel(name_map[col])

    if col == "r":
        y_max = df[col].max()
        y_min = df[col].min()
        height = y_max - y_min if y_max > y_min else 1.0
        y_letter = y_max + 0.15 * height  # position of all letters
        y_top = y_max + 0.25 * height     # top of y-limits so letters stay inside

        ax.set_ylim(y_min - 0.05 * height, y_top)

        for x_pos, season in enumerate(season_order):
            if season in letters_richness_season:
                letter = letters_richness_season[season]
                ax.text(
                    x_pos,
                    y_letter,
                    letter,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

fig.tight_layout()
out_fig = os.path.join(plots_dir, "panel_boxplots_treat_season_statistics.png")
fig.savefig(out_fig, dpi=300)
plt.show()

print("Paper-style BOXPLOT panel with statistics saved to:", out_fig)

