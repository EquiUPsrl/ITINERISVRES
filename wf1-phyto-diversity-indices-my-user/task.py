from scipy.stats import entropy
from math import exp
import os
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from math import log

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

def _safe_sum(x: np.ndarray) -> float:
    """
    Safe sum: returns NaN if the total abundance is <= 0.
    """
    s = np.sum(x)
    return s if s > 0 else np.nan



def R(x: np.ndarray) -> int:
    """Taxonomic richness: number of taxa with positive abundance."""
    return int((x > 0).sum())



def Shannon_H(x: np.ndarray) -> float:
    """Shannon diversity index (natural log)."""
    s = _safe_sum(x)
    if np.isnan(s):
        return np.nan
    return entropy(x, base=np.e)

def Shannon_H_Eq(x: np.ndarray) -> float:
    """Exponentiated Shannon (effective number of species)."""
    H = Shannon_H(x)
    if np.isnan(H):
        return np.nan
    return exp(H)

def Simpson_D(x: np.ndarray) -> float:
    """Simpson diversity (1 - sum p_i^2)."""
    s = _safe_sum(x)
    if np.isnan(s):
        return np.nan
    p = x / s
    return 1 - np.sum(p ** 2)

def Simpson_D_Eq(x: np.ndarray) -> float:
    """Simpson effective number of species (1 / (1 - sum p_i^2))."""
    s = _safe_sum(x)
    if np.isnan(s):
        return np.nan
    p = x / s
    denom = 1 - np.sum(p ** 2)
    if denom <= 0:
        return np.nan
    return 1.0 / denom

def Menhinick_D(x: np.ndarray) -> float:
    """Menhinick's richness index."""
    s = _safe_sum(x)
    if np.isnan(s):
        return np.nan
    return (x > 0).sum() / sqrt(s)

def Margalef_D(x: np.ndarray) -> float:
    """Margalef richness index."""
    s = _safe_sum(x)
    if np.isnan(s) or s <= 0:
        return np.nan
    S = (x > 0).sum()
    if S <= 1:
        return np.nan
    return (S - 1) / log(s)

def Gleason_D(x: np.ndarray) -> float:
    """Gleason richness index."""
    s = _safe_sum(x)
    if np.isnan(s) or s <= 0:
        return np.nan
    S = (x > 0).sum()
    return S / log(s)

def McInthosh_M(x: np.ndarray) -> float:
    """McIntosh diversity index."""
    s = _safe_sum(x)
    if np.isnan(s):
        return np.nan
    return (s + sqrt(np.sum(x ** 2))) / (s + sqrt(s))

def Hurlbert_PIE(x: np.ndarray) -> float:
    """Hurlbert's PIE (probability of interspecific encounter)."""
    s = _safe_sum(x)
    if np.isnan(s):
        return np.nan
    p = x / s
    n = (x > 0).sum()
    if n <= 1:
        return np.nan
    return (n / (n - 1.0)) * (1 - np.sum(p ** 2))



def Pielou_J(x: np.ndarray) -> float:
    """Pielou evenness."""
    S = (x > 0).sum()
    if S <= 1:
        return np.nan
    H = Shannon_H(x)
    if np.isnan(H):
        return np.nan
    return H / log(S)

def Sheldon_J(x: np.ndarray) -> float:
    """Sheldon evenness (H_eq / S)."""
    S = (x > 0).sum()
    if S == 0:
        return np.nan
    H_eq = Shannon_H_Eq(x)
    if np.isnan(H_eq):
        return np.nan
    return H_eq / S

def LudwReyn_J(x: np.ndarray) -> float:
    """Ludwig & Reynolds evenness."""
    S = (x > 0).sum()
    if S <= 1:
        return np.nan
    H_eq = Shannon_H_Eq(x)
    if np.isnan(H_eq):
        return np.nan
    return (H_eq - 1.0) / (S - 1.0)



def BergerParker_B(x: np.ndarray) -> float:
    """Berger–Parker dominance index (max p_i)."""
    s = _safe_sum(x)
    if np.isnan(s) or s == 0:
        return np.nan
    return np.max(x) / s

def McNaughton_Alpha(x: np.ndarray) -> float:
    """McNaughton dominance index based on two most abundant species."""
    x = np.asarray(x)
    s = _safe_sum(x)
    if np.isnan(s) or s == 0 or len(x[x > 0]) < 2:
        return np.nan
    max1 = np.max(x)
    max1_idx = np.argmax(x)
    x_copy = np.delete(x, max1_idx)
    max2 = np.max(x_copy)
    return (max1 + max2) / s

def Hulburt(x: np.ndarray) -> float:
    """Hulburt dominance index as percentage."""
    val = McNaughton_Alpha(x)
    if np.isnan(val):
        return np.nan
    return val * 100.0



output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

traits_file = filtered_file

print("Working directory for outputs:", output_dir)


cluster = ["treatment", "season", "parenteventid"]

tax_col = "scientificname_accepted"


action_name = "diversity_indices"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]

    p = act.set_index("parameter")["value"]
    
    tax_col = p.get("tax_col", tax_col)
    if "cluster" in p:
        cluster = [x for x in p["cluster"].split(",")]

index_names = [
    "R",
    "Shannon_H",
    "Shannon_H_Eq",
    "Simpson_D",
    "Simpson_D_Eq",
    "Menhinick_D",
    "Margalef_D",
    "Gleason_D",
    "McInthosh_M",
    "Hurlbert_PIE",
    "Pielou_J",
    "Sheldon_J",
    "LudwReyn_J",
    "BergerParker_B",
    "McNaughton_Alpha",
    "Hulburt",
]

index_funcs = [
    R,
    Shannon_H,
    Shannon_H_Eq,
    Simpson_D,
    Simpson_D_Eq,
    Menhinick_D,
    Margalef_D,
    Gleason_D,
    McInthosh_M,
    Hurlbert_PIE,
    Pielou_J,
    Sheldon_J,
    LudwReyn_J,
    BergerParker_B,
    McNaughton_Alpha,
    Hulburt,
]


dataset = pd.read_csv(traits_file, sep=None, engine="python", decimal=".")

dataset.columns = (
    dataset.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.lower()
)

tax_col = tax_col.lower()
cluster = [c.lower() for c in cluster]

if "density" not in dataset.columns:
    dataset["density"] = 1.0


if cluster[0].upper() != "WHOLE":
    if len(cluster) > 1:
        ID = dataset[cluster].astype(str).agg(".".join, axis=1)
        info = (
            dataset[cluster]
            .dropna()
            .drop_duplicates()
            .astype(str)
        )
        info.index = info.astype(str).agg(".".join, axis=1)
    else:
        ID = dataset[cluster[0]].astype(str)
        info = (
            dataset[[cluster[0]]]
            .drop_duplicates()
            .astype(str)
            .set_index(cluster[0])
        )
else:
    ID = pd.Series(["all"] * len(dataset))
    info = None


den_matz = (
    dataset
    .groupby([ID, dataset[tax_col]])["density"]
    .sum()
    .unstack(fill_value=0.0)
)


index_table = pd.DataFrame(index=den_matz.index)

for name, func in zip(index_names, index_funcs):
    index_table[name] = den_matz.apply(func, axis=1)

index_table = index_table.round(3)


if cluster[0].upper() == "WHOLE" or info is None:
    final = index_table.reset_index().rename(columns={"index": "site_id"})
else:
    meta = info.loc[index_table.index].reset_index().rename(columns={"index": "site_id"})
    final = pd.concat([meta, index_table.reset_index(drop=True)], axis=1)


indices_file = os.path.join(output_dir, "DiversityIndices_Output.csv")
final.to_csv(indices_file, sep=";", decimal=".", index=False)
print("Diversity indices written to:", indices_file)


warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

plots_dir = os.path.join(output_dir, "Diversity_Plots")
os.makedirs(plots_dir, exist_ok=True)

df = pd.read_csv(indices_file, sep=";", decimal=".")
df.columns = df.columns.str.lower().str.strip()

if "treatment" in df.columns:
    group_treat = "treatment"
else:
    raise KeyError("Treatment column not found in DiversityIndices_Output.csv")

if "season" in df.columns:
    group_season = "season"
else:
    raise KeyError("Season column not found in DiversityIndices_Output.csv")

index_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric indices found:", index_cols)

sns.set_theme(style="white")
plt.rcParams["axes.grid"] = False


for idx in index_cols:
    if df[idx].isna().all():
        continue

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=df,
        x=group_treat,
        y=idx,
        hue=group_treat,
        dodge=False,
        palette="Set2",
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x=group_treat,
        y=idx,
        color="black",
        size=4,
        alpha=0.7,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax)
    ax.set_title(f"{idx} — Treatment", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{idx}_boxplot_treatment.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=df,
        x=group_season,
        y=idx,
        hue=group_season,
        dodge=False,
        palette="Set3",
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x=group_season,
        y=idx,
        color="black",
        size=4,
        alpha=0.7,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax)
    ax.set_title(f"{idx} — Season", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{idx}_boxplot_season.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(
        data=df,
        x=group_treat,
        y=idx,
        hue=group_treat,
        dodge=False,
        inner="box",
        palette="Set2",
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax)
    ax.set_title(f"{idx} — Violin (Treatment)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{idx}_violin_treatment.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=df,
        x=group_treat,
        y=idx,
        hue=group_treat,
        dodge=False,
        palette="Set2",
        errorbar="sd",
        capsize=0.2,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax)
    ax.set_title(f"{idx} — Mean ± SD (Treatment)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{idx}_bar_treatment.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=df,
        x=group_season,
        y=idx,
        hue=group_season,
        dodge=False,
        palette="Set3",
        errorbar="sd",
        capsize=0.2,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax)
    ax.set_title(f"{idx} — Mean ± SD (Season)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{idx}_bar_season.png"), dpi=300)
    plt.close(fig)

print("\nAll individual diversity plots saved to:", plots_dir)


panel_cols = ["r", "shannon_h", "pielou_j"]
panel_cols = [c for c in panel_cols if c in df.columns]

name_map = {
    "r": "Richness",
    "shannon_h": "Shannon",
    "pielou_j": "Pielou",
}

print("Panel columns used:", panel_cols)

if panel_cols:

    fig, axes = plt.subplots(2, len(panel_cols), figsize=(5 * len(panel_cols), 8))
    axes = axes.reshape(2, len(panel_cols))

    for j, col in enumerate(panel_cols):
        ax = axes[0, j]
        sns.boxplot(
            data=df, x=group_treat, y=col,
            hue=group_treat,
            dodge=False,
            palette="Set2",
            ax=ax,
        )
        sns.stripplot(
            data=df, x=group_treat, y=col,
            color="black", size=3, alpha=0.7, ax=ax,
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
            data=df, x=group_season, y=col,
            hue=group_season,
            dodge=False,
            palette="Set3",
            ax=ax,
        )
        sns.stripplot(
            data=df, x=group_season, y=col,
            color="black", size=3, alpha=0.7, ax=ax,
        )
        if ax.get_legend():
            ax.get_legend().remove()
        sns.despine(ax=ax)
        ax.set_title(f"{name_map[col]} — Season", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel(name_map[col])

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "panel_boxplots_treat_season.png"), dpi=300)
    plt.show()

    fig, axes = plt.subplots(2, len(panel_cols), figsize=(5 * len(panel_cols), 8))
    axes = axes.reshape(2, len(panel_cols))

    for j, col in enumerate(panel_cols):
        ax = axes[0, j]
        sns.violinplot(
            data=df, x=group_treat, y=col,
            hue=group_treat,
            dodge=False,
            inner="box",
            palette="Set2",
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
        sns.violinplot(
            data=df, x=group_season, y=col,
            hue=group_season,
            dodge=False,
            inner="box",
            palette="Set3",
            ax=ax,
        )
        if ax.get_legend():
            ax.get_legend().remove()
        sns.despine(ax=ax)
        ax.set_title(f"{name_map[col]} — Season", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel(name_map[col])

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "panel_violins_treat_season.png"), dpi=300)
    plt.show()

print("Panel plots saved to:", plots_dir)

file_indices_file = open("/tmp/indices_file_" + id + ".json", "w")
file_indices_file.write(json.dumps(indices_file))
file_indices_file.close()
