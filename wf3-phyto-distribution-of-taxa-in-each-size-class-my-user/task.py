import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--biotic_size_class', action='store', type=str, required=True, dest='biotic_size_class')

arg_parser.add_argument('--out_filtered_file_axis1', action='store', type=str, required=True, dest='out_filtered_file_axis1')

arg_parser.add_argument('--out_filtered_file_axis2', action='store', type=str, required=True, dest='out_filtered_file_axis2')


args = arg_parser.parse_args()
print(args)

id = args.id

biotic_size_class = args.biotic_size_class.replace('"','')
out_filtered_file_axis1 = args.out_filtered_file_axis1.replace('"','')
out_filtered_file_axis2 = args.out_filtered_file_axis2.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

niche_file_axis1 = out_filtered_file_axis1
niche_file_axis2 = out_filtered_file_axis2

bio_sc_file = biotic_size_class

def plot_distribution(niche_file, bio_sc_file, axis_to_use=1):

    print(f"Using Axis {axis_to_use}")
    print(f"Niche file: {niche_file}")
    print(f"Density file: {bio_sc_file}")
    
    
    niche = pd.read_csv(niche_file, encoding="utf-8-sig")
    
    for col in ["scientificName", "sizeClass", "optimum", "tolerance"]:
        if col not in niche.columns:
            raise ValueError(
                f"Missing column '{col}' in {niche_file}. Columns found: {niche.columns.tolist()}"
            )
    
    niche["sizeClass"] = niche["sizeClass"].astype(str)
    
    niche["xmin"] = niche["optimum"] - niche["tolerance"] / 2
    niche["xmax"] = niche["optimum"] + niche["tolerance"] / 2
    
    niche = niche[
        niche["xmin"].notna()
        & niche["xmax"].notna()
        & ((niche["xmax"] - niche["xmin"]) > 0)
    ].copy()
    
    print("Species with valid niche interval:", niche.shape[0])
    
    
    bio_sc = pd.read_csv(bio_sc_file, encoding="utf-8-sig", sep=",")
    
    if not {"scientificName", "density"}.issubset(bio_sc.columns):
        raise ValueError(
            f"{bio_sc_file} must contain at least 'scientificName' and 'density'. "
            f"Columns found: {bio_sc.columns.tolist()}"
        )
    
    dens_species = (
        bio_sc.groupby("scientificName", as_index=False)["density"]
        .sum()
        .rename(columns={"density": "total_density"})
    )
    
    df = niche.merge(dens_species, on="scientificName", how="left")
    
    print("Rows after merging niche + density:", df.shape[0])
    print("Rows with non-NaN total_density:", df["total_density"].notna().sum())
    
    df["log_density"] = np.log(df["total_density"] + 1)
    df = df[df["log_density"].notna()].copy()
    
    print("Species with valid niche + log(total_density):", df.shape[0])
    
    
    
    counts = df.groupby("sizeClass")["scientificName"].nunique()
    valid_classes = counts[counts >= 3].index.tolist()
    
    print("Size classes with ≥ 3 species:", valid_classes)
    
    df = df[df["sizeClass"].isin(valid_classes)].copy()
    
    desired_order = ["4", "3", "2", "1", "0", "-1"]
    size_classes = [c for c in desired_order if c in valid_classes]
    if not size_classes:
        size_classes = sorted(valid_classes, key=lambda x: float(x))
    
    
    
    n_panels = len(size_classes)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4 * n_panels, 6),
        sharey=True
    )
    
    if n_panels == 1:
        axes = [axes]
    
    colors = ["navy", "royalblue", "cornflowerblue", "skyblue"]
    color_map = {cls: colors[i % len(colors)] for i, cls in enumerate(size_classes)}
    
    global_xmin = df["xmin"].min()
    global_xmax = df["xmax"].max()
    x_margin = 0.5
    
    global_ymin = df["log_density"].min()
    global_ymax = df["log_density"].max()
    if global_ymax == global_ymin:
        global_ymax = global_ymin + 1.0
    y_range = global_ymax - global_ymin
    y_margin = 0.1 * y_range
    bar_height = 0.04 * y_range
    
    
    
    for ax, cls in zip(axes, size_classes):
        sub = df[df["sizeClass"] == cls].copy()
    
        if sub.empty:
            ax.set_title(f"Size class {cls} (no data)")
            continue
    
        sub = sub.sort_values("log_density", ascending=True)
    
        for _, row in sub.iterrows():
            xmin = row["xmin"]
            xmax = row["xmax"]
            y = row["log_density"]
    
            width = xmax - xmin
            if not np.isfinite(width) or width <= 0:
                continue
    
            rect = Rectangle(
                (xmin, y - bar_height / 2.0),
                width,
                bar_height,
                edgecolor="black",
                facecolor=color_map[cls],
                linewidth=0.8
            )
            ax.add_patch(rect)
    
            label = row["scientificName"]
            ax.text(
                xmax + 0.03,   # slightly to the right of the rectangle
                y,
                label,
                va="center",
                ha="left",
                fontsize=7
            )
    
        ax.set_title(f"Size class {cls}")
        ax.set_xlim(global_xmin - x_margin, global_xmax + x_margin)
        ax.set_ylim(global_ymin - y_margin, global_ymax + y_margin)
        ax.set_xlabel(f"Niche axis (Axis {axis_to_use})")
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    
    axes[0].set_ylabel("log(sum density)")
    
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    plt.tight_layout()

    fig_name = os.path.join(output_dir, f"taxa_distribution_by_size_class_axis{axis_to_use}.png")
    plt.savefig(fig_name, dpi=300)
    print(f"Figure saved as: {fig_name}")
    
    plt.show()


plot_distribution(niche_file_axis1, bio_sc_file, 1)

plot_distribution(niche_file_axis2, bio_sc_file, 2)

