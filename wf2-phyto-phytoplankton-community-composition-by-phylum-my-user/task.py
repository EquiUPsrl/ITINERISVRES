import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


conf_output_path = conf_output_path = '/tmp/data/WF2/' + 'output'

plt.rcParams.update({
    "font.family": "DejaVu Sans",       # full font with superscript
    "mathtext.fontset": "dejavusans",   # force mathtext to use DejaVu Sans
})

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

phyto = pd.read_csv(
    biotic_file,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

phyto["locality"]   = phyto["locality"].astype(str).str.strip()
phyto["locationID"] = phyto["locationID"].astype(str).str.strip()

phyto["density"]        = pd.to_numeric(phyto["density"],        errors="coerce")
phyto["totalBiovolume"] = pd.to_numeric(phyto["totalBiovolume"], errors="coerce")
phyto["year"]           = pd.to_numeric(phyto["year"],           errors="coerce").astype("Int64")

phyto = phyto[phyto["phylum"].notna() & (phyto["phylum"] != "")]
phyto = phyto.dropna(subset=["density", "totalBiovolume"])

agg = (
    phyto
    .groupby(["locality", "year", "phylum"], dropna=False)
    .agg(
        total_density=("density", "sum"),
        total_biovolume=("totalBiovolume", "sum")
    )
    .reset_index()
)


sums = agg.groupby(["locality", "year"])[["total_density", "total_biovolume"]].transform("sum")
mask_nonzero = (sums["total_density"] > 0) | (sums["total_biovolume"] > 0)
agg_rel = agg[mask_nonzero].copy()

agg_rel["rel_density"] = (
    agg_rel["total_density"] /
    agg_rel.groupby(["locality", "year"])["total_density"].transform("sum")
).fillna(0) * 100

agg_rel["rel_biovolume"] = (
    agg_rel["total_biovolume"] /
    agg_rel.groupby(["locality", "year"])["total_biovolume"].transform("sum")
).fillna(0) * 100

agg.to_csv(os.path.join(output_dir, "phytoplankton_composition_ABS_by_phylum_locality_year.csv"), index=False)
agg_rel.to_csv(os.path.join(output_dir, "phytoplankton_composition_REL_by_phylum_locality_year_PERCENT.csv"), index=False)


localities = sorted(agg["locality"].unique())

phyla = sorted(agg["phylum"].dropna().unique())

custom_colors = {
    "Chlorophyta":       "#33cc33",
    "Cryptista":         "#c2b280",
    "Dinoflagellata":    "#8b4513",
    "Bacillariophyta":   "#fb8072",
    "Cyanobacteria":     "#80b1d3",
    "Charophyta":        "#ffa500",
    "Euglenophyta":      "#a9a9a9",
    "Haptophyta":        "#ffb3de",
    "Heterokontophyta":  "#fffac8"
}

remaining_phyla = [p for p in phyla if p not in custom_colors]
auto_colors = plt.cm.tab20(np.linspace(0, 1, len(remaining_phyla)))
auto_color_map = dict(zip(remaining_phyla, auto_colors))
color_map = {**custom_colors, **auto_color_map}


def plot_faceted_stacked(data, value_col, ylabel, title, outfile,
                         y_is_relative=False, scale_factor=1.0):

    n_loc = len(localities)
    ncols = 4
    nrows = 2

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        sharey=False
    )
    axes = np.array(axes).reshape(nrows, ncols)

    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].axis("off")

    for idx, loc in enumerate(localities):
        r = 0 if idx < 4 else 1
        c = idx if idx < 4 else (idx - 4) + 1

        ax = axes[r, c]
        ax.axis("on")
        
        ax.minorticks_off()

        sub = data[data["locality"] == loc]
        if sub.empty:
            ax.axis("off")
            continue

        pivot = sub.pivot_table(
            index="year", columns="phylum",
            values=value_col, aggfunc="sum", fill_value=0
        )

        years = pivot.index.values
        bottom = np.zeros(len(years))

        if y_is_relative:
            ax.set_ylim(0, 100)
        else:
            totals = pivot.sum(axis=1).values / scale_factor
            ax.set_ylim(0, max(totals) * 1.05 if len(totals) > 0 else 1)

        for ph in phyla:
            if ph not in pivot.columns:
                continue
            vals = pivot[ph].values / scale_factor
            ax.bar(years, vals, bottom=bottom, color=color_map[ph], width=0.8)
            bottom += vals

        ax.set_title(loc, fontsize=10)
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)

        if c == 0:
            ax.set_ylabel(ylabel)

    if len(localities) == 7:
        axes[1, 0].axis("off")

    plt.tight_layout(rect=[0, 0, 0.82, 0.95])

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[ph]) for ph in phyla]
    fig.legend(
        handles, phyla,
        title="Phylum",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0.,
        fontsize=8,
        title_fontsize=9
    )

    plt.savefig(outfile, dpi=300, format="png", bbox_inches="tight")
    plt.savefig(outfile.replace(".png", ".svg"), format="svg", bbox_inches="tight")
    plt.show()


plot_faceted_stacked(
    data=agg_rel,
    value_col="rel_density",
    ylabel="Relative Density (%)",
    title="Phytoplankton Community Composition Over Time",
    outfile=os.path.join(output_dir, "REL_density_phylum_locality_year.png"),
    y_is_relative=True
)

plot_faceted_stacked(
    data=agg_rel,
    value_col="rel_biovolume",
    ylabel="Relative Biovolume (%)",
    title="Phytoplankton Community Composition Over Time",
    outfile=os.path.join(output_dir, "REL_biovolume_phylum_locality_year.png"),
    y_is_relative=True
)


plot_faceted_stacked(
    data=agg,
    value_col="total_density",
    ylabel="Density (cells/L × 10⁶)",
    title="Phytoplankton Community Composition Over Time",
    outfile=os.path.join(output_dir, "ABS_density_phylum_locality_year.png"),
    y_is_relative=False,
    scale_factor=1e6
)

plot_faceted_stacked(
    data=agg,
    value_col="total_biovolume",
    ylabel="Biovolume (mm³/m³ × 10³)",
    title="Phytoplankton Community Composition Over Time",
    outfile=os.path.join(output_dir, "ABS_biovolume_phylum_locality_year.png"),
    y_is_relative=False,
    scale_factor=1e3
)





phyto = pd.read_csv(
    biotic_file,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

for col in ["phylum", "class", "order", "family", "genus", "specificEpithet", "acceptedNameUsage"]:
    if col in phyto.columns:
        phyto[col] = phyto[col].astype(str).str.strip()

tax_cols = ["phylum", "class", "order", "family", "genus", "specificEpithet", "acceptedNameUsage"]
tax_cols = [c for c in tax_cols if c in phyto.columns]

summary_lake = (
    phyto
    .groupby("locality")[tax_cols]
    .nunique()
    .reset_index()
)

rename_map = {
    "phylum":           "N_phyla",
    "class":            "N_classes",
    "order":            "N_orders",
    "family":           "N_families",
    "genus":            "N_genera",
    "specificEpithet":  "N_species",
    "acceptedNameUsage":"N_taxa"
}

summary_lake = summary_lake.rename(columns=rename_map)

summary_lake["Species_identification_%"] = (
    summary_lake["N_species"] / summary_lake["N_taxa"] * 100
).round(2)

print(summary_lake)

summary_lake.to_csv(
    os.path.join(output_dir, "Taxonomic_richness_by_lake_with_speciesID_percentage.csv"),
    sep = ";",
    index=False
)




phyto = pd.read_csv(
    biotic_file,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

phyto["year"] = pd.to_numeric(phyto["year"], errors="coerce").astype("Int64")


phyto_year = phyto.copy()


summary_year = (
    phyto_year
    .groupby(["locality", "year"])[tax_cols]
    .nunique()
    .reset_index()
    .rename(columns=rename_map)
)


print(summary_year.head())

summary_year.to_csv(
    os.path.join(output_dir, "Taxonomic_richness_by_lake_year.csv"),
    sep = ";",
    index=False
)


if "season" not in phyto.columns:
    season_map = {
        1:"Winter", 2:"Winter",
        3:"Spring", 4:"Spring", 5:"Spring",
        6:"Summer", 7:"Summer", 8:"Summer",
        9:"Autumn", 10:"Autumn", 11:"Autumn",
        12:"Winter"
    }
    phyto["month"] = pd.to_numeric(phyto["month"], errors="coerce")
    phyto["season"] = phyto["month"].map(season_map)

summary_season = (
    phyto
    .groupby(["locality", "season"])[tax_cols]
    .nunique()
    .reset_index()
    .rename(columns=rename_map)
)

print(summary_season.head())

summary_season.to_csv(os.path.join(output_dir, "Taxonomic_richness_by_lake_season.csv"), sep = ";", index=False)

