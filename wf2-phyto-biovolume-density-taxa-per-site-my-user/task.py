import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

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

phyto["totalBiovolume"] = pd.to_numeric(phyto["totalBiovolume"], errors="coerce")
phyto["density"]        = pd.to_numeric(phyto["density"],        errors="coerce")

phyto = phyto.dropna(subset=["density", "totalBiovolume"])


summary = (
    phyto
    .groupby("locality")
    .agg(
        Taxa=("acceptedNameUsage", "nunique"),
        Mean_biovolume=("totalBiovolume", "mean"),   # mm³/m³
        Total_biovolume=("totalBiovolume", "sum"),
        Mean_density=("density", "mean"),            # cells/L
        Total_density=("density", "sum")
    )
    .reset_index()
)

site_order = sorted(summary["locality"].unique())

summary["locality"] = pd.Categorical(summary["locality"],
                                     categories=site_order,
                                     ordered=True)
summary = summary.sort_values("locality")

print("\nSummary by lake (after filter location_config.csv):")
print(summary)

summary.to_csv(
    os.path.join(output_dir, "Phyto_summary_biovolume_density_taxa_by_site.csv"),
    sep = ";",
    index=False
)

max_taxa = summary["Taxa"].max()
taxa_ylim = max_taxa * 1.15  # a little space above the maximum

def make_sci_formatter(exp):
    """Returns a formatter that writes coeff×10^exp without any unnecessary zeros."""
    def _formatter(x, pos):
        if x == 0:
            return "0"
        coeff = x / (10**exp)
        s = f"{coeff:.1f}".rstrip("0").rstrip(".")
        return rf"${{{s}}}\times10^{{{exp}}}$"
    return _formatter

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(summary["locality"], summary["Total_biovolume"],
        color="#a6cee3")  # light blue
ax1.set_xlabel("Lake")

max_bv = summary["Total_biovolume"].max()
exp_bv = int(np.floor(np.log10(max_bv)))  # ex. if max≈1e6 -> 6

ax1.set_ylabel("Total biovolume (mm³/m³)", color="black")
ax1.tick_params(axis="y", labelcolor="black")
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(make_sci_formatter(exp_bv)))

ax2 = ax1.twinx()
ax2.plot(summary["locality"], summary["Taxa"], "ko", markersize=6)

for x, y in zip(summary["locality"], summary["Taxa"]):
    ax2.text(x, y + max_taxa * 0.03, f"{y}",
             ha="center", va="bottom", fontsize=9)

ax2.set_ylabel("Taxa richness", color="black")
ax2.tick_params(axis="y", labelcolor="black")
ax2.set_ylim(0, taxa_ylim)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(os.path.join(output_dir, "Fig1A_TotalBiovolume_Taxa_by_site.png"), dpi=300)
plt.savefig(os.path.join(output_dir, "Fig1A_TotalBiovolume_Taxa_by_site.svg"), format="svg")
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(summary["locality"], summary["Total_density"], color="#a6cee3")
ax1.set_xlabel("Lake")

max_den = summary["Total_density"].max()
exp_den = int(np.floor(np.log10(max_den)))

ax1.set_ylabel("Total density (cells/L)", color="black")
ax1.tick_params(axis="y", labelcolor="black")
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(make_sci_formatter(exp_den)))

min_den = summary["Total_density"].min()
ax1.set_ylim(bottom=min_den * 0.8)   # <-- remove the zero from the axis!

ax2 = ax1.twinx()
ax2.plot(summary["locality"], summary["Taxa"], "ko", markersize=6)
for x, y in zip(summary["locality"], summary["Taxa"]):
    ax2.text(x, y + max_taxa * 0.03, f"{y}", ha="center", va="bottom", fontsize=9)

ax2.set_ylabel("Taxa richness", color="black")
ax2.tick_params(axis="y", labelcolor="black")
ax2.set_ylim(0, taxa_ylim)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(os.path.join(output_dir, "Fig1B_TotalDensity_Taxa_by_site.png"), dpi=300)
plt.savefig(os.path.join(output_dir, "Fig1B_TotalDensity_Taxa_by_site.svg"), format="svg")
plt.show()

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
