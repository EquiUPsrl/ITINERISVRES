import os
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--out_filtered_file_axis1', action='store', type=str, required=True, dest='out_filtered_file_axis1')

arg_parser.add_argument('--out_filtered_file_axis2', action='store', type=str, required=True, dest='out_filtered_file_axis2')


args = arg_parser.parse_args()
print(args)

id = args.id

out_filtered_file_axis1 = args.out_filtered_file_axis1.replace('"','')
out_filtered_file_axis2 = args.out_filtered_file_axis2.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

output_dir = os.path.join(conf_output_path, "NICHE_OVERLAP")
os.makedirs(output_dir, exist_ok=True)


optTol_filtered_file_axis1 = out_filtered_file_axis1
optTol_filtered_file_axis2 = out_filtered_file_axis2



def compute_overlap(row_i, row_j):
    """
    Compute asymmetric niche overlap from species i to species j.

    Overlap(i → j) = length( intersection( [xmin_i, xmax_i], [xmin_j, xmax_j] ) )
                     -----------------------------------------------------------
                                 width of i's niche (xmax_i - xmin_i)

    Returns 0 if intervals do not overlap or are invalid.
    """
    xi_min, xi_max = row_i["xmin"], row_i["xmax"]
    xj_min, xj_max = row_j["xmin"], row_j["xmax"]

    if any(pd.isna([xi_min, xi_max, xj_min, xj_max])):
        return 0.0

    width_i = xi_max - xi_min
    if width_i <= 0:
        return 0.0

    ov_min = max(xi_min, xj_min)
    ov_max = min(xi_max, xj_max)
    ov_len = max(0.0, ov_max - ov_min)

    return float(ov_len / width_i)


def execute_niche_overlap_intra_class(optTol_filtered_file, axis_to_use):

    print(f"Using Axis {axis_to_use}")
    print(f"Input file: {optTol_filtered_file}")
    
    
    
    df = pd.read_csv(optTol_filtered_file)
    
    df["sizeClass"] = df["sizeClass"].astype(str)
    
    df["xmin"] = df["optimum"] - df["tolerance"] / 2
    df["xmax"] = df["optimum"] + df["tolerance"] / 2
    
    df = df[(df["xmin"].notna()) &
            (df["xmax"].notna()) &
            ((df["xmax"] - df["xmin"]) > 0)].copy()
    
    print("Species with valid niche interval:", df.shape[0])


    
    
    all_long_rows = []
    
    for cls, g in df.groupby("sizeClass", sort=False):
    
        g = g.reset_index(drop=True)
        species = g["scientificName"].tolist()
        n = len(g)
    
        M = np.zeros((n, n))
    
        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i, j] = 1.0
                else:
                    M[i, j] = compute_overlap(g.iloc[i], g.iloc[j])
    
        mat_df = pd.DataFrame(M, index=species, columns=species)
        mat_file = os.path.join(output_dir, f"overlap_axis{axis_to_use}_sizeclass_{cls}.csv")
        mat_df.to_csv(mat_file, encoding="utf-8", index=True)
    
        long_df = (
            mat_df
            .reset_index()
            .melt(id_vars="index", var_name="species_j", value_name="overlap")
            .rename(columns={"index": "species_i"})
        )
        long_df.insert(0, "sizeClass", cls)
    
        all_long_rows.append(long_df)
    
    
    
    all_long = pd.concat(all_long_rows, ignore_index=True)
    all_long_file = os.path.join(output_dir, f"overlap_axis{axis_to_use}_all_sizeclasses_long.csv")
    all_long.to_csv(all_long_file, index=False)
    
    print("\nCreated:")
    print(f" - overlap_axis{axis_to_use}_sizeclass_<CLS>.csv  (intra-class overlap matrices for each size class)")
    print(f" - {all_long_file}  (all intra-class pairs in long format)")


execute_niche_overlap_intra_class(optTol_filtered_file=optTol_filtered_file_axis1, axis_to_use=1)

execute_niche_overlap_intra_class(optTol_filtered_file=optTol_filtered_file_axis2, axis_to_use=2)





def show_overlap_heatmaps(pattern, axis_label="Axis 1", axis_to_use=1):
    """
    Displays all overlap file heatmaps, one after the other, in the notebook.
    
    pattern: e.g., "overlap_axis1_sizeclass_*.csv"
    axis_label: Used only in figure titles
    """
    class_mats = sorted(glob(pattern))

    if not class_mats:
        print(f"No files found with pattern: {pattern}")
        return

    for path in class_mats:
        mat_df = pd.read_csv(path, index_col=0)
        labels = mat_df.index.astype(str).tolist()
        M = mat_df.values.astype(float)
        n = M.shape[0]

        if n < 2:
            print(f"Jump {path} (less than 2 species)")
            continue

        fig_size = (max(6, 0.4 * n + 3), max(6, 0.4 * n + 3))
        fig, ax = plt.subplots(figsize=fig_size)

        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)

        cls = Path(path).stem.split("sizeclass_")[-1]

        ax.set_title(f"Overlap between species — Size class {cls} ({axis_label})")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Overlap")

        for i in range(n):
            for j in range(n):
                val = M[i, j]
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color="white" if val > 0.5 else "black",
                        fontsize=6)

        plt.tight_layout()

        fig_name = os.path.join(output_dir, f"overlap_axis{axis_to_use}_sizeclass_{cls}.png")
        plt.savefig(fig_name, dpi=300)
        print(f"Figure saved as: {fig_name}")
        
        plt.show()


show_overlap_heatmaps(os.path.join(output_dir, f"overlap_axis1_sizeclass_*.csv"), axis_label="Axis 1", axis_to_use=1)

show_overlap_heatmaps(os.path.join(output_dir, f"overlap_axis2_sizeclass_*.csv"), axis_label="Axis 2", axis_to_use=2)

output_dir = conf_output_path

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
