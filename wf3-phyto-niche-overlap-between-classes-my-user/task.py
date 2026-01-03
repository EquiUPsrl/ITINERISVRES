import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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



def interval_overlap_asym(row_i, row_j):
    """
    Asymmetric overlap between two intervals [xmin, xmax]:

        overlap(i -> j) = length( intersection(i, j) ) / width(i)

    where width(i) = xmax_i - xmin_i.

    Returns 0 if intervals do not overlap or are invalid.
    """
    xi_min, xi_max = row_i["class_xmin"], row_i["class_xmax"]
    xj_min, xj_max = row_j["class_xmin"], row_j["class_xmax"]

    if any(pd.isna(v) for v in (xi_min, xi_max, xj_min, xj_max)):
        return 0.0

    width_i = xi_max - xi_min
    if width_i <= 0:
        return 0.0

    ov_min = max(xi_min, xj_min)
    ov_max = min(xi_max, xj_max)
    ov_len = max(0.0, ov_max - ov_min)

    return float(ov_len / width_i)



def execute_niche_overlap_between_classes(optTol_filtered_file, axis_to_use=1):


    print(f"Using Axis {axis_to_use}")
    print(f"Input file: {optTol_filtered_file}")
    
    
    
    df = pd.read_csv(optTol_filtered_file)
    
    df["sizeClass"] = df["sizeClass"].astype(str)
    
    df["xmin"] = df["optimum"] - df["tolerance"] / 2
    df["xmax"] = df["optimum"] + df["tolerance"] / 2
    
    df = df[(df["xmin"].notna()) &
            (df["xmax"].notna()) &
            ((df["xmax"] - df["xmin"]) > 0)].copy()
    
    print(f"Species with valid niche interval (Axis {axis_to_use}):", df.shape[0])
    
    order_classes = ["4", "3", "2", "1", "0", "-1"]
    df = df[df["sizeClass"].isin(order_classes)].copy()
    
    print("Species after filtering to the 6 main size classes:", df.shape[0])
    
    
    
    class_summary = (
        df.groupby("sizeClass")
          .agg(class_xmin=("xmin", "min"),
               class_xmax=("xmax", "max"))
          .reset_index()
    )
    
    class_summary["class_opt"] = (
        (class_summary["class_xmin"] + class_summary["class_xmax"]) / 2.0
    )
    
    intervals_outfile = os.path.join(output_dir, f"sizeclass_intervals_axis{axis_to_use}.csv")
    class_summary.rename(columns={"sizeClass": "class"}).to_csv(
        intervals_outfile,
        index=False,
        encoding="utf-8-sig"
    )
    
    print(f"Saved: {intervals_outfile}")
    print(class_summary)
    
    
    
    G = class_summary.copy().reset_index(drop=True)
    
    classes = (
        G["sizeClass"].astype(str).tolist()
        if "sizeClass" in G.columns
        else G["class"].astype(str).tolist()
    )
    
    n = len(G)
    M = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 1.0
            else:
                M[i, j] = interval_overlap_asym(G.iloc[i], G.iloc[j])
    
    mat_df = pd.DataFrame(M, index=classes, columns=classes)
    matrix_outfile = os.path.join(output_dir, f"overlap_axis{axis_to_use}_sizeclasses_matrix.csv")
    mat_df.to_csv(matrix_outfile, encoding="utf-8-sig")
    
    long_df = (
        mat_df
        .reset_index()
        .melt(id_vars="index", var_name="class_j", value_name="overlap")
        .rename(columns={"index": "class_i"})
    )
    
    long_outfile = os.path.join(output_dir, f"overlap_axis{axis_to_use}_sizeclasses_long.csv")
    long_df.to_csv(long_outfile, index=False, encoding="utf-8-sig")
    
    print("Saved:")
    print(f"- {matrix_outfile}")
    print(f"- {long_outfile}")

    return matrix_outfile, long_outfile, intervals_outfile


matrix_outfile_axis1, long_outfile_axis1, intervals_outfile_axis1 = execute_niche_overlap_between_classes(out_filtered_file_axis1, 1)

matrix_outfile_axis2, long_outfile_axis2, intervals_outfile_axis2 = execute_niche_overlap_between_classes(out_filtered_file_axis2, 2)







def plot_sizeclass_overlap_heatmap(csv_path, title=None, cmap="viridis",
                                   order=("4", "3", "2", "1", "0", "-1"), axis_to_use=1):
    """
    Heatmap for the size class overlap matrix.
    csv_path: path to the CSV file for the NxN matrix (rows/columns = sizeClass)
    title: optional title for the graph
    cmap: colormap (default: viridis)
    order: desired order of the size classes (default: 4 → -1)
    """
    
    mat_df = pd.read_csv(csv_path, index_col=0)
    
    order = list(order)
    classes_present = [c for c in order if c in mat_df.index]
    if classes_present:
        mat_df = mat_df.loc[classes_present, classes_present]
    else:
        classes_present = mat_df.index.tolist()
    
    M = mat_df.values.astype(float)
    labels = classes_present
    n = len(labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, vmin=0, vmax=1, cmap=cmap)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Size class j")
    ax.set_ylabel("Size class i")

    if title is None:
        title = f"Overlap between size class – {csv_path}"
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Asymmetric overlap (i → j)")

    for i in range(n):
        for j in range(n):
            val = M[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="white" if val > 0.5 else "black",
                    fontsize=8)

    plt.tight_layout()

    fig_name = os.path.join(output_dir, f"overlap_axis{axis_to_use}_sizeclasses_matrix.png")
    plt.savefig(fig_name, dpi=300)
    print(f"Figure saved as: {fig_name}")
    
    plt.show()


plot_sizeclass_overlap_heatmap(
    matrix_outfile_axis1,
    title=f"Overlap tra size class – Axis 1",
    axis_to_use=1
)

plot_sizeclass_overlap_heatmap(
    matrix_outfile_axis2,
    title=f"Overlap tra size class – Axis 2",
    axis_to_use=2
)






interval_file_axis1 = intervals_outfile_axis1
interval_file_axis2 = intervals_outfile_axis2


def plot_sizeclass_intervals(interval_file, axis_to_use=1):

    print(f"Using Axis {axis_to_use}")
    print(f"Reading: {interval_file}")
    
    
    df = pd.read_csv(interval_file)
    
    df["class"] = df["class"].astype(str)
    df["class_num"] = df["class"].astype(float)
    
    order = ["4", "3", "2", "1", "0", "-1"]
    df = df[df["class"].isin(order)].copy()
    df["class"] = pd.Categorical(df["class"], categories=order, ordered=True)
    df = df.sort_values("class")
    
    print(df)
    
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for _, row in df.iterrows():
        cls = row["class"]
        x = row["class_num"]
        ymin = row["class_xmin"]
        ymax = row["class_xmax"]
        yopt = row["class_opt"]
    
        ax.vlines(x, ymin, ymax, color="0.75", linewidth=14, zorder=2)
    
        ax.plot(x, yopt, marker="_", markersize=16, color="black", zorder=3)
    
    
    
    ax.set_xlabel("Size class")
    ax.set_ylabel(f"Axis {axis_to_use} (canonical gradient)")
    
    ax.set_xticks(df["class_num"])
    ax.set_xticklabels(df["class"].astype(str))
    
    ax.set_xlim(df["class_num"].max() + 0.5,
                df["class_num"].min() - 0.5)
    
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()

    fig_name = os.path.join(output_dir, f"sizeclass_intervals_axis{axis_to_use}.png")
    plt.savefig(fig_name, dpi=300)
    print(f"Figure saved as: {fig_name}")
    
    plt.show()


plot_sizeclass_intervals(interval_file_axis1, 1)

plot_sizeclass_intervals(interval_file_axis2, 2)

