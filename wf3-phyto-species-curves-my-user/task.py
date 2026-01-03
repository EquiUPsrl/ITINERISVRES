import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import PoissonRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--bio_file_filtered', action='store', type=str, required=True, dest='bio_file_filtered')

arg_parser.add_argument('--biotic_size_class', action='store', type=str, required=True, dest='biotic_size_class')

arg_parser.add_argument('--cca2_site_scores_path', action='store', type=str, required=True, dest='cca2_site_scores_path')


args = arg_parser.parse_args()
print(args)

id = args.id

bio_file_filtered = args.bio_file_filtered.replace('"','')
biotic_size_class = args.biotic_size_class.replace('"','')
cca2_site_scores_path = args.cca2_site_scores_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)


bio_file         = bio_file_filtered
site_scores_file = cca2_site_scores_path   # site scores from CCA model in R
sizeclass_file   = biotic_size_class              # sizeClass per species



def normalize_name(s: str) -> str:
    """Normalize species names in a robust way (lowercase, remove spaces and dots)."""
    return (
        str(s)
        .lower()
        .replace(" ", "")
        .replace(".", "")
    )


def fit_poisson_quadratic_sklearn(x, y):
    """
    Fit a quadratic Poisson GLM with a log link:

        log(mu) = beta0 + beta1 * x + beta2 * x^2

    This is equivalent to the R formula:
        glm(y ~ ax + I(ax^2),
            family = poisson(link = "log"),
            weights = sqrt(y + 1))

    Parameters
    ----------
    x : array-like
        Canonical axis scores (e.g. CCA1 or CCA2, scaling 2, standardized).
    y : array-like
        Species abundances (non-negative).

    Returns
    -------
    model : PoissonRegressor or None
    poly  : PolynomialFeatures or None
    """
    mask = ~np.isnan(y) & ~np.isnan(x)
    x_valid = x[mask]
    y_valid = y[mask]

    if y_valid.size == 0 or np.all(y_valid == 0):
        return None, None

    w = np.sqrt(y_valid + 1.0)

    x_valid = x_valid.reshape(-1, 1)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(x_valid)

    model = PoissonRegressor(alpha=0.0, max_iter=1000)
    try:
        model.fit(X_poly, y_valid, sample_weight=w)
    except Exception:
        return None, None

    return model, poly


def extract_opt_tol_from_model(model, window=(-3.0, 3.0)):
    """
    Compute optimum, tolerance and unimodality flag from a quadratic Poisson GLM
    with:

        log(mu) = beta0 + beta1 * x + beta2 * x^2

    Optimum (u*) and tolerance (t) follow the standard Gaussian-curve
    parameterization used in CANOCO-style plots.

    - optimum and tolerance are computed without truncation.
    - unimodal = True only if:
        beta2 < 0  and  optimum is within [window_min, window_max]

    Parameters
    ----------
    model : PoissonRegressor
    window : tuple (float, float)
        Interval in which the optimum must lie in order to be considered "unimodal".

    Returns
    -------
    dict with keys: 'optimum', 'tolerance', 'unimodal'
    """
    if model is None:
        return dict(optimum=np.nan, tolerance=np.nan, unimodal=False)

    coefs = model.coef_
    if coefs.size < 2:
        return dict(optimum=np.nan, tolerance=np.nan, unimodal=False)

    beta1, beta2 = coefs[0], coefs[1]

    if np.isnan(beta2) or beta2 >= 0:
        return dict(optimum=np.nan, tolerance=np.nan, unimodal=False)

    optimum = -beta1 / (2.0 * beta2)
    tolerance = np.sqrt(-1.0 / (2.0 * beta2))

    w_min, w_max = window
    unimodal = (w_min <= optimum <= w_max)

    return dict(optimum=optimum, tolerance=tolerance, unimodal=unimodal)


def predict_curve(model, poly, grid):
    """
    Predict species response over the given axis grid.
    """
    Xg = poly.transform(grid.reshape(-1, 1))
    return model.predict(Xg)



def load_bio(bio_file):
    """
    Load species abundance (wide format: samples x species).
    """
    bio = pd.read_csv(
        bio_file,
        sep=",",
        na_values=["", "NA", "NaN", "na", "-"],
        encoding="utf-8-sig"
    )

    if "ID" in bio.columns:
        bio.set_index("ID", inplace=True)

    bio = bio.apply(pd.to_numeric, errors="coerce")

    print("bio shape (samples x species):", bio.shape)
    return bio


def get_axis_column_name(sites, axis_number: int) -> str:
    """
    Find the column name for the requested canonical axis (1 or 2).
    """
    if axis_number == 1:
        candidates_axis = ["CCA1_std", "CCA1", "Axis1", "CCA1_raw"]
    elif axis_number == 2:
        candidates_axis = ["CCA2_std", "CCA2", "Axis2", "CCA2_raw"]
    else:
        raise ValueError("axis_number must be 1 or 2.")

    axis_col = next((c for c in candidates_axis if c in sites.columns), None)
    if axis_col is None:
        raise ValueError(
            f"Cannot find column for canonical axis {axis_number} in "
            f"{site_scores_file}. Available columns: {sites.columns.tolist()}"
        )
    return axis_col


def load_sites_and_align(site_scores_file, bio, axis_number: int):
    """
    Load site scores, align samples with 'bio', and extract the chosen axis.
    """
    sites = pd.read_csv(site_scores_file)

    if "SampleID" in sites.columns:
        sites.set_index("SampleID", inplace=True)

    axis_col = get_axis_column_name(sites, axis_number)

    common_ids = bio.index.intersection(sites.index)
    bio   = bio.loc[common_ids].copy()
    sites = sites.loc[common_ids].copy()

    bio = bio.loc[bio.sum(axis=1) > 0]
    sites = sites.loc[bio.index]

    print("Samples used for curves:", bio.shape[0])

    axis_vals = sites[axis_col].values  # np.array
    return bio, sites, axis_vals, axis_col


def load_sizeclasses(sizeclass_file, bio):
    """
    Build taxa and size-class mapping for species in 'bio'.

    Returns
    -------
    taxa_info : DataFrame
        Index = bio_name (column names in 'bio'),
        Columns = ["SizeClass", "Taxa"].
    class_to_species : dict
        { sizeClass: [bio_name, ...] }
    size_classes : list
        Ordered list of sizeClass to be used in plots.
    """
    sc = pd.read_csv(sizeclass_file, sep=",", encoding="utf-8-sig")

    if "scientificName" not in sc.columns or "sizeClass" not in sc.columns:
        raise ValueError(
            f"'scientificName' or 'sizeClass' not found in {sizeclass_file}. "
            f"Available columns: {sc.columns.tolist()}"
        )

    sc_simple = sc[["scientificName", "sizeClass"]].dropna().copy()
    sc_simple["scientificName"] = sc_simple["scientificName"].astype(str)
    sc_simple["norm_name"] = sc_simple["scientificName"].apply(normalize_name)

    sc_group = (
        sc_simple.groupby("norm_name")
        .agg({
            "scientificName": lambda x: x.iloc[0],
            "sizeClass":     lambda x: x.value_counts().index[0]
        })
        .reset_index()
    )

    bio_cols = pd.DataFrame({"bio_name": bio.columns})
    bio_cols["norm_name"] = bio_cols["bio_name"].astype(str).apply(normalize_name)

    merge_sc = pd.merge(
        bio_cols,
        sc_group[["norm_name", "scientificName", "sizeClass"]],
        on="norm_name",
        how="inner"
    )

    print("Species in bio:", bio.shape[1])
    print("Species with assigned sizeClass:", merge_sc["bio_name"].nunique())

    taxa_info = merge_sc.set_index("bio_name")[["sizeClass", "scientificName"]].copy()
    taxa_info.columns = ["SizeClass", "Taxa"]

    class_to_species = (
        taxa_info.reset_index()
        .groupby("SizeClass")["bio_name"]
        .apply(list)
        .to_dict()
    )

    desired_order = [4, 3, 2, 1, 0, -1]
    size_classes = [c for c in desired_order if c in class_to_species]

    print("Size classes used for curves:", size_classes)
    return taxa_info, class_to_species, size_classes



def run_axis_curves(
    axis_number: int,
    bio_file: str,
    site_scores_file: str,
    sizeclass_file: str,
    pred_range=None
):
    """
    Run CANOCO-style Poisson response curves along the chosen axis.

    Parameters
    ----------
    axis_number : int
        1 or 2 (which canonical axis to use).
    bio_file : str
        Path to species abundance table.
    site_scores_file : str
        Path to CCA site scores table.
    sizeclass_file : str
        Path to size-class file (scientificName + sizeClass).
    pred_range : array-like, optional
        Range of axis values used for curve prediction.

    Returns
    -------
    curves_axis : dict
        { "AxisX|sizeClass|bio_name": predicted_values }
    optTol_df : DataFrame
        Optimum, tolerance and unimodality info for each species.
    """
    if pred_range is None:
        pred_range = np.arange(-3.0, 3.01, 0.01)

    bio = load_bio(bio_file)

    bio, sites, axis_vals, axis_col = load_sites_and_align(
        site_scores_file, bio, axis_number
    )

    taxa_info, class_to_species, size_classes = load_sizeclasses(
        sizeclass_file, bio
    )

    n_classes = len(size_classes)
    nrows, ncols = 3, 2
    if n_classes < 6:
        nrows = int(np.ceil(n_classes / 2))

    curves_axis = {}   # key: f"Axis{axis_number}|sizeClass|bio_name"
    optTol_list = []   # list of dicts with optimum/tolerance/unimodal

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False
    )

    axis_label = f"Axis {axis_number} ({axis_col})"

    for idx_cls, cls in enumerate(size_classes):
        row = idx_cls // ncols
        col = idx_cls % ncols
        ax = axs[row, col]

        species_list = class_to_species[cls]

        curves_cls = {}
        labels_cls = {}
        max_y_all = -np.inf

        for sp in species_list:
            if sp not in bio.columns:
                continue

            y = bio[sp].values.astype(float)
            if np.all((y == 0) | np.isnan(y)):
                continue

            model, poly = fit_poisson_quadratic_sklearn(axis_vals, y)
            if model is None:
                continue

            ypred = predict_curve(model, poly, pred_range)
            curves_cls[sp] = ypred
            max_y_all = max(max_y_all, np.nanmax(ypred))

            key = f"Axis{axis_number}|{cls}|{sp}"
            curves_axis[key] = ypred

            sci_name = taxa_info.loc[sp, "Taxa"]
            labels_cls[sp] = sci_name

            ot = extract_opt_tol_from_model(model, window=(-3.0, 3.0))
            optTol_list.append({
                "axis": axis_number,
                "axis_col": axis_col,
                "sizeClass": cls,
                "bio_name": sp,
                "scientificName": sci_name,
                "optimum": ot["optimum"],
                "tolerance": ot["tolerance"],
                "unimodal": ot["unimodal"],
            })

        if not np.isfinite(max_y_all) or max_y_all <= 0 or len(curves_cls) == 0:
            ax.set_xlim([-4, 4])
            ax.set_ylim([0, 1])
            ax.set_xlabel(axis_label)
            ax.set_ylabel("Abundance")
            ax.set_title(f"Size Class {cls}")
            ax.axhline(0, linestyle=":", linewidth=0.5)
            ax.axvline(0, linestyle=":", linewidth=0.5)
            continue

        ax.set_xlim([-4, 4])                          # fixed SD range
        ax.set_ylim([0, max_y_all * 1.1])             # dynamic y-limit
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Abundance")
        ax.set_title(f"Size Class {cls}")
        ax.axhline(0, linestyle=":", linewidth=0.5)
        ax.axvline(0, linestyle=":", linewidth=0.5)

        for sp, ypred in curves_cls.items():
            ax.plot(pred_range, ypred)
            max_idx = int(np.nanargmax(ypred))
            xval = float(pred_range[max_idx])
            yval = float(ypred[max_idx])
            lab  = labels_cls[sp]
            ax.text(xval, yval, lab, fontsize=6, va="bottom")

    total_panels = nrows * ncols
    for k in range(len(size_classes), total_panels):
        r = k // ncols
        c = k % ncols
        axs[r, c].axis("off")

    plt.tight_layout()
    
    fig_name = os.path.join(output_dir, f"output_axis{axis_number}_curves.png")
    plt.savefig(fig_name, dpi=300)
    print(f"Figure saved as: {fig_name}")
    
    plt.show()

    if curves_axis:
        df_axis = pd.DataFrame(
            {name: vals for name, vals in curves_axis.items()},
            index=pred_range
        )
        df_axis.index.name = f"axis{axis_number}_value"
        curves_file_out = os.path.join(output_dir, f"output_axis{axis_number}_python_CANOCOstyle_curves.csv")
        df_axis.to_csv(curves_file_out)
        print("Saved:", curves_file_out)

    optTol_df = pd.DataFrame(optTol_list)
    optTol_file_out = os.path.join(output_dir, f"output_axis{axis_number}_python_CANOCOstyle_optTol.csv")
    optTol_df.to_csv(optTol_file_out, index=False)
    print("Saved:", optTol_file_out)

    return curves_file_out, optTol_file_out


curves_file_out_axis1, optTol_file_out_axis1 = run_axis_curves(
    axis_number=1,
    bio_file=bio_file,
    site_scores_file=site_scores_file,
    sizeclass_file=sizeclass_file
)

curves_file_out_axis2, optTol_file_out_axis2 = run_axis_curves(
    axis_number=2,
    bio_file=bio_file,
    site_scores_file=site_scores_file,
    sizeclass_file=sizeclass_file
)

file_optTol_file_out_axis1 = open("/tmp/optTol_file_out_axis1_" + id + ".json", "w")
file_optTol_file_out_axis1.write(json.dumps(optTol_file_out_axis1))
file_optTol_file_out_axis1.close()
file_optTol_file_out_axis2 = open("/tmp/optTol_file_out_axis2_" + id + ".json", "w")
file_optTol_file_out_axis2.write(json.dumps(optTol_file_out_axis2))
file_optTol_file_out_axis2.close()
