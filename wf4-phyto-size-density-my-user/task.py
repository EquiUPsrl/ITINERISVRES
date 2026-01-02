import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import stats

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--data_filtering_dir', action='store', type=str, required=True, dest='data_filtering_dir')


args = arg_parser.parse_args()
print(args)

id = args.id

data_filtering_dir = args.data_filtering_dir.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF4/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF4/' + 'data'

input_dir = data_filtering_dir

output_dir = os.path.join(conf_output_path, "size_density")
os.makedirs(output_dir, exist_ok=True)


taxlev = "scientificname"
x_col  = "mean biovolume"
y_col  = "density"

palette = {
    "100%": "#1f77b4",
    "99%":  "#ff7f0e",
    "95%":  "#2ca02c",
    "90%":  "#d62728",
    "75%":  "#9467bd",
}


action_name = "size_density"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]
    
    p = act.set_index("parameter")["value"]
    
    taxlev = p.get("taxlev", taxlev)
    x_col = p.get("x_col", x_col)
    y_col = p.get("y_col", y_col)

print("Selected taxlev: ", taxlev)
print("Selected x_col: ", x_col)
print("Selected y_col: ", y_col)

def fit_and_ci(x, y):
    lx, ly = np.log(x), np.log(y)
    beta, alpha = np.polyfit(lx, ly, 1)     # ly = alpha + beta*lx
    a, b = np.exp(alpha), beta

    ly_hat = alpha + beta * lx
    ss_res = np.sum((ly - ly_hat)**2)
    ss_tot = np.sum((ly - ly.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    X = np.column_stack([np.ones_like(lx), lx])
    n, p = X.shape
    sigma2 = ss_res / max(n - p, 1)
    XtX_inv = np.linalg.inv(X.T @ X)

    xs  = np.linspace(x.min(), x.max(), 200)
    lxs = np.log(xs)
    ly_pred = alpha + beta * lxs

    try:
        tcrit = stats.t.ppf(0.975, df=max(n - p, 1))
    except Exception:
        tcrit = 1.96

    V = np.vstack([np.ones_like(lxs), lxs]).T
    var_mean = np.einsum('ij,jk,ik->i', V, XtX_inv, V) * sigma2
    se_mean  = np.sqrt(np.maximum(var_mean, 0))

    y_fit = np.exp(ly_pred)
    y_lo  = np.exp(ly_pred - tcrit * se_mean)
    y_hi  = np.exp(ly_pred + tcrit * se_mean)
    return a, b, r2, xs, y_fit, y_lo, y_hi

folders = [
    f for f in os.listdir(input_dir)
    if os.path.isdir(os.path.join(input_dir, f)) and not f.startswith(".")
]

print("Folders: ", folders)

if not folders:
    print(f"⚠️ No folders found in: {input_dir}")
    raise SystemExit

for folder in folders:
    folder_path = os.path.join(input_dir, folder)

    all_files = [
        f for f in os.listdir(folder_path)
        if f.lower().startswith("filtered_") and f.lower().endswith(".csv")
    ]

    if not all_files:
        print(f"⚠️ No 'filtered_*.csv' files found in: {folder_path}")
        continue  # move to the next folder

    files = {}
    for f in sorted(all_files):
        m = re.search(r"_thr(\d{2,3})_", f.lower())
        if m:
            thr = m.group(1)
            label = f"{thr}%"
            files[label] = f

    sorted_files = sorted(
        files.items(),
        key=lambda x: int(x[0].rstrip('%')),
        reverse=True
    )


    all_series = []   # (label, x_points, y_points, xs_fit, y_fit)

    out_dir = os.path.join(output_dir, folder)    # where to save graphs and fits
    os.makedirs(out_dir, exist_ok=True)
    
    
    for label, fname in sorted_files:
        path = os.path.join(folder_path, fname)
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            continue
    
        df = pd.read_csv(path, sep=';', decimal='.', low_memory=False)
        df.columns = df.columns.str.strip()
    
        if taxlev not in df.columns or x_col not in df.columns or y_col not in df.columns:
            print(f"⚠️ Missing columns in {fname}: needed '{taxlev}', '{x_col}', '{y_col}'. Salto.")
            continue
    
        den = df.groupby(taxlev, dropna=False)[y_col].sum()
        size = df.groupby(taxlev, dropna=False)[x_col].mean()
        mat = pd.concat([den, size], axis=1).reset_index()
        mat.columns = [taxlev, y_col, x_col]
        valid = mat[(mat[y_col] > 0) & (mat[x_col] > 0)].copy()
    
        if valid.empty:
            print(f"⚠️ No valid data for: {label}")
            continue
    
        a, b, r2, xs, y_fit, y_lo, y_hi = fit_and_ci(valid[x_col].to_numpy(), valid[y_col].to_numpy())
    
        plt.figure(figsize=(6, 5))
        plt.scatter(valid[x_col], valid[y_col], s=18)
        plt.plot(xs, y_fit)
        plt.fill_between(xs, y_lo, y_hi, color='#ADD8E6', alpha=0.35)
    
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('average biovolume ($\\mu m^3$)')
        plt.ylabel('density (cell·L$^{-1}$)')
        plt.title(f"Size–density (threshold {label})")
        plt.legend([f"a={a:.2g}, b={b:.2f}, R²={r2:.2f}"])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sizedensity_{label}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
        pd.DataFrame({'threshold':[label], 'a':[a], 'b':[b], 'R2':[r2]}).to_csv(
            os.path.join(out_dir, f"sizedensity_fit_{label}.csv"),
            sep=';', index=False
        )
    
        all_series.append((label, valid[x_col].to_numpy(), valid[y_col].to_numpy(), xs, y_fit))
    
    if all_series:
        plt.figure(figsize=(7, 6))
        for label, xpts, ypts, xs, yfit in all_series:
            c = palette.get(label, None)
            plt.scatter(xpts, ypts, s=18, label=label, color=c)
            plt.plot(xs, yfit, color=c, linestyle='-')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('average biovolume ($\\mu m^3$)')
        plt.ylabel('density (cell·L$^{-1}$)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sizedensity_ALL_colored.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Charts created in: {out_dir}")
    else:
        print("⚠️ No valid series found.")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
