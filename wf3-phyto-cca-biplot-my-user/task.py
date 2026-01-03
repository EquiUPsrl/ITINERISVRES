import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--cca1_env_loadings_path', action='store', type=str, required=True, dest='cca1_env_loadings_path')


args = arg_parser.parse_args()
print(args)

id = args.id

cca1_env_loadings_path = args.cca1_env_loadings_path.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

output_dir = os.path.join(conf_output_path, "CCA")
os.makedirs(output_dir, exist_ok=True)

env_file = cca1_env_loadings_path  # output R
flip_y   = False   # True to flip axis Y
flip_x   = False   # True to flip axis X
savefig  = True
fig_name = os.path.join(output_dir, "cca1_biplot_abiotic_only.png")

env = pd.read_csv(env_file)

x = env["CCA1"].values
y = env["CCA2"].values
labels = env["Variable"].astype(str).values

if flip_x:
    y = -y
if flip_y:
    x = -x

max_abs = max(np.max(np.abs(x)), np.max(np.abs(y)))
pad = 0.1 * max_abs if max_abs > 0 else 0.1
xlim = (-max_abs - pad, max_abs + pad)
ylim = (-max_abs - pad, max_abs + pad)

fig, ax = plt.subplots(figsize=(7, 7))

ax.axhline(0, linewidth=1)
ax.axvline(0, linewidth=1)

for xi, yi, lab in zip(x, y, labels):
    ax.annotate(
        "", xy=(xi, yi), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", linewidth=1.5)
    )
    ax.text(xi * 1.05, yi * 1.05, lab,
            fontsize=10, ha="center", va="center")

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("CCA1")
ax.set_ylabel("CCA2")
ax.set_title("CCA1 – Biplot abiotic variables")

ax.set_aspect("equal", adjustable="box") 
plt.tight_layout()

if savefig:
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    print(f"Figura salvata come: {fig_name}")

plt.show()

