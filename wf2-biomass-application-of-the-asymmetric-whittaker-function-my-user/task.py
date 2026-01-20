import pandas as pd
import os
import numpy as np

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--stats_path', action='store', type=str, required=True, dest='stats_path')


args = arg_parser.parse_args()
print(args)

id = args.id

stats_path = args.stats_path.replace('"','')


conf_input_path = conf_input_path = '/tmp/data/WF2/work/' + 'input'

def whittaker_asymmetric(y, lam=500, p=0.1, niter=10):
    """
    Whittaker asymmetric smoother (MODIS-type):
    - lam : smoothing parameter (higher = smoother)
    - p : weight for points above the curve (usually small, e.g., 0.1)
    - niter : number of reweighting iterations
    """
    y = np.asarray(y, dtype=float)
    m = y.size

    E = np.eye(m)
    D = np.diff(E, n=2, axis=0)
    DTD = D.T @ D

    w = np.ones(m)
    z = y.copy()

    for _ in range(niter):
        W = np.diag(w)
        A = W + lam * DTD
        b = W @ y
        z = np.linalg.solve(A, b)

        w = np.where(y >= z, p, 1 - p)

    return z




output_base_folder = stats_path
final_stats_path = output_base_folder

products_list = []
target_columns = ["mean"]

df = pd.read_csv(os.path.join(conf_input_path, "config_whittaker.csv"), sep=';')
products_list = df['products'].dropna().tolist()

print("Loaded Products:", products_list)

for filename in os.listdir(output_base_folder):
    
    product_found = [p for p in products_list if p in filename]
    
    if filename.endswith(".csv") and product_found:
        
        file_path = os.path.join(output_base_folder, filename)
        print(f"Processing of {filename}...")

        df = pd.read_csv(file_path)

        for target_col in target_columns:
            y = df[target_col].values        

            lam = 5
            p   = 0.1
            niter = 1
            
            y_whit_asym = whittaker_asymmetric(y, lam=lam, p=p, niter=niter)

            df[target_col] = y_whit_asym

        new_path = os.path.join(final_stats_path, filename)

        df.to_csv(new_path, index=False)
        print(f"✅ Correct file saved in: {new_path}\n")

file_final_stats_path = open("/tmp/final_stats_path_" + id + ".json", "w")
file_final_stats_path.write(json.dumps(final_stats_path))
file_final_stats_path.close()
