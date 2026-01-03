import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--optTol_file_out_axis1', action='store', type=str, required=True, dest='optTol_file_out_axis1')

arg_parser.add_argument('--optTol_file_out_axis2', action='store', type=str, required=True, dest='optTol_file_out_axis2')


args = arg_parser.parse_args()
print(args)

id = args.id

optTol_file_out_axis1 = args.optTol_file_out_axis1.replace('"','')
optTol_file_out_axis2 = args.optTol_file_out_axis2.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)


optTol_file_axis1 = optTol_file_out_axis1
optTol_file_axis2 = optTol_file_out_axis2

def eliminate_outliers(optTol_file, axis_to_use):
    out_filtered_file = os.path.join(output_dir, f"optTol_axis{axis_to_use}_filtered.csv")
    
    print(f"Using Axis {axis_to_use}")
    print(f"Input  file: {optTol_file}")
    print(f"Output file: {out_filtered_file}")
    
    
    
    optTol = pd.read_csv(optTol_file)
    
    
    optTol = optTol.dropna(subset=["optimum"])
    
    if "unimodal" in optTol.columns:
        optTol = optTol[optTol["unimodal"] == True]
    
    
    
    opt_all = optTol["optimum"]
    
    mean_all = opt_all.mean()
    sd_all   = opt_all.std()
    
    lower = mean_all - 3.0 * sd_all
    upper = mean_all + 3.0 * sd_all
    
    print(f"\nAxis {axis_to_use}")
    print(f"Mean optimum (all species)   : {mean_all:.3f}")
    print(f"SD optimum   (all species)   : {sd_all:.3f}")
    print(f"Kept interval [mean ± 3*sd]  : [{lower:.3f}, {upper:.3f}]")
    
    filtered_optTol = optTol[(optTol["optimum"] >= lower) &
                             (optTol["optimum"] <= upper)].copy()
    
    filtered_optTol["globalMeanOptimum"] = mean_all
    filtered_optTol["globalSdOptimum"]   = sd_all
    
    filtered_optTol.to_csv(out_filtered_file, index=False, encoding="utf-8-sig")
    
    orig_n = len(pd.read_csv(optTol_file))
    
    print("\nTotal species (original file)         :", orig_n)
    print("Species after NA/unimodal filter      :", len(optTol))
    print("Species kept after global ±3*sd filter:", len(filtered_optTol))
    
    report = filtered_optTol.groupby("sizeClass")["scientificName"].count().to_frame("n_species")
    report["removed"] = (
        optTol.groupby("sizeClass")["scientificName"].count() - report["n_species"]
    ).fillna(0).astype(int)
    
    print("\nSummary by size class:\n", report)
    
    print(f"\nCreated file: {out_filtered_file}")

    return out_filtered_file


out_filtered_file_axis1 = eliminate_outliers(optTol_file_axis1, 1)

out_filtered_file_axis2 = eliminate_outliers(optTol_file_axis2, 2)

file_out_filtered_file_axis1 = open("/tmp/out_filtered_file_axis1_" + id + ".json", "w")
file_out_filtered_file_axis1.write(json.dumps(out_filtered_file_axis1))
file_out_filtered_file_axis1.close()
file_out_filtered_file_axis2 = open("/tmp/out_filtered_file_axis2_" + id + ".json", "w")
file_out_filtered_file_axis2.write(json.dumps(out_filtered_file_axis2))
file_out_filtered_file_axis2.close()
