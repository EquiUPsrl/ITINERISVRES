import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--aggregated_file', action='store', type=str, required=True, dest='aggregated_file')


args = arg_parser.parse_args()
print(args)

id = args.id

aggregated_file = args.aggregated_file.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF5/' + 'data'

INPUT_FILE = aggregated_file  # input CSV file
abundance_col = "density"                # abundance column
biomass_col = "biovolume"                # biomass column
thresholds = [1, 5, 10, 25]      # default cumulative contribution thresholds (percent)
selected_threshold = 5                     # user-selected threshold for downstream analysis
OUTPUT_DIR = os.path.join(conf_output_path, "filtered_data")  # folder to save filtered dataset
os.makedirs(OUTPUT_DIR, exist_ok=True)

filtered_file = aggregated_file


action_name = "data_filtering"
config_file = os.path.join(conf_input_path, "config.csv")
run_action = False
if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]
    
    p = act.set_index("parameter")["value"]

    run_action = p.get("active").lower() == "true"

if not run_action:
    print(f"Action '{action_name}' is disabled or config file missing. Cell skipped.")
else:
    
    abundance_col = p.get("abundance_col", abundance_col)
    biomass_col = p.get("biomass_col", biomass_col)
    selected_threshold = float(p.get("selected_threshold", selected_threshold))

    if "thresholds" in p:
        thresholds = [float(x) for x in p["thresholds"].split(",")]

    print("abundance_col", abundance_col)
    print("biomass_col", biomass_col)
    print("thresholds", thresholds)
    print("selected_threshold", selected_threshold)

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print("Input data loaded:", df.shape)
    
    for thresh in thresholds:
        df_sorted = df.sort_values(abundance_col, ascending=False).copy()
        
        df_sorted["cumulative_perc_ab"] = 100 * df_sorted[abundance_col].cumsum() / df_sorted[abundance_col].sum()
        
        df_sorted["cumulative_perc_bio"] = 100 * df_sorted[biomass_col].cumsum() / df_sorted[biomass_col].sum()
        
        df_filtered = df_sorted[
            (df_sorted["cumulative_perc_ab"] <= (100 - thresh)) &
            (df_sorted["cumulative_perc_bio"] <= (100 - thresh))
        ].drop(columns=["cumulative_perc_ab", "cumulative_perc_bio"])
        
        out_file = os.path.join(OUTPUT_DIR, f"filtered_{thresh}perc.csv")
        df_filtered.to_csv(out_file, index=False)
        print(f"Saved filtered dataset ({thresh}% cumulative contribution):", out_file)
    
    selected_file = os.path.join(OUTPUT_DIR, f"filtered_{selected_threshold}perc.csv")
    print(f"\nSelected file for downstream analysis (selected_threshold = {selected_threshold}%):")
    print(selected_file)
    filtered_file = selected_file

file_filtered_file = open("/tmp/filtered_file_" + id + ".json", "w")
file_filtered_file.write(json.dumps(filtered_file))
file_filtered_file.close()
