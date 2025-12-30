import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--final_input', action='store', type=str, required=True, dest='final_input')


args = arg_parser.parse_args()
print(args)

id = args.id

final_input = args.final_input.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF6/' + 'output'
conf_input_path = conf_input_path = '/tmp/data/WF6/' + 'data'

plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.grid'] = True

sns.set(style="whitegrid", context="talk")

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)
print(f"Folder '{output_dir}' ready.")

csv_path = final_input
event_df = pd.read_csv(csv_path, low_memory=False)

abiotic_cols = [
    'alcalinity', 'ammonium', 'nitrate', 'nitrite', 'totalNitrogen', 'calcium',
    'dissolvedOrganicCarbon', 'conductivity', 'totalPhosphorous',
    'orthophosphate', 'dissolvedOxygen', 'ph', 'depth', 'reactiveSilica', 'TDS',
    'waterTemperature', 'airTemperature', 'transparency'
]


action_name = "correlation_analysis"
config_file = os.path.join(conf_input_path, "config.csv")

if os.path.exists(config_file):
    cfg = pd.read_csv(config_file, sep=";")
    act = cfg[cfg["action"] == action_name]
    p = act.set_index("parameter")["value"]
    
    value = p.get("abiotic_cols")

    if isinstance(value, str) and value.strip():
        abiotic_cols = [x.strip() for x in value.split(",")]


print("Abiotic variables used:", abiotic_cols)


corr_df = event_df[abiotic_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, cmap='viridis', annot=False)
plt.title("")
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "heatmap_abiotici.svg"), format="svg", dpi=300)
plt.savefig(os.path.join(output_dir, "heatmap_abiotici.jpg"), format="jpg", dpi=300)
print(f"Heatmap saved to {os.path.join(output_dir, 'heatmap_abiotici.svg')} and .jpg")
plt.show()

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
