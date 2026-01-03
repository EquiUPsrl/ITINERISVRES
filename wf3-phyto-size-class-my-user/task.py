import os
import pandas as pd
import numpy as np

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--input_biotic_csv', action='store', type=str, required=True, dest='input_biotic_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

input_biotic_csv = args.input_biotic_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

biotic_file = input_biotic_csv

df = pd.read_csv(biotic_file, sep=";", encoding="utf-8-sig")

df["sizeClass"] = np.round(np.log(df["meanBiomass"]), 0)

biotic_size_class = os.path.join(output_dir, "bio_sizeclass.csv")
df.to_csv(biotic_size_class, index=False, encoding="utf-8-sig")
print("Biotic Size Class saved to: " + biotic_size_class)

file_biotic_size_class = open("/tmp/biotic_size_class_" + id + ".json", "w")
file_biotic_size_class.write(json.dumps(biotic_size_class))
file_biotic_size_class.close()
