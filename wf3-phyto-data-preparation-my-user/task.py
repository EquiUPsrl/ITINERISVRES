import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--input_abiotic_csv', action='store', type=str, required=True, dest='input_abiotic_csv')

arg_parser.add_argument('--input_biotic_csv', action='store', type=str, required=True, dest='input_biotic_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

input_abiotic_csv = args.input_abiotic_csv.replace('"','')
input_biotic_csv = args.input_biotic_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF3/' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)


biotic_file = input_biotic_csv
abiotic_file = input_abiotic_csv

df_abio = pd.read_csv(abiotic_file, sep=";",encoding="utf-8-sig")
df_bio = pd.read_csv(biotic_file, sep=";",encoding="utf-8-sig")


df_abio['ID'] = df_abio['cruise'].astype(str) + "_" + df_abio['station'].astype(str) + "_" + df_abio['depth'].astype(str)
df_abio.set_index('ID', inplace=True)
cols_to_keep = ['depth', 'Silicate', 'DIP', 'Temperature', 'DO', 'DIN'] 
abio = df_abio[cols_to_keep]

abio_file = os.path.join(output_dir, "abio.csv")
abio.to_csv(abio_file, encoding="utf-8-sig")
print("Abiotic data saved to: " + abio_file)

df_bio["ID"] = (
    df_bio["cruise"].astype(str) + "_" +
    df_bio["station"].astype(str) + "_" +
    df_bio["depth"].astype(str)
)

bio = df_bio.pivot_table(
    index="ID",
    columns="scientificName", # use scientificName or species_full
    values="density",   
    aggfunc="sum",
    fill_value=0
)

df_bio_file = os.path.join(output_dir, "df_bio.csv")
df_bio.to_csv(df_bio_file, encoding="utf-8-sig")
print("Biotic data saved to: " + df_bio_file)

bio_file = os.path.join(output_dir, "bio.csv")
bio.to_csv(bio_file, encoding="utf-8-sig")
print("Biotic pivot table saved to: " + bio_file)

file_abio_file = open("/tmp/abio_file_" + id + ".json", "w")
file_abio_file.write(json.dumps(abio_file))
file_abio_file.close()
file_bio_file = open("/tmp/bio_file_" + id + ".json", "w")
file_bio_file.write(json.dumps(bio_file))
file_bio_file.close()
