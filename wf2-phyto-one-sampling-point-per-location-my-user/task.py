import pandas as pd
import os
import re
from functools import reduce
import operator

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--input_abiotic', action='store', type=str, required=True, dest='input_abiotic')

arg_parser.add_argument('--input_biotic', action='store', type=str, required=True, dest='input_biotic')


args = arg_parser.parse_args()
print(args)

id = args.id

input_abiotic = args.input_abiotic.replace('"','')
input_biotic = args.input_biotic.replace('"','')


conf_output_path = conf_output_path = '' + 'output'
conf_input_path = conf_input_path = '' + 'data'

out_dir = conf_output_path
os.makedirs(out_dir, exist_ok=True)

def filter_by_location(data_df: pd.DataFrame, config_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a dataframe based on locality and locationID combinations specified in a config dataframe.
    If no config is provided, automatically determine the pairs using the first occurrence per locality.
    
    Parameters:
        data_df (pd.DataFrame): The main dataframe to filter.
        config_df (pd.DataFrame): A dataframe with columns 'locality' and 'locationID' specifying allowed combinations.
    
    Returns:
        pd.DataFrame: Filtered dataframe including specified locations and all other locations not listed.
    """

    if config_df is not None and not config_df.empty:
        masks = []
        
        for _, row in config_df.iterrows():
            mask = (data_df["locality"] == row["locality"]) & (data_df["locationID"] == row["locationID"])
            masks.append(mask)
        
        mask_other = ~data_df["locality"].isin(config_df["locality"].tolist())
        
        if masks:
            combined_mask = reduce(operator.or_, masks) | mask_other
        else:
            combined_mask = mask_other
        
        return data_df[combined_mask].copy()
    else:
        first_pairs = data_df.drop_duplicates(subset=["locality"], keep="first")[["locality", "locationID"]]
        masks = [(data_df["locality"] == row["locality"]) & (data_df["locationID"] == row["locationID"])
                 for _, row in first_pairs.iterrows()]
        mask_other = ~data_df["locality"].isin(first_pairs["locality"].tolist())
        combined_mask = reduce(operator.or_, masks) | mask_other if masks else mask_other
        return data_df[combined_mask].copy()



config_file = os.path.join(conf_input_path, 'locations_config.csv')

if os.path.exists(config_file):
    location_config = pd.read_csv(config_file, sep=';')
else:
    location_config = None


phyto_bio = pd.read_csv(
    input_biotic,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

phyto_bio.columns = [re.sub(r'[^\x00-\x7F]+','', x) for x in phyto_bio.columns]


phyto_bio_filt = filter_by_location(phyto_bio, location_config)

print("Biotic Total Records:", phyto_bio.shape[0])
print("Record after filter (location_config.csv):", phyto_bio_filt.shape[0])
print("\nLocationID for lake after filter:")
print(phyto_bio_filt.groupby("locality")["locationID"].unique())

biotic_file = os.path.join(out_dir, "biotic_filtered.csv")

phyto_bio_filt.to_csv(biotic_file, sep=";", index=False)



phyto_abio = pd.read_csv(
    input_abiotic,
    sep=";",
    encoding="utf-8",
    low_memory=False
)

phyto_abio.columns = [re.sub(r'[^\x00-\x7F]+','', x) for x in phyto_abio.columns]


phyto_abio_filt = filter_by_location(phyto_abio, location_config)

print("Abiotic Total Records:", phyto_abio.shape[0])
print("Record after filter (location_config.csv):", phyto_abio_filt.shape[0])
print("\nLocationID for lake after filter:")
print(phyto_abio_filt.groupby("locality")["locationID"].unique())

abiotic_file = os.path.join(out_dir, "abiotic_filtered.csv")

phyto_abio_filt.to_csv(abiotic_file, sep=";", index=False)

file_abiotic_file = open("/tmp/abiotic_file_" + id + ".json", "w")
file_abiotic_file.write(json.dumps(abiotic_file))
file_abiotic_file.close()
file_biotic_file = open("/tmp/biotic_file_" + id + ".json", "w")
file_biotic_file.write(json.dumps(biotic_file))
file_biotic_file.close()
