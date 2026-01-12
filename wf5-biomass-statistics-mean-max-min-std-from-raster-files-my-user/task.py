from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
import rasterio

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--dataset_path', action='store', type=str, required=True, dest='dataset_path')

arg_parser.add_argument('--filtered_input', action='store', type=str, required=True, dest='filtered_input')


args = arg_parser.parse_args()
print(args)

id = args.id

dataset_path = args.dataset_path.replace('"','')
filtered_input = args.filtered_input.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF5/' + 'output'

def extract_product_date(filename):
    fname = filename.name if hasattr(filename, 'name') else filename
    
    m2 = re.search(r'(\d{8})_\d{8}.*?\.(\w+)[._]', fname)
    if m2:
        date_str = m2.group(1)
        prodotto = m2.group(2).upper()
        return prodotto, date_str

    m3 = re.search(r'^([a-z]+)\.(\d{7})_', fname, re.IGNORECASE)
    if m3:
        prodotto = m3.group(1).upper()
        date_str = m3.group(2)
        return prodotto, date_str

    m1 = re.search(r'^([a-z]+)[._].*?(\d{7})', fname, re.IGNORECASE)
    if m1:
        prodotto = m1.group(1).upper()
        date_str = m1.group(2)
        return prodotto, date_str

    return None, None

def extract_product_date_by_position(filename):
    fname = filename.name if hasattr(filename, 'name') else filename
    parts = fname.split('.')

    if len(parts) >= 5 and '_' in parts[1]:
        try:
            data = parts[1][:8]       
            prodotto = parts[4].upper()  
        except IndexError:
            return None, None
    elif len(parts) >= 2 and '_' in parts[1]:
        try:
            prodotto = parts[0].upper()   
            data = parts[1][:7]           
        except IndexError:
            return None, None
    elif len(parts) >= 3:
        try:
            prodotto = parts[0].upper()
            data = parts[2][:7]
        except IndexError:
            return None, None
    else:
        return None, None

    return prodotto, data


def calc_stats_from_rasters(
    input_csv: str,
    raster_folder: str,
):
    input_csv = Path(input_csv)
    raster_folder = Path(raster_folder)

    df = pd.read_csv(input_csv, delimiter=";")

    prodotti_map = {
        'VGPM': 'PNN',
        'CHL': 'CHL-a',
        'SST': 'SST',
        'POC': 'POC',
        'PIC': 'PIC',
        'PAR': 'PAR',
    }

    results = []

    for idx, row in df.iterrows():
        lat, lon = row['LATITUDE'], row['LONGITUDE']
        year = int(row['YEAR'])

        valori = {f"{v}_{stat}": np.nan for v in prodotti_map.values() for stat in ['MEAN', 'MAX', 'MIN', 'STD']}
        vals_by_prod = {v: [] for v in prodotti_map.values()}

        for file in raster_folder.glob("*.tif"):
            prodotto, date_str = extract_product_date_by_position(file)
            if prodotto is None or date_str is None:
                continue

            file_year = int(date_str[:4])
            if file_year != year:
                continue

            col_prefix = prodotti_map.get(prodotto)
            if not col_prefix:
                continue

            with rasterio.open(file) as src:
                try:
                    val = list(rasterio.sample.sample_gen(src, [(lon, lat)]))[0][0]
                    if val is not None and not np.isnan(val) and val >= 0:
                        vals_by_prod[col_prefix].append(val)
                except Exception as e:
                    print(f"⚠️ Errore su {file.name} punto ({lat}, {lon}): {e}")

        for col_prefix, vals in vals_by_prod.items():
            if vals:
                valori[f"{col_prefix}_MEAN"] = float(np.mean(vals))
                valori[f"{col_prefix}_MAX"] = float(np.max(vals))
                valori[f"{col_prefix}_MIN"] = float(np.min(vals))
                valori[f"{col_prefix}_STD"] = float(np.std(vals))

        res = {
            'ID': row.get('ID'),
            'STUDY_ID': row.get('STUDY_ID'),
            'YEAR': year,
            'LATITUDE': lat,
            'LONGITUDE': lon,
            'siteID': row.get('siteID'),
            'nSPECIES': row.get('nSPECIES'),
            'sumRAW_ABUN': row.get('sumRAW_ABUN', np.nan),
            'sumRAW_BIOM': row.get('sumRAW_BIOM', np.nan),
        }
        res.update(valori)
        results.append(res)

    df_out = pd.DataFrame(results)
    
    return df_out


output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

stats_file = os.path.join(output_dir, "stats_output.csv")
df_stats = calc_stats_from_rasters(filtered_input, dataset_path)

df_stats.to_csv(stats_file, sep=";", index=False)
print(f"✅ File saved in: {stats_file}")

file_stats_file = open("/tmp/stats_file_" + id + ".json", "w")
file_stats_file.write(json.dumps(stats_file))
file_stats_file.close()
