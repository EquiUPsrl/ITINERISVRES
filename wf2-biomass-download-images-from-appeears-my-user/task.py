import requests
import os
import pandas as pd
import time

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--appeears_raster_csv', action='store', type=str, required=True, dest='appeears_raster_csv')

arg_parser.add_argument('--param_appeears_password', action='store', type=str, required=True, dest='param_appeears_password')
arg_parser.add_argument('--param_appeears_username', action='store', type=str, required=True, dest='param_appeears_username')

args = arg_parser.parse_args()
print(args)

id = args.id

appeears_raster_csv = args.appeears_raster_csv.replace('"','')

param_appeears_password = args.param_appeears_password.replace('"','')
param_appeears_username = args.param_appeears_username.replace('"','')

conf_tmp_path = conf_tmp_path = '/tmp/data/WF2/work/' + 'tmp'

USERNAME = param_appeears_username
PASSWORD = param_appeears_password

response = requests.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=(USERNAME, PASSWORD))
res = response.json()

TOKEN = res['token']

print(TOKEN)

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

tmp_dir = conf_tmp_path

appeears_path = os.path.join(tmp_dir, "appeears_downloads")
appeears_qc_path = os.path.join(tmp_dir, "appeears_downloads_qc")
os.makedirs(appeears_path, exist_ok=True)
os.makedirs(appeears_qc_path, exist_ok=True)



def get_dir(file_name, selected_vars):
    variabile_trovata = ''
    for var in selected_vars:
        if var in file_name:
            variabile_trovata = var
            break

    return variabile_trovata


def download_geotiff_files(task_id, variable, qc, appeears_path, appeears_qc_path, max_retries=10, delay=30):

    bundle_url = f"https://appeears.earthdatacloud.nasa.gov/api/bundle/{task_id}"

    for attempt in range(1, max_retries + 1):
        print(f"[{attempt}/{max_retries}] Check file availability for task {task_id}...")
        response = requests.get(bundle_url, headers=HEADERS)

        try:
            data = response.json()
        except ValueError:
            print("[!] Invalid JSON response, retrying...")
            time.sleep(delay)
            continue

        if "files" not in data:
            print("[!] 'files' key not found. Waiting before retrying...")
            time.sleep(delay)
            continue

        files = data["files"]
        for file in files:
            if file["file_type"] != "tif":
                continue  # ignoriamo altri tipi di file

            filename = os.path.basename(file['file_name'])
            is_qc = "_qc_" in filename.lower()
            target_var = qc if is_qc else variable

            if target_var.lower() not in filename.lower():
                continue  # salta i file che non corrispondono

            folder_path = os.path.join(appeears_qc_path if is_qc else appeears_path, target_var)
            os.makedirs(folder_path, exist_ok=True)

            output_path = os.path.join(folder_path, filename)
            if os.path.isfile(output_path):
                print(f"[✔] File already exists: {output_path}")
                continue

            download_url = f"https://appeears.earthdatacloud.nasa.gov/api/bundle/{task_id}/{file['file_id']}/{filename}"
            print(f"[↓] Downloading: {filename}")
            r = requests.get(download_url, headers=HEADERS, stream=True)
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[✔] Download completed: {output_path}")

        return  # esce dopo aver processato i file disponibili

    print(f"[✘] Error: files not available for task {task_id} after {max_retries} attempts.")







df = pd.read_csv(appeears_raster_csv, sep=';')


df_with_task = df[df["TaskID"].notna() & (df["TaskID"] != "")]

for _, row in df_with_task.iterrows():
    variable = row["Variable"]
    qc = row["QC"]
    task_id = row["TaskID"]

    print(f"\n=== Processing task: {task_id} | Variable: {variable} | QC: {qc} ===")
    download_geotiff_files(task_id, variable, qc, appeears_path, appeears_qc_path)

file_appeears_path = open("/tmp/appeears_path_" + id + ".json", "w")
file_appeears_path.write(json.dumps(appeears_path))
file_appeears_path.close()
file_appeears_qc_path = open("/tmp/appeears_qc_path_" + id + ".json", "w")
file_appeears_qc_path.write(json.dumps(appeears_qc_path))
file_appeears_qc_path.close()
