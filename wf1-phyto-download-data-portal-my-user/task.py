import pandas as pd
import os
import requests

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--dataportal_csv', action='store', type=str, required=True, dest='dataportal_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

dataportal_csv = args.dataportal_csv.replace('"','')


conf_input_path = conf_input_path = '/tmp/data/WF1/' + 'data'

def download_from_dataportal(url, output_file):
    
    if not isinstance(url, str):
        raise TypeError("Parameter 'url' must be a string")
    
    uuid = url.split("/bitstreams/")[1].split("/")[0]
    print("UUID:", uuid)
    
    if uuid != "":
        url = "https://data.lifewatchitaly.eu/server/api/core/bitstreams/" + uuid + "/content"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            print("File CSV salvato con successo.")
        else:
            print(f"Errore nel download: {response.status_code}")


input_file = dataportal_csv
df = pd.read_csv(input_file, sep=';', encoding='utf-8', low_memory=False)

download_dir = os.path.join(conf_input_path, 'download')
os.makedirs(download_dir, exist_ok=True)

for idx, (_, row) in enumerate(df.iterrows(), start=1):

    filename = row['file_name']
    if not filename.endswith(".csv"):
        filename = "file_" + idx + ".csv"
    
    print(row['url'], filename)
    download_from_dataportal(row['url'], os.path.join(download_dir, filename))

file_download_dir = open("/tmp/download_dir_" + id + ".json", "w")
file_download_dir.write(json.dumps(download_dir))
file_download_dir.close()
