import pandas as pd
import os

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--param_datain', action='store', type=str, required=True, dest='param_datain')

args = arg_parser.parse_args()
print(args)

id = args.id


param_datain = args.param_datain.replace('"','')


filepath = f"/tmp/data/{param_datain}"

if not os.path.exists(filepath):
    raise FileNotFoundError(f"❌ File non trovato: {filepath}")

df = pd.read_csv(filepath)
print(f"✅ File letto: {filepath}")
print(df.head())

