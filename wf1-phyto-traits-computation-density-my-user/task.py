import pandas as pd
import os
import math
import numpy as np
import time

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--input_file', action='store', type=str, required=True, dest='input_file')


args = arg_parser.parse_args()
print(args)

id = args.id

input_file = args.input_file.replace('"','')


conf_input_path = conf_input_path = '/tmp/data/WF1/' + 'data'
conf_output_path = conf_output_path = '/tmp/data/WF1/' + 'output'

pd.options.mode.chained_assignment = None  # disable warning chained assignment

input_dir   = conf_input_path
output_dir  = conf_output_path
datain      = input_file
density_file   = os.path.join(output_dir, 'final_input_density.csv')


def ensure_dir(path: str) -> None:
    print("Create folder: ", path)
    os.makedirs(path, exist_ok=True)

def read_csv_safe(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=';', encoding='utf-8', low_memory=False)
    print(f"Read csv: {os.path.basename(filepath)} | rows={len(df)} col={len(df.columns)}")
    return df

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set all labels to lowercase, with no spaces.
    Ex: 'organismQuantity' -> 'organismquantity'
        'Settling Volume'  -> 'settlingvolume'
        'd_chamber'        -> 'd_chamber'
    """
    df.columns = df.columns.str.lower().str.replace(' ', '')
    return df

def to_num(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def area(d):
    return (d / 2.0) ** 2 * math.pi


def density_random_fields(df: pd.DataFrame) -> pd.Series:
    """
    Counting for random fields:
    density1 = organismquantity * 1000 * area_chamber / (numberofrandomfields * area_field * settlingvolume)
    """
    req = ['organismquantity', 'diameterofsedimentationchamber',
           'diameteroffieldofview', 'numberofrandomfields', 'settlingvolume']
    df = to_num(df, req)

    area_chamber = area(df['diameterofsedimentationchamber'])
    area_field   = area(df['diameteroffieldofview'])

    denom = df['numberofrandomfields'] * area_field * df['settlingvolume']
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = df['organismquantity'] * 1000.0 * area_chamber / denom

    dens = dens.replace([np.inf, -np.inf], np.nan)
    return dens.round(2)

def density_transects(df: pd.DataFrame) -> pd.Series:
    """
    Counting by diameter transects:
    density2 = ((organismquantity / numberoftransects) * (π/4) *
                (diameterofsedimentationchamber / diameteroffieldofview)) * 1000 / settlingvolume
    """
    req = ['organismquantity', 'numberoftransects',
           'diameterofsedimentationchamber', 'diameteroffieldofview',
           'settlingvolume']
    df = to_num(df, req)

    with np.errstate(divide='ignore', invalid='ignore'):
        factor = (math.pi / 4.0) * (df['diameterofsedimentationchamber'] /
                                    df['diameteroffieldofview'])
        dens = (df['organismquantity'] / df['numberoftransects']) * factor
        dens = dens * 1000.0 / df['settlingvolume']

    dens = dens.replace([np.inf, -np.inf], np.nan)
    return dens.round(2)

def density_whole_chamber(df: pd.DataFrame) -> pd.Series:
    """
    Counting by whole chamber:
    density3 = organismquantity * 1000 / settlingvolume
    """
    req = ['organismquantity', 'settlingvolume']
    df = to_num(df, req)

    with np.errstate(divide='ignore', invalid='ignore'):
        dens = df['organismquantity'] * 1000.0 / df['settlingvolume']

    dens = dens.replace([np.inf, -np.inf], np.nan)
    return dens.round(2)

if __name__ == "__main__":
    start = time.time()
    ensure_dir(output_dir)

    df = read_csv_safe(datain)

    df = normalize_headers(df)

    for col in ['organismquantity', 'settlingvolume',
                'diameterofsedimentationchamber', 'diameteroffieldofview',
                'numberofrandomfields', 'numberoftransects',
                'factor']:
        if col not in df.columns:
            df[col] = np.nan

    df = to_num(df, [
        'organismquantity', 'settlingvolume',
        'diameterofsedimentationchamber', 'diameteroffieldofview',
        'numberofrandomfields', 'numberoftransects',
        'factor'
    ])

    mask_fields = (
        df['numberofrandomfields'].notna() &
        (df['numberofrandomfields'] > 0) &
        df['organismquantity'].notna() &
        df['diameterofsedimentationchamber'].notna() &
        df['diameteroffieldofview'].notna() &
        df['settlingvolume'].notna()
    )
    mask_tran   = df['numberoftransects'].notna() & (df['numberoftransects'] > 0)
    mask_whole  = ~mask_fields & ~mask_tran

    print("random fields rows:", int(mask_fields.sum()))
    print("transects rows    :", int(mask_tran.sum()))
    print("whole chamber rows:", int(mask_whole.sum()))

    dens = pd.Series(np.nan, index=df.index, dtype='float64')
    methods = pd.Series(np.nan, index=df.index, dtype='object')

    if mask_fields.any():
        dens_rf = density_random_fields(df.loc[mask_fields])
        dens.loc[mask_fields] = dens_rf
        methods.loc[mask_fields] = 'random_fields'

    if mask_tran.any():
        dens_tr = density_transects(df.loc[mask_tran])
        dens.loc[mask_tran] = dens_tr
        methods.loc[mask_tran] = 'transects'

    if mask_whole.any():
        dens_wc = density_whole_chamber(df.loc[mask_whole])
        dens.loc[mask_whole] = dens_wc
        methods.loc[mask_whole] = 'whole_chamber'

    if 'factor' in df.columns:
        dens = dens / df['factor'].fillna(1)

    df['density'] = dens.round(2)
    df['methods'] = methods

    df.to_csv(density_file, sep=';', index=False, encoding='utf-8')
    print("Outpt file saved in:", density_file)
    print("Execution time: %.2f s" % (time.time() - start))

file_density_file = open("/tmp/density_file_" + id + ".json", "w")
file_density_file.write(json.dumps(density_file))
file_density_file.close()
