import pandas as pd
import os
import numpy as np
import math
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


conf_output_path = conf_output_path = '/tmp/data/WF6_0/' + 'output'

pd.options.mode.chained_assignment = None  # disable chained assignment warning

output_dir  = conf_output_path
datain      = input_file
density_file   = os.path.join(output_dir, 'traits_density.csv')

os.makedirs(output_dir, exist_ok=True)

CountingStrategy = 'density_default'

def ensure_dir(path: str) -> None:
    print("Create the folder", path)
    os.makedirs(path, exist_ok=True)

def read_csv_safe(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=';', encoding='utf-8', low_memory=False)
    print(f"Letto: {os.path.basename(filepath)} | righe={len(df)} col={len(df.columns)}")
    return df

def normalize_dataframe_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.lower() for col in df.columns]
    
    rename_map = {
        "volumeofsedimentationchamber": "settlingvolume",
        "transectcounting": "numberofrandomfields"
    }
    
    df.rename(columns=rename_map, inplace=True)
    return df

def to_num(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def add_missing_cols(df: pd.DataFrame, defaults: dict) -> pd.DataFrame:
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v
    return df

def area_from_diameter(d):
    return (np.square(d / 2.0)) * math.pi



def compute_density_default(df: pd.DataFrame) -> pd.Series:
    """
    'density_default' strategy:
    If both 'settlingvolume' and 'numberofrandomfields' are present,
    use volume classes with the provided coefficients.
    dens = organismquantity / numberofrandomfields * 1000 / coeff
    
    classes (mL): <=5, (5,10], (10,25], (25,50], >50
    coeff: 0.001979 | 0.00365 | 0.010555 | 0.021703 | 0.041598
    """
    required = ['organismquantity', 'numberofrandomfields', 'settlingvolume']
    df = add_missing_cols(df, {c: np.nan for c in required})
    df = to_num(df, required)

    vol = df['settlingvolume']
    trn = df['numberofrandomfields']
    qta = df['organismquantity']

    conds = [
        vol.le(5) & vol.notna() & trn.notna(),
        vol.gt(5) & vol.le(10) & trn.notna(),
        vol.gt(10) & vol.le(25) & trn.notna(),
        vol.gt(25) & vol.le(50) & trn.notna(),
        vol.gt(50) & trn.notna()
    ]
    coeffs = [0.001979, 0.00365, 0.010555, 0.021703, 0.041598]

    with np.errstate(divide='ignore', invalid='ignore'):
        base = (qta / trn) * 1000.0
        dens = np.select(conds, [base / c for c in coeffs], default=np.nan)

    dens = pd.Series(dens, index=df.index, dtype='float64')
    dens = dens.replace([np.inf, -np.inf], np.nan)
    return dens.round(2)
    
def compute_density1(df: pd.DataFrame) -> pd.Series:
    """
    Counting for random fields:
    density1 = organismquantity * 1000 * area_chamber / (numberofrandomfields * area_field * settlingvolume)
    """
    required = ['organismquantity', 'diameterofsedimentationchamber',
                'numberofrandomfields', 'diameterFOV', 'settlingvolume']
    df = add_missing_cols(df, {c: np.nan for c in required})
    df = to_num(df, required)

    area_chamber = area_from_diameter(df['diameterofsedimentationchamber'])
    area_field   = area_from_diameter(df['diameterFOV'])

    denom = df['numberofrandomfields'] * area_field * df['settlingvolume']
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = df['organismquantity'] * 1000.0 * area_chamber / denom

    dens = dens.replace([np.inf, -np.inf], np.nan)
    return dens.round(2)
    
def compute_density2(df: pd.DataFrame) -> pd.Series:
    """
    Counting per transects/Counting per diametral transects :
    density2 = ((organismquantity / numberoftransects) * (π/4) * (diameterofsedimentationchamber / diameterFOV)) * 1000 / settlingvolume
    """
    required = ['organismquantity', 'numberoftransects', 'diameterofsedimentationchamber',
                'diameterFOV', 'settlingvolume']
    df = add_missing_cols(df, {c: np.nan for c in required})
    df = to_num(df, required)

    with np.errstate(divide='ignore', invalid='ignore'):
        factor = (math.pi / 4.0) * (df['diameterofsedimentationchamber'] / df['diameterFOV'])
        dens = (df['organismquantity'] / df['numberoftransects']) * factor
        dens = dens * 1000.0 / df['settlingvolume']

    dens = dens.replace([np.inf, -np.inf], np.nan)
    return dens.round(2)
    
def compute_density3(df: pd.DataFrame) -> pd.Series:
    """
    Counting whole chamber:
    density3 = organismquantity * 1000 / settlingvolume
    """
    required = ['organismquantity', 'settlingvolume']
    df = add_missing_cols(df, {c: np.nan for c in required})
    df = to_num(df, required)

    with np.errstate(divide='ignore', invalid='ignore'):
        dens = df['organismquantity'] * 1000.0 / df['settlingvolume']

    dens = dens.replace([np.inf, -np.inf], np.nan)
    return dens.round(2)

if __name__ == "__main__":
    start_time = time.time()
    ensure_dir(output_dir)

    df_datain = read_csv_safe(datain)
    df_datain = normalize_dataframe_headers(df_datain)

    df_merged = df_datain.copy()
    df_merged = add_missing_cols(df_merged, {
        'dilutionfactor': 1.0,
        'transectcounting': np.nan,                # for compatibility
        'numberofrandomfields': np.nan,
        'numberoftransects': np.nan,
        'diameterofsedimentationchamber': np.nan,
        'diameterFOV': np.nan,
        'settlingvolume': np.nan,
        'organismquantity': np.nan
    })
    df_merged = to_num(df_merged, [
        'dilutionfactor','numberofcountedfields','numberoftransects',
        'diameterofsedimentationchamber','diameteroffieldofview','settlingvolume',
        'organismquantity'
    ])

    if 'density' in df_datain.columns:
        df_datain = df_datain.drop(columns=['density'])

    df_merged['density'] = np.nan

    if CountingStrategy in ('density_default', 'density0'):
        df_merged['density'] = compute_density_default(df_merged)

    elif CountingStrategy == 'density1':  # counts per random field
        df_merged['density'] = compute_density1(df_merged)

    elif CountingStrategy == 'density2':  # counts per diameter transects
        df_merged['density'] = compute_density2(df_merged)

    elif CountingStrategy == 'density3':  # whole chamber
        df_merged['density'] = compute_density3(df_merged)

    else:
        raise ValueError(
            "CountingStrategy not recognized. Usa 'density_default' (o 'density0') | 'density1' | 'density2' | 'density3'."
        )

    if 'factor' in df_merged.columns:
        df_datain['density'] = (df_merged['density'] / df_merged['factor']).round(2)
    else:
        df_datain['density'] = df_merged['density']


    df_datain.to_csv(density_file, index=False, encoding='utf-8', sep=';')
    print(f"Output file saved in: {density_file}")

    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f} s")

file_density_file = open("/tmp/density_file_" + id + ".json", "w")
file_density_file.write(json.dumps(density_file))
file_density_file.close()
