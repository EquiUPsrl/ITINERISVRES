import os
import pandas as pd
import numpy as np
import math
import re

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--density_file', action='store', type=str, required=True, dest='density_file')

arg_parser.add_argument('--input_formulas', action='store', type=str, required=True, dest='input_formulas')


args = arg_parser.parse_args()
print(args)

id = args.id

density_file = args.density_file.replace('"','')
input_formulas = args.input_formulas.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF1/' + 'output'

output_dir = conf_output_path
os.makedirs(output_dir, exist_ok=True)

density_file   = density_file
formulas_file = input_formulas
traits_file   = os.path.join(output_dir, "final_input.csv")


def convert_formula(f):
    """Converts ^ to ** and leaves pi unchanged."""
    if pd.isna(f) or f == "":
        return None
    return str(f).replace("^", "**")

def safe_eval_row(formula_text, row):
    """Evaluate a formula on a single line safely."""
    if formula_text is None or pd.isna(formula_text):
        return np.nan
    f = convert_formula(formula_text)
    env = row.to_dict()

    allowed = {
        "pi": math.pi,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "arcsin": math.asin,
        "log": math.log,
        "abs": abs,
    }

    try:
        val = eval(f, {"__builtins__": None, **allowed}, env)
        return float(val)
    except:
        return np.nan

def normalize_shape(s):
    if pd.isna(s):
        return s
    s = s.lower().strip()          # convert to lowercase and remove leading/trailing spaces
    s = re.sub(r'\s+', ' ', s)    # replace multiple spaces with a single space
    return s


dati = pd.read_csv(density_file, sep=";")
formule = pd.read_csv(formulas_file, sep=None, engine="python", encoding="latin1")

dati.columns    = dati.columns.str.lower().str.strip()
formule.columns = formule.columns.str.lower().str.strip()

if "shape" not in formule.columns and "Shape" in formule.columns:
    formule = formule.rename(columns={"Shape": "shape"})

dati['shape_norm'] = dati['shape'].apply(normalize_shape)
formule['shape_norm'] = formule['shape'].apply(normalize_shape)

df = pd.merge(dati, formule, how='left', left_on='shape_norm', right_on='shape_norm')

df = df.drop(columns=['shape_norm'])

if 'shape' not in df.columns:
    if 'shape' in dati.columns:
        df['shape'] = dati['shape']
    elif 'shape' in formule.columns:
        df['shape'] = formule['shape']
    else:
        df['shape'] = np.nan







if all(col in df.columns for col in ['width', 'length']):
    print("Selected formula: simplified")
    formula_biovolume = "formulaforbiovolumesimplified"
    formula_surfacearea = "formulaforsurfacesimplified"
else:
    print("Selected formula: advanced")
    formula_biovolume = "formulaforbiovolume"
    formula_surfacearea = "formulaforsurface"

df["biovolume"] = df.apply(
    lambda r: safe_eval_row(r.get(formula_biovolume, np.nan), r), axis=1
)

df["surfacearea"] = df.apply(
    lambda r: safe_eval_row(r.get(formula_surfacearea, np.nan), r), axis=1
)


diatom_classes = [
    "bacillariophyceae",
    "mediophyceae",
    "coscinodiscophyceae",
    "fragilariophyceae",
    "araphidophyceae",
    "rhizosoleniophyceae",
]

df["is_diatom"] = df["class"].isin(diatom_classes)


bv = df["biovolume"]

df["carboncontent"] = np.select(
    [
        bv < 3000,
        (bv >= 3000) & df["is_diatom"],
        (bv >= 3000) & (~df["is_diatom"]),
    ],
    [
        0.26  * (bv ** 0.86),
        0.288 * (bv ** 0.811),
        0.216 * (bv ** 0.939),
    ],
    default=np.nan
)


df["biomass"] = df["biovolume"] * 1.03 / 1000.0


df["totalbiovolume"]     = df["density"] * df["biovolume"]
df["totalcarboncontent"] = df["density"] * df["carboncontent"]
df["totalbiomass"]       = df["density"] * df["biomass"]


df["surfacevolumeratio"] = df["surfacearea"] / df["biovolume"]
df["surfacevolumeratio"] = df["surfacevolumeratio"].replace([np.inf, -np.inf], np.nan)


numeric_cols = [
    "biovolume", "surfacearea", "surfacevolumeratio",
    "carboncontent", "biomass",
    "totalbiovolume", "totalcarboncontent", "totalbiomass",
    "density"
]

for c in numeric_cols:
    if c in df.columns:
        df[c] = df[c].round(3)


cols_drop = [
    "thesaurus_uri",
    "formulaformissingdimension",
    "hd",
    "formulaforbiovolume",
    "formulaforsurface",
    "missingdimensionsimplified",
    "formulaformissingdimensionsimplified",
    "formulaforbiovolumesimplified",
    "formulaforsurfacesimplified",
    "missingdimension",
    "reference",
    "mandatory_linear_dimension_simplified",
    "mandatory_linear_dimension_advanced",
]

df = df.drop(columns=[c for c in cols_drop if c in df.columns], errors="ignore")


final_order = [
    "country","countrycode","waterbody","survey","locality","parenteventid",
    "replicate","treatment","day","month","year","season",
    "scientificname","kingdom","phylum","class","order","family","genus",
    "subgenus","species","subspecies","aphiaid_accepted","scientificname_accepted",
    "authority_accepted","shape","organismquantity","settlingvolume",
    "diameterofsedimentationchamber","diameteroffieldofview",
    "numberofrandomfields","numberoftransects","factor",
    "length","width","c",
    "density","biovolume","surfacearea","surfacevolumeratio",
    "carboncontent","biomass","totalbiovolume",
    "totalcarboncontent","totalbiomass","is_diatom"
]

final_cols = [c for c in final_order if c in df.columns]

df = df[final_cols]

df.to_csv(traits_file, sep=";", decimal=".", index=False)
print("File written in:", traits_file)

file_traits_file = open("/tmp/traits_file_" + id + ".json", "w")
file_traits_file.write(json.dumps(traits_file))
file_traits_file.close()
