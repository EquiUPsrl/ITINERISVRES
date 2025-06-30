import os
import time
import pandas as pd
import numpy as np

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--inputs', action='store', type=str, required=True, dest='inputs')

arg_parser.add_argument('--outputs', action='store', type=str, required=True, dest='outputs')


args = arg_parser.parse_args()
print(args)

id = args.id

inputs = args.inputs.replace('"','')
outputs = args.outputs.replace('"','')



pd.options.mode.chained_assignment = None  # no warnings for chained assignments

input_dir = 'data'
output_dir = 'output'

os.makedirs(output_dir, exist_ok=True)


datain = inputs['input_datain_file']
operator_file = inputs['input_operator_file']


output_file = os.path.join(output_dir, 'test_advanced.csv')

CalcType = 'advanced'
CompTraits = [
    'biovolume', 'surfacearea', 'cellcarboncontent', 'density',
    'totalbiovolume', 'surfacevolumeratio', 'totalcarboncontent'
]

def check_csv_structure(filepath):
    """Stampa le prime righe e le colonne del csv per verificarne la struttura."""
    try:
        df = pd.read_csv(filepath, sep=';', encoding='utf-8')
        print(f"\nVerifica struttura file: {os.path.basename(filepath)}")
        print("Colonne trovate:", df.columns.tolist())
        print(df.head(2))
        return df
    except Exception as e:
        print(f"Errore nella lettura del file {filepath}:\n{e}")
        return None

def calc_fun(datain, operator_file, CalcType, CompTraits, output_file):
    df_datain = pd.read_csv(datain, sep=';', encoding='utf-8')
    df_operator = pd.read_csv(operator_file, sep=';', encoding='utf-8')

    df_operator.columns = df_operator.columns.str.replace(' ', '')
    df_operator = df_operator.replace(['no', 'see note'], [np.nan, np.nan])
    if 'measurementremarks' in df_operator.columns:
        df_operator['measurementremarks'] = df_operator['measurementremarks'].str.lower()
    df_datain.columns = df_datain.columns.str.replace(' ', '')
    if 'measurementremarks' in df_datain.columns:
        df_datain['measurementremarks'] = df_datain['measurementremarks'].str.lower()

    df_merged = pd.merge(df_datain, df_operator, how='left', on=['scientificname','measurementremarks'])

    if CalcType == 'advanced':
        df_merged_concat = df_merged[df_merged['formulaformissingdimension'].isnull()]
        md_formulas = df_merged[df_merged['formulaformissingdimension'].notnull()]['formulaformissingdimension'].unique()
        for md_form in md_formulas:
            df_temp = df_merged[df_merged['formulaformissingdimension'] == md_form]
            for md in df_temp['missingdimension'].unique():
                df_temp_2 = df_temp[df_temp['missingdimension'] == md].copy()
                df_temp_2[md] = df_temp_2.eval(md_form)
                df_temp_2 = df_temp_2.round({md:2})
                df_merged_concat = pd.concat([df_merged_concat, df_temp_2])
        df_merged = df_merged_concat.sort_index()
    else:
        df_merged_concat = df_merged[df_merged['formulaformissingdimensionsimplified'].isnull()]
        md_formulas = df_merged[df_merged['formulaformissingdimensionsimplified'].notnull()]['formulaformissingdimensionsimplified'].unique()
        for md_form in md_formulas:
            df_temp = df_merged[df_merged['formulaformissingdimensionsimplified'] == md_form]
            for md in df_temp['missingdimensionsimplified'].unique():
                df_temp_2 = df_temp[df_temp['missingdimensionsimplified'] == md].copy()
                df_temp_2[md] = df_temp_2.eval(md_form)
                df_temp_2 = df_temp_2.round({md:2})
                df_merged_concat = pd.concat([df_merged_concat, df_temp_2])
        df_merged = df_merged_concat.sort_index()

    if 'biovolume' in CompTraits:
        if CalcType == 'advanced':
            df_datain['biovolume'] = np.nan
            df_merged_concat = df_merged[df_merged['formulaforbiovolume'].isnull()]
            bv_formulas = df_merged[df_merged['formulaforbiovolume'].notnull()]['formulaforbiovolume'].unique()
            for bv_form in bv_formulas:
                df_temp = df_merged[df_merged['formulaforbiovolume'] == bv_form].copy()
                bv_form = bv_form.replace('pi', '3.141592654').replace('^', '**').replace('asin', 'arcsin')
                df_temp['biovolume'] = df_temp.eval(bv_form)
                df_merged_concat = pd.concat([df_merged_concat, df_temp])
            df_merged = df_merged_concat.sort_index()
            df_merged['biovolume'] = np.round(df_merged['biovolume'], 2)
            df_datain['biovolume'] = df_merged['biovolume']
        else:
            df_datain['biovolume'] = np.nan
            df_merged_concat = df_merged[df_merged['formulaforbiovolumesimplified'].isnull()]
            bv_formulas = df_merged[df_merged['formulaforbiovolumesimplified'].notnull()]['formulaforbiovolumesimplified'].unique()
            for bv_form in bv_formulas:
                df_temp = df_merged[df_merged['formulaforbiovolumesimplified'] == bv_form].copy()
                bv_form = bv_form.replace('pi', '3.141592654').replace('^', '**').replace('asin', 'arcsin')
                df_temp['biovolume'] = df_temp.eval(bv_form)
                df_merged_concat = pd.concat([df_merged_concat, df_temp])
            df_merged = df_merged_concat.sort_index()
            df_merged['biovolume'] = np.round(df_merged['biovolume'], 2)
            df_datain['biovolume'] = df_merged['biovolume']

    if 'surfacearea' in CompTraits:
        if CalcType == 'advanced':
            df_datain['surfacearea'] = np.nan
            df_merged_concat = df_merged[df_merged['formulaforsurface'].isnull()]
            sa_formulas = df_merged[df_merged['formulaforsurface'].notnull()]['formulaforsurface'].unique()
            for sa_form in sa_formulas:
                df_temp = df_merged[df_merged['formulaforsurface'] == sa_form].copy()
                sa_form = sa_form.replace('pi', '3.141592653589793').replace('^', '**').replace('asin', 'arcsin')
                df_temp['surfacearea'] = df_temp.eval(sa_form)
                df_merged_concat = pd.concat([df_merged_concat, df_temp])
            df_merged = df_merged_concat.sort_index()
            df_merged['surfacearea'] = np.round(df_merged['surfacearea'], 2)
            df_datain['surfacearea'] = df_merged['surfacearea']
        else:
            df_datain['surfacearea'] = np.nan
            df_merged_concat = df_merged[df_merged['formulaforsurfacesimplified'].isnull()]
            sa_formulas = df_merged[df_merged['formulaforsurfacesimplified'].notnull()]['formulaforsurfacesimplified'].unique()
            for sa_form in sa_formulas:
                df_temp = df_merged[df_merged['formulaforsurfacesimplified'] == sa_form].copy()
                sa_form = sa_form.replace('pi', '3.141592653589793').replace('^', '**').replace('asin', 'arcsin')
                df_temp['surfacearea'] = df_temp.eval(sa_form)
                df_merged_concat = pd.concat([df_merged_concat, df_temp])
            df_merged = df_merged_concat.sort_index()
            df_merged['surfacearea'] = np.round(df_merged['surfacearea'], 2)
            df_datain['surfacearea'] = df_merged['surfacearea']

    if 'cellcarboncontent' in CompTraits:
        df_datain['cellcarboncontent'] = np.nan
        if 'biovolume' in CompTraits:
            df_merged_concat = df_merged[df_merged['biovolume'].isnull()]
            df_cc = df_merged[df_merged['biovolume'].notnull()]
            df_cc_1 = df_cc[df_cc['biovolume'] <= 3000]
            df_cc_2 = df_cc[df_cc['biovolume'] > 3000]
            cc_formulas_1 = df_merged[df_merged['formulaforweight1'].notnull()]['formulaforweight1'].unique()
            for cc_form in cc_formulas_1:
                df_temp = df_cc_1[df_cc_1['formulaforweight1'] == cc_form].copy()
                cc_form = cc_form.replace('^', '**').lower()
                df_temp['cellcarboncontent'] = df_temp.eval(cc_form)
                df_merged_concat = pd.concat([df_merged_concat, df_temp])
            cc_formulas_2 = df_merged[df_merged['formulaforweight2'].notnull()]['formulaforweight2'].unique()
            for cc_form in cc_formulas_2:
                df_temp = df_cc_2[df_cc_2['formulaforweight2'] == cc_form].copy()
                cc_form = cc_form.replace('^', '**').lower()
                df_temp['cellcarboncontent'] = df_temp.eval(cc_form)
                df_merged_concat = pd.concat([df_merged_concat, df_temp])
            df_merged = df_merged_concat.sort_index()
            df_merged['cellcarboncontent'] = np.round(df_merged['cellcarboncontent'], 2)
            df_datain['cellcarboncontent'] = df_merged['cellcarboncontent']

    if 'density' in CompTraits:
        df_datain['density'] = np.nan
        df_datain_concat = df_datain[df_datain['volumeofsedimentationchamber'].isnull() & df_datain['transectcounting'].isnull()]
        df_temp = df_datain[df_datain['volumeofsedimentationchamber'].notnull() & df_datain['transectcounting'].notnull()]
        df_temp_1 = df_temp[df_temp['volumeofsedimentationchamber'] <= 5].copy()
        df_temp_1['density'] = df_temp_1['organismquantity'] / df_temp_1['transectcounting'] * 1000 / 0.001979
        df_datain_concat = pd.concat([df_datain_concat, df_temp_1])
        df_temp_2 = df_temp[(df_temp['volumeofsedimentationchamber'] > 5) & (df_temp['volumeofsedimentationchamber'] <= 10)].copy()
        df_temp_2['density'] = df_temp_2['organismquantity'] / df_temp_2['transectcounting'] * 1000 / 0.00365
        df_datain_concat = pd.concat([df_datain_concat, df_temp_2])
        df_temp_3 = df_temp[(df_temp['volumeofsedimentationchamber'] > 10) & (df_temp['volumeofsedimentationchamber'] <= 25)].copy()
        df_temp_3['density'] = df_temp_3['organismquantity'] / df_temp_3['transectcounting'] * 1000 / 0.010555
        df_datain_concat = pd.concat([df_datain_concat, df_temp_3])
        df_temp_4 = df_temp[(df_temp['volumeofsedimentationchamber'] > 25) & (df_temp['volumeofsedimentationchamber'] <= 50)].copy()
        df_temp_4['density'] = df_temp_4['organismquantity'] / df_temp_4['transectcounting'] * 1000 / 0.021703
        df_datain_concat = pd.concat([df_datain_concat, df_temp_4])
        df_temp_5 = df_temp[df_temp['volumeofsedimentationchamber'] > 50].copy()
        df_temp_5['density'] = df_temp_5['organismquantity'] / df_temp_5['transectcounting'] * 1000 / 0.041598
        df_datain_concat = pd.concat([df_datain_concat, df_temp_5])
        df_datain = df_datain_concat.sort_index()
        df_datain['density'] = np.round(df_datain['density'], 2)

    if 'totalbiovolume' in CompTraits:
        if 'density' not in CompTraits:
            df_datain['density'] = np.nan
        if 'biovolume' not in CompTraits:
            df_datain['biovolume'] = np.nan
        df_datain['totalbiovolume'] = df_datain['density'] * df_datain['biovolume']
        df_datain['totalbiovolume'] = np.round(df_datain['totalbiovolume'], 2)

    if 'surfacevolumeratio' in CompTraits:
        if 'surfacearea' not in CompTraits:
            df_datain['surfacearea'] = np.nan
        if 'biovolume' not in CompTraits:
            df_datain['biovolume'] = np.nan
        df_datain['surfacevolumeratio'] = df_datain['surfacearea'] / df_datain['biovolume']
        df_datain['surfacevolumeratio'] = np.round(df_datain['surfacevolumeratio'], 2)

    if 'totalcarboncontent' in CompTraits:
        if 'density' not in CompTraits:
            df_datain['density'] = np.nan
        if 'cellcarboncontent' not in CompTraits:
            df_datain['cellcarboncontent'] = np.nan
        df_datain['totalcarboncontent'] = df_datain['density'] * df_datain['cellcarboncontent']
        df_datain['totalcarboncontent'] = np.round(df_datain['totalcarboncontent'], 2)

    for col in CompTraits:
        if col in df_datain.columns:
            df_datain[col] = df_datain[col].fillna('NA')

    df_datain.to_csv(output_file, index=False, encoding='utf-8', sep=';')
    print(f"File di output salvato in: {output_file}")

start_time = time.time()
calc_fun(datain, operator_file, CalcType, CompTraits, output_file)
end_time = time.time()
print(f"Tempo di esecuzione: {end_time - start_time:.2f} secondi")

outputs["result"] = output_file

