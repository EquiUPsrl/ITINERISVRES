import os
import time
import pandas as pd
import numpy as np
import re

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--density_file', action='store', type=str, required=True, dest='density_file')

arg_parser.add_argument('--info_csv', action='store', type=str, required=True, dest='info_csv')


args = arg_parser.parse_args()
print(args)

id = args.id

density_file = args.density_file.replace('"','')
info_csv = args.info_csv.replace('"','')


conf_output_path = conf_output_path = '/tmp/data/WF6_0/' + 'output'

pd.options.mode.chained_assignment = None  # avoid warning chained assignment

output_dir = conf_output_path

os.makedirs(output_dir, exist_ok=True)

datain = density_file
operator_file = info_csv
traits_file = os.path.join(output_dir, 'traits_advanced.csv')

CalcType = 'advanced'
CompTraits = [
    'biovolume','surfacearea','cellcarboncontent','density',
    'totalbiovolume','surfacevolumeratio','totalcarboncontent','biomass','totalbiomass'
]

def prepare_columns_for_formula(df, formula):
    cols = list(set(re.findall(r'[a-zA-Z_]\w*', formula)))
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df, cols

def calc_fun(datain, operator_file, CalcType, CompTraits, traits_file):
    error_log = []
    fallback_log = []

    try:
        df_datain = pd.read_csv(datain, sep=';', encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df_datain = pd.read_csv(datain, sep=';', encoding='latin1', low_memory=False)
    df_operator = pd.read_csv(operator_file, sep=';', encoding='utf-8', low_memory=False)

    df_datain.columns  = df_datain.columns.str.strip().str.replace(' ', '').str.lower()
    df_operator.columns = df_operator.columns.str.strip().str.replace(' ', '').str.lower()

    df_in_orig = df_datain.copy()

    if 'measurementremarks' in df_datain.columns:
        df_datain['measurementremarks'] = df_datain['measurementremarks'].str.lower()
    if 'measurementremarks' in df_operator.columns:
        df_operator['measurementremarks'] = df_operator['measurementremarks'].str.lower()

    df_operator = df_operator.replace(['no','see note'], np.nan)

    on_keys = [k for k in ['scientificname','measurementremarks'] if k in df_datain.columns and k in df_operator.columns]
    df_merged = pd.merge(df_datain, df_operator, how='left', on=on_keys)

    if CalcType == 'advanced':
        df_merged_concat = df_merged[df_merged['formulaformissingdimension'].isnull()] if 'formulaformissingdimension' in df_merged.columns else df_merged.copy()
        md_forms = df_merged[df_merged.get('formulaformissingdimension').notnull()]['formulaformissingdimension'].unique() \
                   if 'formulaformissingdimension' in df_merged.columns else []
        for md_form in md_forms:
            df_temp_all = df_merged[df_merged['formulaformissingdimension'] == md_form]
            for md in df_temp_all['missingdimension'].dropna().unique():
                df_temp = df_temp_all[df_temp_all['missingdimension'] == md].copy()
                df_temp, _ = prepare_columns_for_formula(df_temp, md_form)
                try:
                    res = df_temp.eval(md_form)
                    if md in df_temp.columns:
                        df_temp.drop(columns=[md], inplace=True)
                    df_temp[md] = res.round(2)
                except Exception as e:
                    error_log.append({'formula': md_form, 'target': md, 'error': str(e),
                                      'scientificnames': df_temp['scientificname'].dropna().unique().tolist()})
                    continue
                df_merged_concat = pd.concat([df_merged_concat, df_temp])
        df_merged = df_merged_concat.sort_index()

    if 'biovolume' in CompTraits:
        df_datain['biovolume'] = np.nan
        col_formula = 'formulaforbiovolume' if CalcType == 'advanced' else 'formulaforbiovolumesimplified'
        df_merged_concat = df_merged[df_merged.get(col_formula).isnull()] if col_formula in df_merged.columns else df_merged.copy()
        bv_forms = df_merged[df_merged.get(col_formula).notnull()][col_formula].unique() if col_formula in df_merged.columns else []
        for bv_form in bv_forms:
            df_temp = df_merged[df_merged[col_formula] == bv_form].copy()
            bv_eval = bv_form.replace('pi', '3.141592654').replace('^', '**').replace('asin', 'arcsin')
            df_temp, _ = prepare_columns_for_formula(df_temp, bv_eval)
            try:
                df_temp['biovolume'] = df_temp.eval(bv_eval, engine='python')
            except Exception as e:
                error_log.append({'formula': bv_eval, 'target':'biovolume', 'error': str(e),
                                  'scientificnames': df_temp['scientificname'].dropna().unique().tolist()})
                continue
            df_merged_concat = pd.concat([df_merged_concat, df_temp])
        df_merged = df_merged_concat.sort_index()
        if 'biovolume' in df_merged.columns:
            df_merged['biovolume'] = df_merged['biovolume'].round(2)
            df_datain['biovolume'] = df_merged['biovolume']

    if 'surfacearea' in CompTraits and 'formulaforsurface' in df_merged.columns:
        df_datain['surfacearea'] = np.nan
        df_merged_concat = df_merged[df_merged['formulaforsurface'].isnull()]
        sa_forms = df_merged[df_merged['formulaforsurface'].notnull()]['formulaforsurface'].unique()
        for sa_form in sa_forms:
            df_temp = df_merged[df_merged['formulaforsurface'] == sa_form].copy()
            sa_eval = sa_form.replace('pi','3.141592653589793').replace('^','**').replace('asin','arcsin')
            df_temp, _ = prepare_columns_for_formula(df_temp, sa_eval)
            try:
                df_temp['surfacearea'] = df_temp.eval(sa_eval)
            except Exception as e:
                error_log.append({'formula': sa_eval, 'target':'surfacearea', 'error': str(e),
                                  'scientificnames': df_temp['scientificname'].dropna().unique().tolist()})
                continue
            df_merged_concat = pd.concat([df_merged_concat, df_temp])
        df_merged = df_merged_concat.sort_index()
        if 'surfacearea' in df_merged.columns:
            df_merged['surfacearea'] = df_merged['surfacearea'].round(2)
            df_datain['surfacearea'] = df_merged['surfacearea']

    if 'cellcarboncontent' in CompTraits:
        df_datain['cellcarboncontent'] = np.nan
        bv_series = pd.to_numeric(df_merged.get('biovolume'), errors='coerce')
        df_cc_1 = df_merged[bv_series.le(3000, fill_value=False)].copy()
        df_cc_2 = df_merged[bv_series.gt(3000, fill_value=False)].copy()

        if 'formulaforweight1' in df_merged.columns:
            forms1 = df_merged[df_merged['formulaforweight1'].notnull()]['formulaforweight1'].unique()
            for cc_form in forms1:
                df_temp = df_cc_1[df_cc_1['formulaforweight1'] == cc_form].copy()
                cc_eval = cc_form.replace('^','**').lower()
                df_temp, _ = prepare_columns_for_formula(df_temp, cc_eval)
                try:
                    df_temp['cellcarboncontent'] = df_temp.eval(cc_eval)
                except Exception as e:
                    error_log.append({'formula': cc_eval, 'target':'cellcarboncontent', 'error': str(e),
                                      'scientificnames': df_temp['scientificname'].dropna().unique().tolist()})
                    continue
                df_merged.loc[df_temp.index, 'cellcarboncontent'] = df_temp['cellcarboncontent']

        if 'formulaforweight2' in df_merged.columns:
            forms2 = df_merged[df_merged['formulaforweight2'].notnull()]['formulaforweight2'].unique()
            for cc_form in forms2:
                df_temp = df_cc_2[df_cc_2['formulaforweight2'] == cc_form].copy()
                cc_eval = cc_form.replace('^','**').lower()
                df_temp, _ = prepare_columns_for_formula(df_temp, cc_eval)
                try:
                    df_temp['cellcarboncontent'] = df_temp.eval(cc_eval)
                except Exception as e:
                    error_log.append({'formula': cc_eval, 'target':'cellcarboncontent', 'error': str(e),
                                      'scientificnames': df_temp['scientificname'].dropna().unique().tolist()})
                    continue
                df_merged.loc[df_temp.index, 'cellcarboncontent'] = df_temp['cellcarboncontent']

        if 'cellcarboncontent' in df_merged.columns:
            df_merged['cellcarboncontent'] = df_merged['cellcarboncontent'].round(2)
            df_datain['cellcarboncontent'] = df_merged['cellcarboncontent']

    if 'biovolume' in df_datain.columns and 'biovolume' in df_in_orig.columns:
        orig_bv = pd.to_numeric(df_in_orig['biovolume'], errors='coerce')
        mask = df_datain['biovolume'].isna() & orig_bv.notna()
        if mask.any():
            df_datain.loc[mask, 'biovolume'] = orig_bv[mask]
            for idx in df_datain.index[mask]:
                fallback_log.append({'target':'biovolume', 'index': int(idx),
                                     'scientificname': df_datain.at[idx,'scientificname'] if 'scientificname' in df_datain.columns else None,
                                     'value_used': orig_bv.loc[idx]})

    if 'cellcarboncontent' in df_datain.columns and 'cellcarboncontent' in df_in_orig.columns:
        orig_cc = pd.to_numeric(df_in_orig['cellcarboncontent'], errors='coerce')
        mask = df_datain['cellcarboncontent'].isna() & orig_cc.notna()
        if mask.any():
            df_datain.loc[mask, 'cellcarboncontent'] = orig_cc[mask]
            for idx in df_datain.index[mask]:
                fallback_log.append({'target':'cellcarboncontent', 'index': int(idx),
                                     'scientificname': df_datain.at[idx,'scientificname'] if 'scientificname' in df_datain.columns else None,
                                     'value_used': orig_cc.loc[idx]})

    if 'totalbiovolume' in CompTraits:
        if 'density' not in df_datain.columns: df_datain['density'] = np.nan
        if 'biovolume' not in df_datain.columns: df_datain['biovolume'] = np.nan
        df_datain['totalbiovolume'] = (pd.to_numeric(df_datain['density'], errors='coerce') *
                                       pd.to_numeric(df_datain['biovolume'], errors='coerce')).round(2)

    if 'surfacevolumeratio' in CompTraits:
        df_datain['surfacevolumeratio'] = (pd.to_numeric(df_datain.get('surfacearea'), errors='coerce') /
                                           pd.to_numeric(df_datain.get('biovolume'), errors='coerce')).round(2)

    if 'totalcarboncontent' in CompTraits:
        df_datain['totalcarboncontent'] = (pd.to_numeric(df_datain.get('density'), errors='coerce') *
                                           pd.to_numeric(df_datain.get('cellcarboncontent'), errors='coerce')).round(2)

    if 'biomass' in CompTraits:
        if 'biovolume' in df_datain.columns:
            df_datain['biomass'] = (pd.to_numeric(df_datain['biovolume'], errors='coerce') * 1.03 / 1000).round(2)
        if 'totalbiomass' in CompTraits:
            df_datain['totalbiomass'] = (pd.to_numeric(df_datain.get('biomass'), errors='coerce') *
                                         pd.to_numeric(df_datain.get('density'), errors='coerce')).round(2)

    for col in CompTraits:
        if col in df_datain.columns:
            df_datain[col] = df_datain[col].where(pd.notna(df_datain[col]), np.nan)

    df_datain.to_csv(traits_file, index=False, encoding='utf-8', sep=';')
    print(f"✅ Output file saved in: {traits_file}")

    if error_log:
        pd.DataFrame(error_log).to_csv(os.path.join(output_dir, 'formula_errors.log'), sep=';', index=False, encoding='utf-8')
        print(f"ℹ️  Logged {len(error_log)} formula errors -> formula_errors.log")

    if fallback_log:
        pd.DataFrame(fallback_log).to_csv(os.path.join(output_dir, 'fallback_log.csv'), sep=';', index=False, encoding='utf-8')
        print(f"ℹ️  Fallbacks applied: {len(fallback_log)} -> fallback_log.csv")

start = time.time()
calc_fun(datain, operator_file, CalcType, CompTraits, traits_file)
print(f"⏱️ Tempo: {time.time()-start:.2f}s")

file_output_dir = open("/tmp/output_dir_" + id + ".json", "w")
file_output_dir.write(json.dumps(output_dir))
file_output_dir.close()
file_traits_file = open("/tmp/traits_file_" + id + ".json", "w")
file_traits_file.write(json.dumps(traits_file))
file_traits_file.close()
