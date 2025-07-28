import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--output_file_traits', action='store', type=str, required=True, dest='output_file_traits')


args = arg_parser.parse_args()
print(args)

id = args.id

output_file_traits = args.output_file_traits.replace('"','')



input_dir = 'data'
output_dir = '/tmp/data/output'

print("Creo la cartella " + output_dir)
os.makedirs(output_dir, exist_ok=True)


datain = output_file_traits

cluster = ['country', 'locality', 'year', 'month', 'day', 'parenteventid', 'eventid']
taxlev = 'scientificname'
param = ['density', 'totalbiovolume']
threshold = 0.75

dataset = pd.read_csv(datain, sep=';', decimal='.', low_memory=False)

if 'density' not in dataset.columns:
    dataset['density'] = 1
if 'biovolume' not in dataset.columns:
    dataset['biovolume'] = 1
if 'cellcarboncontent' not in dataset.columns:
    dataset['cellcarboncontent'] = 1

if 'totalbiovolume' not in dataset.columns:
    dataset['totalbiovolume'] = dataset['biovolume'] * dataset['density']
if 'totalcarboncontent' not in dataset.columns:
    dataset['totalcarboncontent'] = dataset['cellcarboncontent'] * dataset['density']

if cluster[0] != "WHOLE":
    if len(cluster) > 1:
        ID = dataset[cluster].apply(lambda x: '.'.join(x.astype(str)), axis=1)
    else:
        ID = dataset[cluster[0]]
else:
    ID = pd.Series(['all'] * len(dataset))

if 'density' in param:
    IDZ = ID.unique()
    IDLIST = {}
    for idz in IDZ:
        ddd = dataset[ID == idz]
        totz = ddd['density'].sum()
        matz = ddd.groupby(taxlev)['density'].sum() / totz
        matz = matz.sort_values(ascending=False)
        k = 1
        trs = matz.max()
        while trs < threshold:
            matz.iloc[k] = matz.iloc[k] + matz.iloc[k-1]
            trs = matz.iloc[k]
            k = k + 1
        matzx = matz.iloc[:k]
        IDLIST[idz] = ddd[ddd[taxlev].isin(matzx.index)]

    dataset_d = pd.concat(IDLIST.values())

else:
    dataset_d = pd.DataFrame()

if 'totalbiovolume' in param:
    IDZ = ID.unique()
    IDLIST = {}
    for idz in IDZ:
        ddd = dataset[ID == idz]
        totz = ddd['totalbiovolume'].sum()
        matz = ddd.groupby(taxlev)['totalbiovolume'].sum() / totz
        matz = matz.sort_values(ascending=False)
        k = 1
        trs = matz.max()
        while trs < threshold:
            matz.iloc[k] = matz.iloc[k] + matz.iloc[k-1]
            trs = matz.iloc[k]
            k = k + 1
        matzx = matz.iloc[:k]
        IDLIST[idz] = ddd[ddd[taxlev].isin(matzx.index)]

    dataset_b = pd.concat(IDLIST.values())

else:
    dataset_b = pd.DataFrame()

if 'totalcarboncontent' in param:
    IDZ = ID.unique()
    IDLIST = {}
    for idz in IDZ:
        ddd = dataset[ID == idz]
        totz = ddd['totalcarboncontent'].sum()
        matz = ddd.groupby(taxlev)['totalcarboncontent'].sum() / totz
        matz = matz.sort_values(ascending=False)
        k = 1
        trs = matz.max()
        while trs < threshold:
            matz.iloc[k] = matz.iloc[k] + matz.iloc[k-1]
            trs = matz.iloc[k]
            k = k + 1
        matzx = matz.iloc[:k]
        IDLIST[idz] = ddd[ddd[taxlev].isin(matzx.index)]

    dataset_c = pd.concat(IDLIST.values())

else:
    dataset_c = pd.DataFrame()

dtc = pd.concat([dataset_d, dataset_b, dataset_c])

final = dtc.drop_duplicates()

threshold_str = f'ThresholdValue_{threshold}'

output_file_df = f'{output_dir}/FilteringOutput_{threshold_str}.csv'
final.to_csv(output_file_df, index=False, sep=';', decimal='.', quoting=0)

print(f"Output saved to {output_file_df}")

file_output_file_df = open("/tmp/output_file_df_" + id + ".json", "w")
file_output_file_df.write(json.dumps(output_file_df))
file_output_file_df.close()
