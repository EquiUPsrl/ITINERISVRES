import os
import pandas as pd

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--datain', action='store', type=str, required=True, dest='datain')


args = arg_parser.parse_args()
print(args)

id = args.id

datain = args.datain.replace('"','')



output_dir = 'Output'

print("Creo la cartella " + output_dir)
os.makedirs(output_dir, exist_ok=True)

cluster = ['country','locality','year','month','parentEventID','eventID']
taxlev = 'acceptedNameUsage'
param = ['Density', 'totalbiovolume']
threshold = 0.75

dataset = pd.read_csv(datain, sep=',', decimal='.', low_memory=False)

if 'Density' not in dataset.columns:
    dataset['Density'] = 1
if 'Biovolume' not in dataset.columns:
    dataset['Biovolume'] = 1
if 'CellCarbonContent' not in dataset.columns:
    dataset['CellCarbonContent'] = 1

if 'totalbiovolume' not in dataset.columns:
    dataset['totalbiovolume'] = dataset['Biovolume'] * dataset['Density']
if 'totalcarboncontent' not in dataset.columns:
    dataset['totalcarboncontent'] = dataset['CellCarbonContent'] * dataset['Density']

if cluster[0] != "WHOLE":
    if len(cluster) > 1:
        ID = dataset[cluster].apply(lambda x: '.'.join(x.astype(str)), axis=1)
    else:
        ID = dataset[cluster[0]]
else:
    ID = pd.Series(['all'] * len(dataset))

if 'Density' in param:
    IDZ = ID.unique()
    IDLIST = {}
    for idz in IDZ:
        ddd = dataset[ID == idz]
        totz = ddd['Density'].sum()
        matz = ddd.groupby(taxlev)['Density'].sum() / totz
        matz = matz.sort_values(ascending=False)
        k = 1
        trs = matz.max()
        while trs < threshold:
            matz.iloc[k] = matz.iloc[k] + matz.iloc[k-1]
            trs = matz.iloc[k]
            k += 1
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
            k += 1
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
            k += 1
        matzx = matz.iloc[:k]
        IDLIST[idz] = ddd[ddd[taxlev].isin(matzx.index)]

    dataset_c = pd.concat(IDLIST.values())

else:
    dataset_c = pd.DataFrame()

dtc = pd.concat([dataset_d, dataset_b, dataset_c])

final = dtc.drop_duplicates()

threshold_str = f'ThresholdValue_{threshold}'

output_file = f'{output_dir}/FilteringOutput_{threshold_str}.csv'
final.to_csv(output_file, index=False, sep=';', decimal='.', quoting=0)

print(f"Output saved to {output_file}")

