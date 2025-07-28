import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import textwrap

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--output_file_df', action='store', type=str, required=True, dest='output_file_df')


args = arg_parser.parse_args()
print(args)

id = args.id

output_file_df = args.output_file_df.replace('"','')



output_dir = '/tmp/data/output'

print("Creo la cartella " + output_dir)
os.makedirs(output_dir, exist_ok=True)

datain = output_file_df

SizeUnit = 'biovolume'

cluster = ['country', 'locality', 'year', 'month', 'day', 'parenteventid', 'eventid']

base = 2


dataset = pd.read_csv(datain, sep=';', decimal='.')

if SizeUnit == 'biovolume':
    var = dataset['biovolume'].dropna().astype(float)
    xlabz = f"log{int(base)} biovolume (µm³)" if base in [2, 10] else 'ln biovolume (µm³)'
elif SizeUnit == 'cellcarboncontent':
    var = dataset['cellcarboncontent'].dropna().astype(float)
    xlabz = f"log{int(base)} cell carbon content (pg C)" if base in [2, 10] else 'ln cell carbon content (pg C)'

if cluster[0] == "WHOLE":
    logvar = np.round(np.log(var) / np.log(base))
    ttz = pd.Series(logvar).value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(ttz.index, ttz.values)
    plt.xlabel(xlabz)
    plt.ylabel('N of cells')
    plt.title("Whole dataset")
    plt.ylim(0, max(ttz.values))

    pdf_path = os.path.join(output_dir, f"SizeClassOutput_log{base}_{SizeUnit}.pdf")
    plt.savefig(pdf_path, format='pdf')

    final = pd.DataFrame({f'log{base}_{SizeUnit}': ttz.index, 'N of cells': ttz.values})

else:
    if len(cluster) > 1:
        ID = dataset[cluster].astype(str).apply('.'.join, axis=1)
        info = dataset[cluster].dropna().drop_duplicates().astype(str)
        info.index = info.apply('.'.join, axis=1)
    else:
        ID = dataset[cluster[0]].dropna().astype(str)
        info = dataset[cluster].drop_duplicates()
        info.set_index(info[cluster[0]], inplace=True)

    subtitle = '\n'.join(textwrap.wrap(f'cluster: {", ".join(cluster)}', width=50))
    pdf_path = os.path.join(output_dir, f"SizeClassOutput_log{base}_{SizeUnit}.pdf")
    filegraph = PdfPages(pdf_path)

    def ccfun(x, mainz, xlb, subtitle, filegraph):
        logvar = np.round(np.log(var[x.index]) / np.log(base))
        ttz = pd.Series(logvar).value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        plt.bar(ttz.index, ttz.values)
        plt.xlabel(xlb)
        plt.ylabel('N of cells')
        plt.title(mainz)
        plt.ylim(0, max(ttz.values))
        plt.text(0, 1.1, subtitle, transform=plt.gca().transAxes, fontsize=10)
        plt.savefig(filegraph, format='pdf')

        return ttz

    idz = ID.unique()
    cclist = {id_: ccfun(var[ID == id_], mainz=id_, xlb=xlabz, subtitle=subtitle, filegraph=filegraph) for id_ in idz}
    filegraph.close()

    data_rbind = np.transpose(pd.concat([cclist[id_] for id_ in idz], axis=1, sort=True, ignore_index=True))

    if len(idz) > 1:
        final = pd.concat([info.reset_index(drop=True), data_rbind], axis=1)
    else:
        final = pd.concat([info, data_rbind], axis=1, sort=True)

    final = final.fillna(0)

if base in [2, 10]:
    csv_path = os.path.join(output_dir, f"SizeClassOutput_log{base}_{SizeUnit}.csv")
else:
    csv_path = os.path.join(output_dir, f"SizeClassOutput_ln{SizeUnit}.csv")

final.to_csv(csv_path, sep=';', decimal='.', index=False, encoding='latin1')

output_file_sc = csv_path

