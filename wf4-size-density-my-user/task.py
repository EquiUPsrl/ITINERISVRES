import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import statsmodels.api as sm

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



input_dir = 'data'
output_dir = '/tmp/data/output'

print("Creo la cartella " + output_dir)
os.makedirs(output_dir, exist_ok=True)

datain = output_file_df #f'{output_dir}/FilteringOutput_ThresholdValue_0.75.csv'

output_file_sd = f"{output_dir}/sizedensity_DATA_.csv"

cluster = ['country', 'locality', 'year', 'month', 'day', 'parenteventid', 'eventid']
taxlev = 'scientificname'
param = 'biovolume'

dataset = pd.read_csv(datain, sep=";", decimal=".", encoding="UTF-8")

if taxlev != 'community':
    if len(dataset[taxlev].unique()) == 1:
        taxlev = 'community'

if 'density' not in dataset.columns:
    dataset['density'] = 1
if 'biovolume' not in dataset.columns:
    dataset['biovolume'] = np.nan
if 'cellcarboncontent' not in dataset.columns:
    dataset['cellcarboncontent'] = np.nan

if not (taxlev == 'community' and cluster[0] == 'WHOLE'):
    if len(cluster) > 1:
        ID = dataset[cluster].astype(str).apply('.'.join, axis=1)
        info = dataset[cluster].dropna().drop_duplicates().astype(str)
        info.index = info.apply('.'.join, axis=1)
    elif len(cluster) == 1:
        ID = dataset[cluster[0]].dropna().astype(str)
        info = dataset[cluster].drop_duplicates()
        info.set_index(info[cluster[0]], inplace=True)

    if taxlev != 'community' and cluster[0] != 'WHOLE':
        den = dataset.groupby([ID, dataset[taxlev]])['density'].sum().reset_index()
        biom = dataset.groupby([ID, dataset[taxlev]])['biovolume'].mean().reset_index()
        cc = dataset.groupby([ID, dataset[taxlev]])['cellcarboncontent'].mean().reset_index()
        if len(cluster) > 1:
            mat = den.merge(biom, on=['level_0', 'scientificname']).merge(cc, on=['level_0', 'scientificname'])
            mat.columns = ['cluster', taxlev, 'density', 'biovolume', 'cellcarboncontent']
        elif len(cluster) == 1:
            mat = den.merge(biom, on=[cluster[0], 'scientificname']).merge(cc, on=[cluster[0], 'scientificname'])
            mat.columns = [cluster[0], taxlev, 'density', 'biovolume', 'cellcarboncontent']
    elif taxlev == 'community' and cluster[0] != 'WHOLE':
        den = dataset.groupby(ID)['density'].sum()
        biom = dataset.groupby(ID)['biovolume'].mean()
        cc = dataset.groupby(ID)['cellcarboncontent'].mean()
        mat = pd.concat([den, biom, cc], axis=1)
        mat.columns = ['density', 'biovolume', 'cellcarboncontent']
        if len(mat) == 1:
            xx = mat[param]
            plt.plot(xx, den, marker='o')
            if param == 'biovolume':
                plt.xlabel('average biovolume (μm^3)')
            elif param == 'cellcarboncontent':
                plt.xlabel('average cell carbon content (pg C)')
            plt.ylabel('density (cell·L^-1)')
            plt.title('cluster' + taxlev, line=2.5)
            subt = 'cluster: ' + ', '.join(cluster)
            subtitle = '\n'.join(textwrap.wrap(subt, width=50))
            plt.gcf().text(0.5, 0.05, subtitle)
            mat = mat.reset_index()
            mat.columns = ['cluster', 'density', 'average biovolume', 'average cell carbon content']
            final = pd.concat([info.loc[mat['cluster']], mat], axis=1)
            final.to_csv(output_file_sd, sep=';', index=False, quoting=0, encoding='latin1')

    elif taxlev != 'community' and cluster[0] == 'WHOLE':
        den = dataset.groupby(dataset[taxlev])['density'].sum()
        biom = dataset.groupby(dataset[taxlev])['biovolume'].mean()
        cc = dataset.groupby(dataset[taxlev])['cellcarboncontent'].mean()
        mat = pd.concat([den, biom, cc], axis=1)
        mat.columns = ['density', 'biovolume', 'cellcarboncontent']

    if len(mat) > 1:
        valid = mat[mat['density'].notna() & mat[param].notna() & (mat['density'] > 0) & (mat[param] > 0)]

        print(f"Righe valide per la regressione: {len(valid)} su {len(mat)}")
        if len(valid) > 1:
            xx = valid[param]
            mod = sm.OLS(np.log(valid['density']), sm.add_constant(np.log(xx))).fit()
            rr = pd.DataFrame({
                'Estimate': mod.params,
                'Std. Error': mod.bse,
                't-value': mod.tvalues,
                'P-value': mod.pvalues,
                'Rsquared': [np.nan, mod.rsquared]
            })
            rr.index = ['Intercept', 'log average ' + param]

            file_graph = f"{output_dir}/sizedensityOutput_.pdf"
            sq = np.linspace(valid[param].min(), valid[param].max(), 101)
            pr = np.exp(mod.get_prediction(sm.add_constant(np.log(sq))).predicted_mean)
            ci = pd.DataFrame(np.exp(mod.get_prediction(sm.add_constant(np.log(sq))).conf_int()))

            if cluster[0] != 'WHOLE':
                subt = 'cluster: ' + ', '.join(cluster)
            else:
                subt = 'no temporal or spatial cluster'

            plt.scatter(xx, valid['density'])
            plt.xscale('log')
            plt.yscale('log')
            if param == 'biovolume':
                plt.xlabel('average biovolume ($\mu m^3$)')
            elif param == 'cellcarboncontent':
                plt.xlabel('average cell carbon content (pg C)')
            plt.ylabel('density (cell·L⁻¹)')
            if cluster[0] != 'WHOLE':
                plt.title('cluster*' + taxlev)
            else:
                plt.title('taxonomic level: ' + taxlev)
            plt.text(0, 1.1, subt, transform=plt.gca().transAxes, fontsize=10)
            plt.plot(sq, pr, color='red', linestyle='-')
            plt.plot(sq, ci[0], color='red', linestyle='--')
            plt.plot(sq, ci[1], color='red', linestyle='--')
            eq = str(round(np.exp(mod.params[0]), 2)) + '*M^' + str(round(mod.params[1], 2))
            r2 = 'R^2=' + str(round(mod.rsquared, 2))
            plt.legend([eq, r2], loc='upper right')
            plt.savefig(file_graph)

            if taxlev != 'community' and cluster[0] != 'WHOLE':
                mat.columns = ['cluster', taxlev, 'density', 'average biovolume', 'average cell carbon content']
                if len(cluster) > 1:
                    final = pd.concat([info.loc[mat['cluster']].reset_index(), mat], axis=1)
                    final = final.drop(columns=['index', 'cluster'])
                if len(cluster) == 1:
                    mat.columns = [cluster[0], taxlev, 'density', 'average biovolume', 'average cell carbon content']
                    final = mat
                final = final.dropna(thresh=3)
                final = final.round(2)
                final.to_csv(output_file_sd, sep=';', index=False, quoting=0, encoding='latin1')

            elif taxlev == 'community' and cluster[0] != 'WHOLE':
                mat = mat.reset_index()
                if len(cluster) > 1:
                    mat.columns = ['cluster', 'density', 'average biovolume', 'average cell carbon content']
                    mat = pd.concat([info.loc[mat['cluster']].reset_index(), mat], axis=1)
                    mat = mat.drop(columns=['index', 'cluster'])
                elif len(cluster) == 1:
                    mat.columns = [cluster[0], 'density', 'average biovolume', 'average cell carbon content']
                mat = mat.round(2)
                mat.to_csv(f"{output_dir}//sizedensity_DATA_.csv", sep=';', index=False, quoting=0, encoding='latin1')

            elif taxlev != 'community' and cluster[0] == 'WHOLE':
                mat = mat.reset_index()
                mat.columns = [taxlev, 'density', 'average biovolume', 'average cell carbon content']
                mat.to_csv(output_file_sd, sep=';', index=False, quoting=0, encoding='latin1')

            rr.to_csv(f"{output_dir}/sizedensity_MODEL_LM_.csv", sep=';', index=True, header=True, quoting=0, encoding='latin1')
        else:
            print("Attenzione: non ci sono abbastanza dati validi per eseguire la regressione.")

    plt.show()

else:
    den = dataset['density'].sum()
    biom = round(dataset['biovolume'].mean(), 2)
    cc = round(dataset['cellcarboncontent'].mean(), 2)
    mat = pd.Series({'density': den, 'biovolume': biom, 'cellcarboncontent': cc})
    xx = mat[param]

    file_graph = f"{output_dir}/sizedensityOutput_.pdf"
    plt.figure(figsize=(8, 6))
    plt.plot(xx, den, marker='o')
    if param == 'biovolume':
        plt.xlabel('average biovolume (μm^3)')
    elif param == 'cellcarboncontent':
        plt.xlabel('average cell carbon content (pg C)')
    plt.ylabel('density (cell·L^-1)')
    plt.title('Whole dataset')
    plt.savefig(file_graph)

    mat.index = ['density', 'average biovolume', 'average cell carbon content']
    mat.to_csv(output_file_sd, sep=';', index=True, header=False, quoting=0, encoding='latin1')

    plt.show()

file_output_file_sd = open("/tmp/output_file_sd_" + id + ".json", "w")
file_output_file_sd.write(json.dumps(output_file_sd))
file_output_file_sd.close()
