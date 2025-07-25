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


arg_parser.add_argument('--datain', action='store', type=str, required=True, dest='datain')


args = arg_parser.parse_args()
print(args)

id = args.id

datain = args.datain.replace('"','')



input_dir = 'data'
output_dir = '/tmp/data/output'

print("Creo la cartella " + output_dir)
os.makedirs(output_dir, exist_ok=True)


cluster = ['country','locality','year','month','parentEventID','eventID']
taxlev = 'acceptedNameUsage'
param = 'Biovolume'

dataset = pd.read_csv(datain, sep=";", decimal=".", encoding="UTF-8")

if taxlev != 'community':
    if len(dataset[taxlev].unique()) == 1:
        taxlev = 'community'

if 'Density' not in dataset.columns:
    dataset['Density'] = 1
if 'Biovolume' not in dataset.columns:
    dataset['Biovolume'] = np.nan
if 'CellCarbonContent' not in dataset.columns:
    dataset['CellCarbonContent'] = np.nan

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
        den = dataset.groupby([ID, dataset[taxlev]])['Density'].sum().reset_index()
        biom = dataset.groupby([ID, dataset[taxlev]])['Biovolume'].mean().reset_index()
        cc = dataset.groupby([ID, dataset[taxlev]])['CellCarbonContent'].mean().reset_index()
        if len(cluster) > 1:
            mat = den.merge(biom, on=['level_0', 'acceptedNameUsage']).merge(cc, on=['level_0', 'acceptedNameUsage'])
            mat.columns = ['cluster', taxlev, 'Density', 'Biovolume', 'CellCarbonContent']
        elif len(cluster) == 1:
            mat = den.merge(biom, on=[cluster[0], 'acceptedNameUsage']).merge(cc, on=[cluster[0], 'acceptedNameUsage'])
            mat.columns = [cluster[0], taxlev, 'Density', 'Biovolume', 'CellCarbonContent']
    elif taxlev == 'community' and cluster[0] != 'WHOLE':
        den = dataset.groupby(ID)['Density'].sum()
        biom = dataset.groupby(ID)['Biovolume'].mean()
        cc = dataset.groupby(ID)['CellCarbonContent'].mean()
        mat = pd.concat([den, biom, cc], axis=1)
        mat.columns = ['Density', 'Biovolume', 'CellCarbonContent']
        if len(mat) == 1:
            xx = mat[param]
            plt.plot(xx, den, marker='o')
            if param == 'Biovolume':
                plt.xlabel('average Biovolume (μm^3)')
            elif param == 'CellCarbonContent':
                plt.xlabel('average cell carbon content (pg C)')
            plt.ylabel('Density (cell·L^-1)')
            plt.title('cluster' + taxlev, line=2.5)
            subt = 'cluster: ' + ', '.join(cluster)
            subtitle = '\n'.join(textwrap.wrap(subt, width=50))
            plt.gcf().text(0.5, 0.05, subtitle)
            mat = mat.reset_index()
            mat.columns = ['cluster', 'Density', 'average Biovolume', 'average cell carbon content']
            final = pd.concat([info.loc[mat['cluster']], mat], axis=1)
            final.to_csv(f"{output_dir}/sizedensity_DATA_.csv", sep=';', index=False, quoting=0, encoding='latin1')

    elif taxlev != 'community' and cluster[0] == 'WHOLE':
        den = dataset.groupby(dataset[taxlev])['Density'].sum()
        biom = dataset.groupby(dataset[taxlev])['Biovolume'].mean()
        cc = dataset.groupby(dataset[taxlev])['CellCarbonContent'].mean()
        mat = pd.concat([den, biom, cc], axis=1)
        mat.columns = ['Density', 'Biovolume', 'CellCarbonContent']

    if len(mat) > 1:
        valid = mat[mat['Density'].notna() & mat[param].notna() & (mat['Density'] > 0) & (mat[param] > 0)]

        print(f"Righe valide per la regressione: {len(valid)} su {len(mat)}")
        if len(valid) > 1:
            xx = valid[param]
            mod = sm.OLS(np.log(valid['Density']), sm.add_constant(np.log(xx))).fit()
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

            plt.scatter(xx, valid['Density'])
            plt.xscale('log')
            plt.yscale('log')
            if param == 'Biovolume':
                plt.xlabel('average Biovolume ($\mu m^3$)')
            elif param == 'CellCarbonContent':
                plt.xlabel('average cell carbon content (pg C)')
            plt.ylabel('Density (cell·L⁻¹)')
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
                mat.columns = ['cluster', taxlev, 'Density', 'average Biovolume', 'average cell carbon content']
                if len(cluster) > 1:
                    final = pd.concat([info.loc[mat['cluster']].reset_index(), mat], axis=1)
                    final = final.drop(columns=['index', 'cluster'])
                if len(cluster) == 1:
                    mat.columns = [cluster[0], taxlev, 'Density', 'average Biovolume', 'average cell carbon content']
                    final = mat
                final = final.dropna(thresh=3)
                final = final.round(2)
                final.to_csv(f"{output_dir}/sizedensity_DATA_.csv", sep=';', index=False, quoting=0, encoding='latin1')

            elif taxlev == 'community' and cluster[0] != 'WHOLE':
                mat = mat.reset_index()
                if len(cluster) > 1:
                    mat.columns = ['cluster', 'Density', 'average Biovolume', 'average cell carbon content']
                    mat = pd.concat([info.loc[mat['cluster']].reset_index(), mat], axis=1)
                    mat = mat.drop(columns=['index', 'cluster'])
                elif len(cluster) == 1:
                    mat.columns = [cluster[0], 'Density', 'average Biovolume', 'average cell carbon content']
                mat = mat.round(2)
                mat.to_csv(f"{output_dir}/sizedensity_DATA_.csv", sep=';', index=False, quoting=0, encoding='latin1')

            elif taxlev != 'community' and cluster[0] == 'WHOLE':
                mat = mat.reset_index()
                mat.columns = [taxlev, 'Density', 'average Biovolume', 'average cell carbon content']
                mat.to_csv(f"{output_dir}/sizedensity_DATA_.csv", sep=';', index=False, quoting=0, encoding='latin1')

            rr.to_csv(f"{output_dir}/sizedensity_MODEL_LM_.csv", sep=';', index=True, header=True, quoting=0, encoding='latin1')
        else:
            print("Attenzione: non ci sono abbastanza dati validi per eseguire la regressione.")

    plt.show()

else:
    den = dataset['Density'].sum()
    biom = round(dataset['Biovolume'].mean(), 2)
    cc = round(dataset['CellCarbonContent'].mean(), 2)
    mat = pd.Series({'Density': den, 'Biovolume': biom, 'CellCarbonContent': cc})
    xx = mat[param]

    file_graph = f"{output_dir}/sizedensityOutput_.pdf"
    plt.figure(figsize=(8, 6))
    plt.plot(xx, den, marker='o')
    if param == 'Biovolume':
        plt.xlabel('average Biovolume (μm^3)')
    elif param == 'CellCarbonContent':
        plt.xlabel('average cell carbon content (pg C)')
    plt.ylabel('Density (cell·L^-1)')
    plt.title('Whole dataset')
    plt.savefig(file_graph)

    mat.index = ['Density', 'average Biovolume', 'average cell carbon content']
    mat.to_csv(f"{output_dir}/sizedensity_DATA_.csv", sep=';', index=True, header=False, quoting=0, encoding='latin1')

    plt.show()

