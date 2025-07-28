import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm

import argparse
import json
import os
arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--output_file_sd', action='store', type=str, required=True, dest='output_file_sd')


args = arg_parser.parse_args()
print(args)

id = args.id

output_file_sd = args.output_file_sd.replace('"','')



input_dir = 'data'
output_dir = '/tmp/data/output'

print("Creo la cartella " + output_dir)
os.makedirs(output_dir, exist_ok=True)

datain = output_file_sd #f'{output_dir}/sizedensity_DATA_.csv'

df = pd.read_csv(datain, sep=";", encoding="latin1")

df_clean = df[
    df['density'].notna() &
    df['average biovolume'].notna() &
    (df['density'] > 0) &
    (df['average biovolume'] > 0)
].copy()
df_clean['log_density'] = np.log(df_clean['density'])
df_clean['log_biovolume'] = np.log(df_clean['average biovolume'])

model = ols('log_density ~ log_biovolume * C(locality)', data=df_clean).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

anova_table.to_csv(f"{output_dir}/ancova_output_results.csv")

sns.scatterplot(data=df_clean, x="log_biovolume", y="log_density", hue="locality", alpha=0.6)

for loc in df_clean['locality'].unique():
    subset = df_clean[df_clean["locality"] == loc]
    sns.regplot(
        data=subset,
        x="log_biovolume",
        y="log_density",
        scatter=False,
        label=f"Regr. {loc}"
    )

plt.title("ANCOVA – Regressione log-log tra biovolume e densità per località")
plt.xlabel("Log(Biovolume)")
plt.ylabel("Log(Densità)")
plt.legend()
plt.tight_layout()

plt.savefig(f"{output_dir}/ancova_plot.png", dpi=300)
plt.show()

