import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

FILE_PATH = './PPEye001_Summary_data.xlsx'

df_em = pd.read_excel(FILE_PATH, sheet_name='emission_scan')
df_ex = pd.read_excel(FILE_PATH, sheet_name='excitation_scan')
df_ref = pd.read_excel(FILE_PATH, sheet_name='reflectance_scan')
df_trans = pd.read_excel(FILE_PATH, sheet_name='transmittance_scan')

print('columns names: ', df_em.columns)

# Select the scan to be plotted. Either one
# df = df_em
# df = df_ex
# df = df_ref
df = df_trans

# Select/Write column names (file names) to be plotted. No selection plots all.
column_names = []
# column_names = ['ppeye001-06-02']

if len(column_names) == 0:
    data = df.to_numpy()
    column_names = df.columns.tolist()[1:]
else:
    data = df[['wavelength', *column_names]].to_numpy()
print('data shape', data.shape)

for idx in range(data.shape[1] - 1):
    plt.plot(data[:, 0], data[:, idx+1], label=column_names[idx])
plt.legend()
plt.show()

