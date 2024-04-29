import pandas as pd
import os
from emission_scan import emission_scan
from excitation_scan import excitation_scan
from reflectance_scan import reflectance_scan
from transmittance import transmittance_scan

DATA_DIR = './data'
workbook_header = 'PPEye001_Summary_data.xlsx'
workbook_name = os.path.join(DATA_DIR, workbook_header)

df_em = emission_scan(DATA_DIR)
# # df_em.to_excel(workbook_name, sheet_name='emission_scan')
#
df_ex = excitation_scan(DATA_DIR)
# # df_ex.to_excel(workbook_name, sheet_name='excitation_scan')

df_ref = reflectance_scan(DATA_DIR)
# df_ref.to_excel(workbook_name, sheet_name='reflectance_scan')

df_trans = transmittance_scan(DATA_DIR)
# df_trans.to_excel(workbook_name, sheet_name='transmittance_scan')
# exit()

writer = pd.ExcelWriter(workbook_header, engine='xlsxwriter')
df_em.to_excel(writer, sheet_name='emission_scan', index=False)
df_ex.to_excel(writer, sheet_name='excitation_scan', index=False)
df_ref.to_excel(writer, sheet_name='reflectance_scan', index=False)
df_trans.to_excel(writer, sheet_name='transmittance_scan', index=False)
writer.save()

