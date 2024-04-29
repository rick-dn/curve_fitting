import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

FILE = '/home/bappadityadebnath/myprojects/hsi_qfl/workbooks/PPEye_stage_1.xlsx'

# df = pd.read_excel(FILE, sheet_name='Lifetime_fit_635')

sheets_dict = pd.read_excel(FILE, sheet_name=None)

all_sheets = []
for name, sheet in sheets_dict.items():

    print('sheet name:', name)
    if name in ['mean', 'PPEyeControl001', 'PPEyeControl002']:
        print('excluded sheet name:', name)
        continue
    sheet['sheet'] = name
    sheet = sheet.rename(columns=lambda x: x.split('\n')[-1])
    all_sheets.append(sheet)

full_table = pd.concat(all_sheets)
full_table.reset_index(inplace=True, drop=True)

print(full_table.columns)
print(full_table.mean())
print(full_table.std())

# print(full_table)

RED_CHI_SQUARES = full_table['red_chi_squares'].to_numpy()
# print(RED_CHI_SQUARES)
mae_red_chi_square = mean_absolute_error(np.ones(RED_CHI_SQUARES.shape),
                                                          RED_CHI_SQUARES)

print('chisquares MEA: {:.2f}'.format(mae_red_chi_square))

exit()

# plt.errorbar(full_table['tau3'], full_table['red_chi_squares'], np.std(full_table['tau3']), linestyle=None, marker='^', label='grade 4', color='red')
x = full_table['tau3']
y = full_table['red_chi_squares']

x = full_table['rel_concentration']
y = full_table['rel_fl_int']

plt.scatter(x, y)
fig, ax = plt.subplots(figsize = (9, 9))
ax.scatter(x, y)

b, a = np.polyfit(x, y, deg=2)

xseq = np.linspace(np.min(x), np.max(x), num=100)
reg_line = a + b * xseq
# ax.plot(xseq, reg_line, color="k", lw=2.5)
ax.plot(xseq, reg_line, color="k", lw=2.5)

# # plt.scatter(full_table['tau3'], full_table['red_chi_squares'])
#
# fig, ax = plt.subplots(figsize = (9, 9))
# ax.scatter(x, y)
#
# b, a = np.polyfit(x, y, deg=1)
#
# xseq = np.linspace(np.min(x), np.max(x), num=100)
# reg_line = a + b * xseq
# # ax.plot(xseq, reg_line, color="k", lw=2.5)
# ax.errorbar(xseq, reg_line, np.std(reg_line), color="k", lw=2.5)
#
plt.show()