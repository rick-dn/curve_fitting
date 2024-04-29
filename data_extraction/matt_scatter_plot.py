import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILE = '/home/bappadityadebnath/Downloads/Lifetime_fits.xlsx'

df = pd.read_excel(FILE, sheet_name='Lifetime_fit_635')
print('df\n', df.columns)

relative_count = {}
for grade in df['Grade'].unique():
    for _type in df['Type'].unique():
        for sl_contents in df['Slide contents'].unique():
            relative_count[str(grade) + '_' + str(_type) + '_' + str(sl_contents)] = \
                df.loc[(df['Grade'] == grade) & (df['Type'] == _type) & (df['Slide contents'] == sl_contents),
                       'Relative cont Pp (%)'].to_numpy()

            # print('relative count', relative_count)
            # exit()

# scatter
box_plot_dict = {}
for key in relative_count.keys():
    # print('key array', key, relative_count[key], len(relative_count[key]))
    if len(relative_count[key]) != 0:
        x = relative_count[key]
        box_plot_dict[key] = x[~np.isnan(x)]
    for value in relative_count[key]:
        pass
        # print('value', value)
        plt.scatter(key, value)
# plt.show()

print('box plot', box_plot_dict.keys())
# exit()
# combine pending with infiltration zone 4.0_GBM_Infiltration zone '4.0_GBM_Pending'
box_plot_dict['4.0_GBM_Infiltration zone'] = np.append(box_plot_dict['4.0_GBM_Infiltration zone'], (box_plot_dict['4.0_GBM_Pending']))
box_plot_dict_keys = ['4.0_GBM_Tumour', '4.0_GBM_Necrosis with tumour', '4.0_GBM_Infiltration zone', '4.0_GBM_No tumour',
                      '3.0_Astro_Tumour', '3.0_Astro_Infiltration zone', '3.0_Astro_Tumour', '3.0_Astro_No tumour',
                      '3.0_Oligo_Tumour', '3.0_Oligo_Infiltration zone', '3.0_Oligo_No tumour',
                      '2.0_Astro_Tumour', '2.0_Astro_Infiltration zone', '0.0_Control_No tumour']

box_plot_dict_reshuffled = {}
for key in box_plot_dict_keys:
    box_plot_dict_reshuffled[key] = box_plot_dict[key]




# # # boxplot
print('box plot', box_plot_dict)
fig, ax = plt.subplots()
# for key, value in box_plot_dict.items():
# ax.boxplot(box_plot_dict.values())
ax.boxplot(box_plot_dict_reshuffled.values())
#
# # print('df2\n', df2)
# plt.xticks(list(range(1, len(box_plot_dict.keys()) + 1)), box_plot_dict.keys())
plt.xticks(list(range(1, len(box_plot_dict_reshuffled.keys()) + 1)), box_plot_dict_reshuffled.keys())
plt.show()
