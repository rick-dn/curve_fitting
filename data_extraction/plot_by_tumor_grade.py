import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

DIR_PATH = '/home/bappadityadebnath/myprojects/matt_automation/rsg/Graph_Plotting/Graph_Plotting'
grader_file = '/home/bappadityadebnath/myprojects/matt_automation/rsg/Stage_1_Pathology.xlsx'
df_grade = pd.read_excel(grader_file, sheet_name='Tumour')
print('df_grade', df_grade.columns)

grade_4, grade_3, grade_2, grade_0 = [], [], [], []
for case in glob.glob(DIR_PATH + '/*.xlsx'):

    print('case: ', case)
    df_em = pd.read_excel(case, sheet_name='emission_scan_1.5')
    df = df_em
    # df_ex = pd.read_excel(case, sheet_name='excitation_scan_1.5')
    # df = df_ex

    for (index, col_name) in enumerate(df.columns[2:]):

        if 'Control' in case:
            print('control case', case)
            grade_0.append(df[col_name].to_numpy())
            continue

        print('col_name: ', index, col_name)
        col_name_grade = col_name.strip('_Em1.5')
        print('col_name_grade: ', index, col_name_grade)

        grade = df_grade.loc[(df_grade['Specimen_ID'] == col_name_grade) & (df_grade['slide_contents'] == 'Tumour')]['WHO_grade'].to_numpy()

        if grade.any():
            if grade[0] == 4:
                grade_4.append(df[col_name].to_numpy())
            elif grade[0] == 3:
                grade_3.append(df[col_name].to_numpy())
            elif grade[0] == 2:
                grade_2.append(df[col_name].to_numpy())
        # exit()
        # print('grade_4', grade_4)
        # print('grade_3', grade_3)
        # print('grade_2', grade_2)
    # break
    # exit()

grade_4 = np.asarray(grade_4, dtype=np.float32)
grade_3 = np.asarray(grade_3, dtype=np.float32)
grade_2 = np.asarray(grade_2, dtype=np.float32)
grade_0 = np.asarray(grade_0, dtype=np.float32)

grade_4 = np.average(grade_4, axis=0)
grade_3 = np.average(grade_3, axis=0)
grade_2 = np.average(grade_2, axis=0)
grade_0 = np.average(grade_0, axis=0)

x = df['wavelength']

# grade 4 = red
# grade 3 = orange
# grade 2 = green
# control = blue

plt.errorbar(x, grade_4, x + np.std(grade_4), linestyle=None, marker='^', label='grade 4', color='red')
plt.errorbar(x, grade_3, x + np.std(grade_3), linestyle=None, marker='^', label='grade 3', color='orange')
plt.errorbar(x, grade_2, x + np.std(grade_2), linestyle=None, marker='^', label='grade 2', color='green')
plt.errorbar(x, grade_0, x + np.std(grade_0), linestyle=None, marker='^', label='control', color='blue')

plt.ylim(0, None)
plt.legend()
plt.show()
