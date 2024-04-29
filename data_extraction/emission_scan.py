import glob
import os

import numpy as np
import pandas as pd


def emission_scan(data_dir):

    df = None
    for file_no, file in enumerate(sorted(glob.glob(data_dir + '/*/*-EmScan.txt'))):

        column_name = os.path.basename(file).strip('-EmScan.txt')
        # print('file', file, column_name)

        file_data = open(file, 'r')
        lines = file_data.readlines()

        data_flag = False
        file_data = []
        for idx, line in enumerate(lines):

            if data_flag is False:
                # print('line, length', len(line), idx)
                if len(line) == 1:
                    data_flag = True
            else:
                # print('line: ', line.split(',')[0], idx)
                file_data.append((line.split(',')[0], line.split(',')[1]))

        file_data = np.array(file_data, dtype=np.float32)
        # print('all_data', file_data)

        if file_no == 0:
            df = pd.DataFrame({'wavelength': file_data[:, 0]})
        df = df.assign(**{column_name: file_data[:, 1]})
        # print('df', df)

    # print('df', df)
    return df
