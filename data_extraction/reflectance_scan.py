import glob
import os

import numpy as np
import pandas as pd


def reflectance_scan(data_dir):

    df = None
    for file_no, file in enumerate(sorted(glob.glob(data_dir + '/*/*Mean.Raw.csv'))):

        column_name = os.path.basename(file).strip('Mean.Raw.csv')
        # print('file', file, column_name)

        if file_no == 0:
            df = pd.read_csv(file, usecols=['nm'])
        df = df.assign(**{column_name: pd.read_csv(file).iloc[:, 1].to_numpy()})
        # print('df', df)
        # exit()

    # print('df', df)
    data = df.to_numpy()
    # print('df info', len(df), len(df.columns))
    data_int = (data[:-1, :] + data[1:, :]) / 2
    # print('data_int', data_int[:, 0])
    data = np.concatenate((np.array(list(zip(data, data_int))).reshape(data_int.shape[0] * 2, -1),
                          data[-1, :].reshape(1, data.shape[1])),
                          axis=0)
    # print('data', data.shape)
    df = pd.DataFrame(np.flip(data, axis=0), columns=df.columns)
    # print('data', df)
    return df
