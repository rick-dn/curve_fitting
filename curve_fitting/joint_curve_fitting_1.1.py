import argparse
import collections
import re

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

from curve_fitting import CurveFitting, plot_curves, post_process, write_to_sheet, get_data, preprocess


class JointCurveFitting(CurveFitting):
    # def __init__(self, **s_args):
    #     super(JointCurveFitting, self).__init__(**s_args)

    @staticmethod
    def __lsq_func(P, X, Y):

        # print('X, Y', np.asarray(X, dtype=np.float32).shape)
        # exit()

        # l = 10000
        # for x in X:
        #     if len(x) < l:
        #         l = len(x)

        # print('min X len', l)
        # exit()

        RES = []
        for idx, (x, y) in enumerate(zip(X, Y)):

            # x = np.asarray(x[:l], dtype=np.float32)
            # y = np.asarray(y[:l], dtype=np.float32)

            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            # plt.plot(x, y)
            # plt.show()

            # if idx > 10:
            #     exit()
            p = P[idx * 6:idx * 6 + 6].tolist()
            p.insert(-1, P[-1])
            # print('p init for single sample', p)
            # exit()
            # print('x, y, shape', len(x), len(y))

            y_fitted, _ = CurveFitting.forward_model(x, p)
            # print('y_fitted', y_fitted.shape[0])
            # exit()
            # res = (1 / (x.shape[0] - len(p))) * np.sum(np.square(y_fitted - y)/y_fitted)
            # res = (1 / (x.shape[0] - len(p) + 4)) * np.square(y_fitted - y)/y_fitted
            # res = (y - y_fitted) / np.sqrt(y)
            res = np.divide((y - y_fitted), np.sqrt(y))
            # print('res', res.shape)
            # if res.shape[0] < 400:
            #     res = np.append(res, np.zeros(400 - y.shape[0]))
            # print('res shape: ', res.shape)
            # RES = RES + res
            RES.extend(res)
            # print('RES', RES.shape)


        # exit()

        return np.asarray(RES, dtype=np.float32)
        # return RES

    # @staticmethod
    # def __lsq_func(P, X, Y):
    #
    #     RES = 0
    #     for idx, (x, y) in enumerate(zip(X, Y)):
    #
    #         x = np.asarray(x, dtype=np.float32)
    #         y = np.asarray(y, dtype=np.float32)
    #         # if idx > 10:
    #         #     exit()
    #         p = P[idx*6:idx*6 + 6].tolist()
    #         p.insert(-1, P[-1])
    #         # print('p init for single sample', p)
    #         # exit()
    #         # print('x, y, shape', len(x), len(y))
    #
    #         y_fitted, _ = CurveFitting.forward_model(x, p)
    #         # print('y_fitted', y_fitted.shape[0])
    #         # exit()
    #         # res = (1 / (x.shape[0] - len(p))) * np.sum(np.square(y_fitted - y)/y_fitted)
    #         # res = (1 / (x.shape[0] - len(p) + 4)) * np.square(y_fitted - y)/y_fitted
    #         # res = (y - y_fitted) / np.sqrt(y)
    #         res = np.divide((y - y_fitted), np.sqrt(y))
    #         if res.shape[0] < 400:
    #             res = np.append(res, np.zeros(400-y.shape[0]))
    #         # print('res shape: ', res.shape)
    #         RES = RES + res
    #
    #     return RES

    def fit_lsq(self, X, Y):

        n_samples = len(X)
        _init_params = np.tile(self.init_params, n_samples)
        _init_params = _init_params.tolist()
        del _init_params[5::7]
        _init_params = [*_init_params, self.init_params[-2]]
        # print('init_params', len(init_params), _init_params)
        # exit()

        lower_bounds = self.bounds[0]
        lower_bounds = np.delete(lower_bounds, -2)
        lower_bounds = np.tile(lower_bounds, n_samples)
        lower_bounds = np.append(lower_bounds, self.bounds[0][-2])
        # print('lower bounds', lower_bounds)

        upper_bounds = self.bounds[1]
        upper_bounds = np.delete(upper_bounds, -2)
        upper_bounds = np.tile(upper_bounds, n_samples)
        upper_bounds = np.append(upper_bounds, self.bounds[1][-2])
        # print('upper_bounds', upper_bounds)

        self.bounds = [lower_bounds, upper_bounds]
        # exit()

        if self.use_jac is True:
            res_lsq = least_squares(self.__lsq_func, x0=_init_params, jac=self._backward_model,
                                    bounds=self.bounds, args=(X, Y))
        else:
            res_lsq = least_squares(self.__lsq_func, x0=_init_params, bounds=self.bounds, args=(X, Y))
        # return least_squares(self._func_lsq, x0=self.p, bounds=(0, np.inf), args=(x, y), loss='arctan') # trial
        # return least_squares(self._func_lsq, x0=self.p, jac='cs', bounds=(0, np.inf), args=(x, y)) # best
        # return least_squares(self._func_lsq, x0=self.p, args=(x, y), jac=self._d_func_lsq, method='lm')
        print('verbose value: ', self.verbose)
        if self.verbose:
            print('res_lsq', res_lsq)

        self.fitted_params = res_lsq['x']

        # print('fitted_params', self.fitted_params)

        FITTED_PARAMS = []
        for idx in range(len(X)):
            params = self.fitted_params[idx*6:idx*6 + 6].tolist()
            params.insert(-1, self.fitted_params[-1])
            # print('p fitted for single sample: ', params)
            FITTED_PARAMS.append(params)

        self.fitted_params = FITTED_PARAMS

        # print('fitted_params', self.fitted_params)


def main(args):

    DATA = get_data(args.path, args.pattern)

    if not bool(DATA):
        raise KeyError('DATA is empty')

    DATA = preprocess(DATA, tail_trim=args.tail_trim, zero_trim=args.zero_trim)
    print('n_files parsed', len(DATA.keys()))
    # exit()
    DATA = collections.OrderedDict(sorted(DATA.items(), key=lambda _i: _i[0].lower()))

    joint_fitting = JointCurveFitting(args.init_params, args.bounds,
                                 use_jac=args.use_jac)
    # curve_fitting = JointCurveFitting(init_params=args.init_params, bounds=args.bounds)

    X, Y = [], []

    for i, (filename, [x, y]) in enumerate(DATA.items()):

        X.append(x)
        Y.append(y)

    print('X, Y shape', len(X))
    joint_fitting.fit_lsq(X, Y)
    # exit()


    PARAMS = {}
    for i, ((filename, [x, y]), params) in enumerate(zip(DATA.items(), joint_fitting.fitted_params)):

        # print(x, y, params)
        # exit()
        # single fit params
        # params = [2.915650, 0.667999, 12418.602747, 3.014385, 1116.071179, 15.373059, 217.187080] # zero_trim = true
        # params = [2.868146, 0.654911, 19944.032432, 2.995318, 1118.685635, 15.445829, 217.612341]
        params = np.asarray(params, dtype=np.float32)

        y_fitted, y_comps = CurveFitting.forward_model(x, params)

        matrices = JointCurveFitting.matrices(y, y_fitted, params)
        print('filename n_points: ', filename, x.shape)
        print('fitted_params', params)
        print('rel_fl_int {:.2f}, rel_concentration, {:.2f}, red_chi_sqr {:.2f}\n'.format(*matrices))

        if args.plot:
            plot_curves(x, y=y, y_fitted=y_fitted, y_comps=y_comps)
        # exit()

        PARAMS[filename] = [*params, *matrices]

    df = post_process(PARAMS)

    if args.write:
        write_to_sheet(df, args.wb_name, args.sheet_name)


if __name__ == '__main__':

    # PATH = '/home/bappadityadebnath/Documents/Curve_fitting'
    # PATH = '/home/bappadityadebnath/Downloads/PPEye001/Fluorimeter Data/13102022'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye002'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye005/Fluorimeter/'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye003/PPEye003/Fluorimeter/'
    # print('PATH', PATH)
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/'
    # IRF_FILE = '/home/bappadityadebnath/datasets/matt_data/IRF/IRF_BW10_500ns.txt'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye001/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye002/Fluorimeter/'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye003/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye004/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye005/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye006/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye007/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye008/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye009/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye010/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye011/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye012/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye013/Fluorimeter'

    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data_control/PPEyeControl001/Fluorimeter'
    PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data_control/PPEyeControl002/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data_control/PPEyeControl003/Fluorimeter'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data_control/PPEyeControl004/Fluorimeter'

    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye006/Fluorimeter/PPEye006-002-11-Decay625.txt'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye006/Fluorimeter/PPEye006-002-11-Decay635.txt'
    # PATH = '/home/bappadityadebnath/datasets/matt_data/PPEye_Stage_1_data/PPEye006/Fluorimeter/PPEye006-002-9-Decay635.txt'

    sheet_name = 'PPEye_Stage_1_data'
    # sheet_name = re.search(r"/(PPEye(Control)?0[0-9][0-9])/", PATH)[1]
    # print('sheet_name', sheet_name)
    # exit()

    # pattern
    # pattern = '/*-Decay*.txt'
    # pattern = '/*_Decay*.txt'
    # pattern = '/*_Decay.txt'
    # pattern = ['/*-Decay*.txt', '/*_Decay*.txt', '/*_Decay.txt']
    pattern = ['*-Decay*.txt', '*_Decay*.txt', '*_Decay.txt']

    # in order of tau, beta
    # p = [17, 1000]
    # p = [1.5, 0.01, 14, 0.1]
    # init_params = [10, 5, 5000, 10, 1000]
    # init_params = [0, .7, 6700, 3.5, 2670, 17, 1150]  # previous
    # init_params = [10, .7, 6700, 3.5, 2670, 17, 1150]  # best
    # init_params = [3, .6, 5791, 2.5, 1110, 15, 220]  # current trial
    # init_params = [3, .6, 5791, 3.5, 1110, 15, 220]  # best 22_03_23 11:26
    init_params = [1, .65, 12000, 3.65, 1370, 16, 2046]  # single fit best 26_03_11
    # init_params = [.6, 5791, 2.5, 1110, 15, 220]  # current trial no dc
    # init_params = [10, .7, 6700, 3, 2670] # best for control
    # init_params = [10, .7, 5791, 3.0, 1110, 5, 1000, 17, 220]
    # init_params = [0, .1, 5000, 1, 2500, 10, 1000]
    # init_params = [.7, 6700, 3.5, 2670, 17, 1150]
    # init_params = [10, 0.001, 10000, .7, 6700, 3.5, 2670, 17, 1150]
    # p = [5, 1000, 5, 1000, 10, 1000]

    # bounds
    # bounds = ([0, 0, 0, 0, 0, 14, 0], [np.inf, 1, np.inf, 7, np.inf, 19, np.inf])
    # bounds = ([0, 0, 10000, 0, 500, 14, 1000], [3, 1, np.inf, 7, np.inf, 19, np.inf])
    # bounds = ([0, 0, 0, 0, 14, 0], [1, np.inf, 7, np.inf, 19, np.inf]) # no dc
    # bounds = ([0, 0.1, 0, 0, 0, 0, 0], [np.inf, 1.5, np.inf, 7, np.inf, 19, np.inf])  # best 22/06/23
    bounds = ([0, 0.1, 0, 0, 0, 14, 0], [np.inf, 1.5, np.inf, 7, np.inf, 19, np.inf])  # best 22/06/23
    # bounds = ([0.205968, 0.129488, 764.241971, 0.451159, 191.841089, 1.179123, 29.245843],
    #           [19.468653, 1, 533683.159065, 4.352204, 3499.430514, 17.927812, 865.303385])  # trial
    # bounds = ([0, 0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    # bounds = ([0, 0, 0, 0, 0, 14, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, 19, np.inf])

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", default=PATH)
    parser.add_argument("--pattern", default=pattern)
    parser.add_argument("--tail_trim", default=400)
    parser.add_argument("--zero_trim", default=True)
    parser.add_argument("--init_params", default=init_params)
    parser.add_argument("--bounds", default=bounds)
    parser.add_argument("--use_jac", default=False)
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--plot", default=False)
    parser.add_argument("--write", default=False)
    parser.add_argument("--wb_name", default="./workbooks/PPEye_stage_1_multifit.xlsx")
    parser.add_argument("--sheet_name", default=sheet_name)

    args = parser.parse_args()
    # print('args', args)

    main(args)