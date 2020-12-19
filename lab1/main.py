import pandas as pd
import numpy as np
import statsmodels.api as sm

from utils import read_data


def find_correlations(data: pd.DataFrame):
    print(data.corr())


def reg_m(y, x):
    x2 = x.copy()
    x2[:, 0], x2[:, 2] = x[:, 2], x[:, 0]
    x2 = x2.T
    ones = np.ones(len(x2[0]))
    X = sm.add_constant(np.column_stack((x2[0], ones)))
    for ele in x2[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results


def predict(x1, x2, x3):
    return -0.1521*x3 + 1.2953 * x2 + 0.7156*x1 - 39.9197


if __name__ == '__main__':
    data = read_data()
    find_correlations(data)
    print(reg_m(data.iloc[:, -1].to_numpy(), data.iloc[:, :-1].to_numpy()).summary())
    print(f"\n\nreal value x4 = {data.iloc[3, -1]}, predicted value x4 = {predict(*data.iloc[3, :-1].values)}")


