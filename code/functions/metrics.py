import numpy as np
import pandas as pd

def rss(y_true: list, y_hat: list) -> float:
    residuals = y_true - y_hat
    rss_ = sum(residuals**2)
    return(rss_)


def ess(y_hat: list) -> float:
    mean_y = np.mean(y_hat)
    deviation = y_hat - mean_y
    ess_ = sum(deviation**2)
    return(ess_)


def tss(y_true: list) -> float:
    mean_y = np.mean(y_true)
    deviation = y_true - mean_y
    tss_ = sum(deviation**2)
    return(tss_)


def r_squared(y_true: list, y_hat: list) -> float:
    rss_ = rss(y_true, y_hat)
    tss_ = tss(y_true)

    r_squared_ = 1 - (rss_/tss_)
    return(r_squared_)


def adj_r_squared(y_true: list, y_hat: list, model) -> float:
    r2 = r_squared(y_true, y_hat)
    N = len(y_true)
    intercept = 0
    if (model.intercept_ != 0):
        intercept = 1
    p = len(model.coef_)
    adjr2 = 1 - ( ( (1-r2)*(N-intercept) ) / (N - p - intercept) )
    return(adjr2)


def rmse(y_true: list, y_hat: list) -> float:
    residuals = y_true - y_hat
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    return(rmse)
