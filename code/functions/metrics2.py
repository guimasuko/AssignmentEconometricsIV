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


def rmse(y_true: list, y_hat: list) -> float:
    residuals = y_true - y_hat
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    return(rmse)
