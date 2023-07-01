import pandas as pd
from functions.metrics import r_squared, adj_r_squared, rmse
from sklearn.linear_model import LinearRegression

def OLS_metrics(y: pd.DataFrame, X: pd.DataFrame, intercept: bool = True):
    R2 = []
    adjR2 = []
    RMSE = []

    for col in y.columns:
        y_ = y[col]

        reg = LinearRegression(fit_intercept=intercept)
        reg.fit(X, y_)

        pred = reg.predict(X)
        r2 = r_squared(y_, pred)*100
        adjr2 = adj_r_squared(y_, pred, reg)*100
        rmse_ = rmse(y_, pred)

        R2.append(r2)
        adjR2.append(adjr2)
        RMSE.append(rmse_)
    
    df = pd.DataFrame({'R2':R2,
                       'Adjusted R2':adjR2,
                       'RMSE':RMSE},
                       index=y.columns)
    
    return(df)