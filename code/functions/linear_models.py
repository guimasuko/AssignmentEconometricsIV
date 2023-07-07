import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA


def PCA_function(df: pd.DataFrame) -> tuple:
    """
    Function to compute PCs (Z), eigenvectors (gammas), explained variance/eigenvalues (lambdas), explained variance ratio (alphas) and plot the cumulated explained variance ratio

    Parameters:
    df (pd.DataFrame): a dataframe containing variables whose we want to reduce dimension
    
    Returns:
    pcs, gammas, lambdas, alphas
    """


    # PCA function
    pca = PCA() 
    # fit in the data
    pca.fit(df) 
    
    # PCs
    Z = pca.fit_transform(df)
    
    # create the name of PCs: "PC k"
    name_cols = []
    for i in range(len(Z)):
        name_col = f'PC {i+1}'
        name_cols.append(name_col)

    # index (daterange) in the dataframe inputed
    daterange = df.index
    daterange

    # convert pcs into a dataframe
    pcs = pd.DataFrame(Z, index=daterange, columns=name_cols)
    pcs.head()

    # eigenvectors
    gammas = pca.components_
    # eigenvalues or explained variance
    lambdas = pca.explained_variance_
    # explained variance ratio
    alphas = pca.explained_variance_ratio_
    # cumulated explained variance ratio
    cumulated_alphas = pca.explained_variance_ratio_.cumsum()

    # plot of cumulated explained variance ratio
    plt.plot(cumulated_alphas)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('% of Variance Explained')
    plt.title('Cumulative % of Variance Explained');

    return(pcs, gammas, lambdas, alphas)




def OLS_regression(y: pd.DataFrame, X: pd.DataFrame, y_column: str = "PC 1"):
    """
    Function to run an OLS regression

    Parameters:
    y (pd.DataFrame): a dataframe containing the response variable 
    X (pd.DataFrame): a dataframe containing the covariates
    y_column (str): column name in the y dataframe whose will be the response variable
    
    Returns:
    model results
    """

    # response variable
    Y = y[y_column]
    # covariates with constant
    X = sm.add_constant(X)

    # model fit
    model = sm.OLS(Y, X)
    results = model.fit()

    # boxplot with 95% CI
    # coefficients with constant
    coef = list(results.params)
    y = coef[1:]
    x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

    # upper bound for 95% CI 
    upper_bound = results.conf_int(alpha=0.05)[1]
    # lower bound for 95% CI
    lower_bound = results.conf_int(alpha=0.05)[0]

    errors = list(upper_bound - coef)[1:]

    plt.figure()
    plt.errorbar(x, y, yerr=errors, fmt = 'o', color = 'k')
    plt.xticks(( 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31), 
            ( 'MKT', 'HML', 'SMB', 'MOM1', 'MOM36', 'ACC', 'BETA', 'CFP', 'CHCSHO', 'DY', 'EP', 'IDIOVOL', 'CMA', 'UMD', 'RMW', 'RETVOL'))
    plt.xticks(rotation=45)

    plt.xlabel('Anomaly Factors')
    plt.ylabel('Coefficients with 95% CI')
    plt.title(f'{y_column} vs Anomaly Factors');

    return(results)