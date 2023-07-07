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
    for i in range(Z.shape[1]):
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




def OLS_regression(y: pd.DataFrame, X: pd.DataFrame, y_column: str = "PC 1", is_pc: bool = False):
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
    X_ = sm.add_constant(X)

    # model fit
    model = sm.OLS(Y, X_)
    results = model.fit()

    # boxplot with 95% CI
    # coefficients with constant
    coef = list(results.params)
    y = coef[1:]
    loc_x = list(range(1, len(X.columns)*2, 2))

    # upper bound for 95% CI 
    upper_bound = results.conf_int(alpha=0.05)[1]
    # lower bound for 95% CI
    lower_bound = results.conf_int(alpha=0.05)[0]

    errors = list(upper_bound - coef)[1:]

    plt.figure()
    plt.errorbar(loc_x, y, yerr=errors, fmt = 'o', color = 'k')

    xticks = list(X.columns)
    plt.xticks(loc_x, 
               xticks)
    plt.xticks(rotation=45)

    if(is_pc == False):
        cov_name = "Anomaly Factors"        
    elif(is_pc == True):
        cov_name = "Anomaly-Based Principal Components"

    plt.xlabel(f'{cov_name}')
    plt.ylabel('Coefficients with 95% CI')
    plt.title(f'{y_column} vs {cov_name}');

    return(results)