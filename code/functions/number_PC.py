import numpy as np



def rule_thumb(alphas: list) -> int:
    """
    Function to compute number of PCs based on the Rule of Thumb

    Parameters:
    alphas (list): list with the explained variance ratio of the PCs

    Returns:
    n_pc (int): number of PCs
    """

    # rule
    rule = alphas >= 0.03
    # compute the number of alphas whose respects the rule
    n_pc = sum(rule)

    # cumulated explained variance ratio
    cumulated_alphas = alphas.cumsum()

    # how much of variance these pcs explain 
    variance_explained = round(cumulated_alphas[n_pc - 1]*100, 2)
    print(f'The first {n_pc} PCs explain {variance_explained}% of the returns variance.')
    
    return(n_pc)




def informal_way(alphas: list) -> int:
    """
    Function to compute number of PCs based on the Informal Way

    Parameters:
    alphas (list): list with the explained variance ratio of the PCs

    Returns:
    n_pc (int): number of PCs
    """

    # cumulated explained variance ratio
    cumulated_alphas = alphas.cumsum()

    # rule
    rule = cumulated_alphas < 0.9
    # compute the number of alphas whose respects the rule
    n_pc = sum(rule)


    # how much of variance these pcs explain 
    variance_explained = round(cumulated_alphas[n_pc - 1]*100, 2)
    print(f'The first {n_pc} PCs explain {variance_explained}% of the returns variance.')
    
    return(n_pc)




def biggest_drop(lambdas: list) -> int:
    """
    Function to compute number of PCs based on the Biggest Drop

    Parameters:
    lambdas (list): list with the explained variance (eigenvectors) of the PCs

    Returns:
    n_pc (int): number of PCs
    """

    # explained variance ratio
    alphas = lambdas/sum(lambdas)
    # cumulated explained variance ratio
    cumulated_alphas = alphas.cumsum()

    # compute and storage the vector with lambda_j/lambda_{j+1} ratio
    r_vector = []
    for i in range(len(lambdas)-1):
        r_i = lambdas[i]/lambdas[i+1]
        r_vector.append(r_i)

    # the last lambda is almost zero, so the last r_i is very large (should we drop?)
    r_vector_ = r_vector[:-1]

    # number de pcs = arg max (add 1 because the index in python starts in 0)
    n_pc = np.argmax(r_vector_) + 1
    n_pc


    # how much of variance these pcs explain 
    variance_explained = round(cumulated_alphas[n_pc - 1]*100, 2)
    print(f'The first {n_pc} PCs explain {variance_explained}% of the returns variance.')
    
    return(n_pc)