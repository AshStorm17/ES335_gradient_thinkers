from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    correct_predictions = (y_hat == y).sum()
    accuracy = correct_predictions / y.size
    
    return accuracy


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    predicted_positive = (y_hat == cls).sum()
    
    if predicted_positive == 0:
        return 0.0  
    
    precision = true_positive / predicted_positive
    return precision


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    total_positive = (y==cls).sum()
    if(total_positive ==0 ):
        return 0.0
    
    recall = true_positive/total_positive


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    return (((y-y_hat)**2).mean())**0.5
    


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    return abs(y-y_hat).mean()
    
