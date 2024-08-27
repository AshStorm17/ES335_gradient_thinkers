"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from scipy.stats import entropy as scipy_entropy

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(X)
    
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(X.columns))
    
    return encoded_df

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if pd.api.types.is_categorical_dtype(y):
        # If y is categorical, it's not real-valued
        return False

    return np.issubdtype(y.dtype, np.floating)


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    if check_ifreal(Y):
        raise ValueError("Entropy is typically used for discrete values, not continuous.")
    
    label_encoder = LabelEncoder()
    encoded_Y = label_encoder.fit_transform(Y)
    
    value_counts = np.bincount(encoded_Y)
    return scipy_entropy(value_counts, base=2)


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """

    if check_ifreal(Y):
        raise ValueError("Gini index is typically used for discrete values, not continuous.")

    label_encoder = LabelEncoder()
    encoded_Y = label_encoder.fit_transform(Y)
    value_counts = np.bincount(encoded_Y)
    probabilities = value_counts / len(Y)
    return 1.0 - np.sum(probabilities ** 2)

def mse(Y: pd.Series):
    return np.mean((Y - Y.mean()) ** 2)
                   
def mse_gain(Y: pd.Series, groups_list: list):
    total_mse = mse(Y)
    weighted_mse = 0.0
    for group in groups_list:
        weighted_mse += (len(group) / len(Y)) * mse(Y.loc[group])
    return total_mse - weighted_mse       

def info_gain(Y: pd.Series, groups_list: list, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion == 'information_gain':
        total_entropy = entropy(Y)
        weighted_entropy = 0.0
        for group in groups_list:
            weighted_entropy += (len(group) / len(Y)) * entropy(Y.loc[group])
        return total_entropy - weighted_entropy

    elif criterion == 'gini_index':
        total_gini = gini_index(Y)
        weighted_gini = 0.0
        for group in groups_list:
            weighted_gini += (len(group) / len(Y)) * gini_index(Y.loc[group])
        return total_gini - weighted_gini

    else:
        raise ValueError("Criterion must be 'entropy' or 'gini'. ")


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_attribute = None
    best_split_value = None

    # discrete input
    if(check_ifreal(X.iloc[:,0])==0):
        informations = []
        for i in X.columns:
            groups = []
            for j in X[i].unique():
                groups.append(X.index[X[i] == j])

                if(check_ifreal(y)==0): # discrete output
                    informations.append(info_gain(y,groups,criterion))
                else :  # real output
                    informations.append(mse_gain(y,groups))
                    
            best_attribute = X.columns[np.argmax(informations)]
            best_split_value = -1  # threshold split value not used for discrete inputs

    # real input
    else:
        for i in X.columns:
            sorted_values = sorted(X[i].unique())
            for j in range(1, len(sorted_values)):
                curr_split_value = (sorted_values[j - 1] + sorted_values[j]) / 2
                x_less = X[X[i] <= curr_split_value]
                y_less = y[X[i] <= curr_split_value]
                x_greater = X[X[i] > curr_split_value]
                y_greater = y[X[i] > curr_split_value]

                if(check_ifreal(y)==0):  # discrete output
                    best_gain = -float('inf')
                    gain = info_gain(y,[y_greater.index,y_less.index],criterion)
                    if gain > best_gain:
                       best_gain = gain
                       best_attribute = i
                       best_split_value = curr_split_value

                else:  # real output
                    min_mse = float('inf')
                    weighted_mse = ((len(x_less)/len(X[i])) * mse(y_less)) + ((len(x_greater)/len(X[i])) * mse(y_greater))
                    if weighted_mse < min_mse:
                       min_mse = weighted_mse
                       best_attribute = i
                       best_split_value = curr_split_value
        
        return best_attribute, best_split_value


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if check_ifreal(X[attribute]):
        # Real
        x_less = X[X[attribute] <= value].drop(columns=[attribute])
        y_less = y[X[attribute] <= value]
        x_greater = X[X[attribute] > value].drop(columns=[attribute])
        y_greater = y[X[attribute] > value]
        return x_less, y_less, x_greater, y_greater
    
    else:
        # Discrete
        splits = []
        for i in X[attribute].unique():
            sub_X = X[X[attribute] == i].drop(columns=[attribute])
            sub_y = y[X[attribute] == i]
            splits.append((sub_X, sub_y, i))
        
        return splits