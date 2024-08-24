"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class DiscreteNode :
    def __init__(self,feature):
        self.feature = feature
        self.daughter = {}

class RealNode :
    def __init__(self,feature,split_value):
        self.feature = feature
        self.split_value = split_value
        self.daughter = {'Less than' : None,'Greater than' : None}  

@dataclass
class TreeNode:
    feature: str = None
    split_value: float = None
    left: 'TreeNode' = None
    right: 'TreeNode' = None
    value: any = None


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
    current_depth = 0
    def fit(self, X: pd.DataFrame, y: pd.Series, current_depth) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal feature to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 
        

        # real input
        if check_ifreal(X.iloc[:,0]):

            if((current_depth == self.max_depth) or (len(y.unique())==1) or (X.shape[1]==0)) :
                if check_ifreal(y) : return y.mean()
                else: return y.mode()[0]
                
            best_feature, split_value = opt_split_attribute(X, y, self.criterion, X.columns)
            curr = RealNode(best_feature,split_value)
            x_less,y_less,x_greater,y_greater = split_data(X,y,best_feature,split_value)
            curr.daughter['Less than'] = self.fit(x_less,y_less,current_depth+1)
            curr.daughter['Greater than'] = self.fit(x_greater,y_greater,current_depth+1)

            return curr

        # discrete input
        else:

            if((current_depth == self.max_depth) or (len(y.unique())==1) or (X.shape[1]==0)) :
                if check_ifreal(y) : return y.mean()
                else: return y.mode()[0]
                
            best_feature, split_value = opt_split_attribute(X, y, self.criterion, X.columns)
            curr = DiscreteNode(best_feature)
            splits = split_data(X,y,best_feature,split_value)
            for i in splits:
                 curr.daughter[i[2]] = self.fit(i[0],i[1],current_depth+1)

            return curr
            

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.


        pass

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
