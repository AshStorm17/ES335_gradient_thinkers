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
        self.root = None
    def fit(self, X: pd.DataFrame, y: pd.Series, current_depth) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal feature to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 
        
        if((current_depth == self.max_depth) or (len(y.unique())==1) or (X.shape[1]==0)) :
                if check_ifreal(y) : return y.mean()
                else: return y.mode()[0]

        # real input
        if check_ifreal(X.iloc[:,0]):
            self.case = 0    
            best_feature, split_value = opt_split_attribute(X, y, self.criterion, X.columns)
            curr = RealNode(best_feature,split_value)
            x_less,y_less,x_greater,y_greater = split_data(X,y,best_feature,split_value)
            curr.daughter['Less than'] = self.fit(x_less,y_less,current_depth+1)
            curr.daughter['Greater than'] = self.fit(x_greater,y_greater,current_depth+1)

            if check_ifreal(y) : curr.daughter['default'] = y.mean()
            else : curr.daughter['default'] = y.mode()[0]
            if current_depth == 0 : self.root = curr
            return curr

        # discrete input
        else:
            self.case = 1  
            best_feature, split_value = opt_split_attribute(X, y, self.criterion, X.columns)
            curr = DiscreteNode(best_feature)
            splits = split_data(X,y,best_feature,split_value)
            for i in splits:
                 curr.daughter[i[2]] = self.fit(i[0],i[1],current_depth+1)

            if check_ifreal(y) : curr.daughter['default'] = y.mean()
            else : curr.daughter['default'] = y.mode()[0]
            if current_depth == 0 : self.root = curr
            return curr
            

    def predict(self, X: pd.DataFrame) -> pd.Series:
        series = []

        for x in X.index:
            curr = self.root
            while isinstance(curr, (RealNode, DiscreteNode)):
                if isinstance(curr, RealNode):
                    if X.loc[x][curr.feature] <= curr.split_value:
                        curr = curr.daughter.get('Less than')
                    else: 
                        curr = curr.daughter.get('Greater than')
                elif isinstance(curr, DiscreteNode):
                    if X.loc[x][curr.feature] in curr.daughter:
                        curr = curr.daughter[X.loc[x][curr.feature]]
                    else:
                        curr = curr.daughter.get('default')
                
                # If the current node is a leaf node (value is not None), break the loop
                if curr is not None and not isinstance(curr, (RealNode, DiscreteNode)):
                    break

            series.append(curr if curr is not None else 0) # Return a default label or error handling
        return pd.Series(series, index=X.index, name='Prediction')


    def plot(self) -> None:
        """
        Function to plot the tree.

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """

        def print_tree(node, indent):
            if isinstance(node, RealNode):
                # Print the decision rule for real-valued features
                print(f'{" " * indent}?({node.feature} > {node.split_value})')
                print_tree(node.daughter['Less than'], indent + 4)
                print(f'{" " * indent}N: ')
                print_tree(node.daughter['Greater than'], indent + 4)
                
            elif isinstance(node, DiscreteNode):
                # Print the decision rule for discrete features
                print(f'{" " * indent}?{node.feature}')
                for key, child_node in node.daughter.items():
                    if key != 'default':  
                        print(f'{" " * indent}{key}: ')
                        print_tree(child_node, indent + 4)
                if 'default' in node.daughter:
                    print(f'{" " * indent}default: {node.daughter["default"]}')

            else:
                # Print the leaf node value
                print(f'{" " * indent}Class: {node}')

        if self.root is None:
            print("The tree is empty.")
        else:
            print_tree(self.root, 0)
    
    def visualize_tree(self, graph=None, parent_id=None, connection_label=None):

        if graph is None:
            graph = graphviz.Digraph(format='png')

        current_label = self.feature

        graph.node(str(id(self)), current_label, shape=self.shape)

        if hasattr(self, 'decision') and self.decision is not None:
            current_label += f"\nDecision: {self.decision}"
    
        if parent_id is not None:
            if connection_label is not None:
                graph.edge(str(parent_id), str(id(self)), label=connection_label)
            else:
                graph.edge(str(parent_id), str(id(self)))

        if hasattr(self, 'children'):
            for child in self.children:
                child.visualize_tree(graph, str(id(self)), connection_label=str(child.val))

        return graph

