import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from metrics import *
from tree.utils import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, sep='\s+', header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)
data['horsepower'] = data['horsepower'].astype(float)
data.drop(['car name'], axis=1, inplace=True)
origin_encoded = one_hot_encoding(data[['origin']])
data = pd.concat([data.drop('origin', axis=1), origin_encoded], axis=1)

X = data.drop('mpg', axis=1) 
y_target = data['mpg']  
y_target = (y_target > y_target.median()).astype(int)  

break_point = int(0.7 * len(data))
X_train = X.iloc[:break_point]
y_train = y_target.iloc[:break_point]
X_test = X.iloc[break_point:]
y_test = y_target.iloc[break_point:]

decisiontree = DecisionTree(criterion="information_gain")
decisiontree.fit(X_train, y_train,0)
predictions = decisiontree.predict(X_test)

# using decision tree module from scikit learn
scikitdecisiontree = DecisionTreeClassifier(criterion="entropy")  
scikitdecisiontree.fit(X_train, y_train,0)
scikit_predictions = scikitdecisiontree.predict(X_test)

print(f"Our decision tree accuracy: {accuracy(predictions, y_test)}")
print(f"Scikit learn decision tree accuracy: {accuracy(scikit_predictions, y_test)}")
print(f"Our decision tree Precision: {precision(predictions, y_test, cls=1)}")
print(f"Scikit learn decision tree Precision: {precision(scikit_predictions, y_test, cls=1)}")
print(f"Our decision tree Recall: {recall(predictions, y_test, cls=1)}")
print(f"Scikit learn decision tree Recall: {recall(scikit_predictions, y_test, cls=1)}")