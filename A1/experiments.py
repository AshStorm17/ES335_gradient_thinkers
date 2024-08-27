import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 1  # Number of times to run each experiment to calculate the average values

# Function to create fake data (take inspiration from usage.py)
# ...
def RealReal(N,M):
    X = pd.DataFrame(np.random.randn(N, M))
    y = pd.Series(np.random.randn(N))
    return X,y
def RealDiscrete(N,M):
    X = pd.DataFrame(np.random.randn(N, M))
    y = pd.Series(np.random.randint(2, size=N), dtype="category")
    return X,y
def DiscreteDiscrete(N,M):
    X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})
    y = pd.Series(np.random.randint(2, size=N), dtype="category")
    return X,y
def DiscreteReal(N,M):
    X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})
    y = pd.Series(np.random.randn(N))
    return X,y

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
def time_taken(num_average_time, n_samples_list, n_features_list):
    results = []
    for N in n_samples_list:
        for M in n_features_list:
            types = [
                ("Real_input_Real_output", RealReal),
                ("Real_input_Discrete_output", RealDiscrete),
                ("Discrete_input_Discrete_output", DiscreteDiscrete),
                ("Discrete_input_Real_output", DiscreteReal),
            ]          
            for type_name, data_create_type in types:
                learn_time = []
                predict_time = []
                
                for k in range(num_average_time):
                    X, y = data_create_type(N, M)
                    decisiontree = DecisionTree(criterion="information_gain")

                    start_time = time.time()
                    decisiontree.fit(X, y,0)
                    learn_time.append(time.time() - start_time)

                    start_time = time.time()
                    decisiontree.predict(X)
                    predict_time.append(time.time() - start_time)

                avg_learn_time = np.mean(learn_time)
                avg_predict_time = np.mean(predict_time)
                
                results.append({
                    "N": N,
                    "M": M,
                    "Type": type_name,
                    "Average Learn Time (s)": avg_learn_time,
                    "Average Predict Time (s)": avg_predict_time,
                })
    
    return pd.DataFrame(results)


n_samples_list = [50, 100, 500, 1000]  
n_features_list = [5, 10, 20]  

results_df = time_taken(num_average_time, n_samples_list, n_features_list)
print(results_df)

# Function to plot the results
# ...
def plot_results(results_df):
    plt.figure(figsize=(12, 6))
    for data_type in results_df["Type"].unique():
        subset = results_df[results_df["Type"] == data_type]
        plt.plot(subset["N"], subset["Average Learn Time (s)"], label=f"{data_type} - Learn Time", marker='o')

    plt.title("Average Learn Time vs. Number of Samples (N)")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Average Learn Time (s)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    for data_type in results_df["Type"].unique():
        subset = results_df[results_df["Type"] == data_type]
        plt.plot(subset["N"], subset["Average Predict Time (s)"], label=f"{data_type} - Predict Time", marker='o')

    plt.title("Average Predict Time vs. Number of Samples (N)")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Average Predict Time (s)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
plot_results(results_df)
