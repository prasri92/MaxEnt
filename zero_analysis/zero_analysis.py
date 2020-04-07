import pandas as pd
import numpy as np 
import csv

z_exact = pd.read_csv('zero_results.csv',index_col=0)
z_approx = pd.read_csv('zero_results_approx.csv',index_col=0)

zeros_compare = z_exact.loc[:,['#_diseases','dataset_num','file_num','tot_zeros']]
zeros_compare.rename(columns={'tot_zeros':'exact_zeros'}, inplace=True)
z_approx = z_approx.loc[:, ['tot_zeros']]
z_approx.rename(columns={'tot_zeros':'approx_zeros'}, inplace=True)

zeros_compare = pd.concat([zeros_compare, z_approx], axis=1)

zeros_compare['difference'] = zeros_compare['approx_zeros'] - zeros_compare['exact_zeros']

zeros_compare.replace({'file_num': {1: 1.25, 2: 0.8, 3:0.63, 4:0.5, 5:0.42}}, inplace=True)
zeros_compare.rename(columns={'file_num':'exponent'},inplace=True)

# Q1: What is the average number of zeros using the exact zero detection method for each number of diseases? 
print("Average number of zeros (using exact method)")
print(zeros_compare.groupby('#_diseases')['exact_zeros'].mean())

# Q2: What is the average number of zeros using the approximate zero detection method for each number of diseases? 
print("Average number of zeros (using approximate method)")
print(zeros_compare.groupby('#_diseases')['approx_zeros'].mean())

# Q3: What is the mean difference in the number of zeros in the exact and approximate methods
print("Average difference between the exact and approximate method")
print(zeros_compare.groupby('#_diseases')['difference'].mean())

# Q4: How many zero atoms are wrongly identified in the approximate detection problem? 
print("# of times zero atoms incorrectly classified using the approximate method")
print(zeros_compare.groupby('#_diseases')['difference'].apply(lambda x: x[x > 0].count()))

zeros_compare.to_csv('zero_analysis_small.csv')