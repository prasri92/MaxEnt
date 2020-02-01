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
print("# of zero atoms incorrectly classified using the approximate method")
print(zeros_compare.where(zeros_compare.difference != 0).difference.count())

res = zeros_compare.groupby('#_diseases')['difference'].agg(zeros_compare)
print(res)
# print(zeros_compare[:20])