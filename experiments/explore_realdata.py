import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

def read_data(filePath):
    """
    Function to read the data as pandas DataFrame and clean it to perform a curvefit
    """
    df = pd.read_csv(filePath)
    
    #Check if any disease does not occur in the dataset at all, if so, that disease has to be removed
    counts = np.sum(df, axis=0)
    to_drop = list(counts[counts==0].index)
    if len(to_drop)!=0:
        print("Disease " + str(to_drop) + " do not occur. Removing them to proceed")
        df.drop(columns=to_drop, inplace=True)
        new_index = np.arange(len(df.columns))
        new_index = [str(i) for i in new_index]
        df.columns = new_index
    
    tups = [tuple(x) for x in df.values]
    data_arr = np.asarray(tups)
    data_arr = data_arr.astype(int)

    # Map all positive values to 1 since any > 0 indicates the disease
    data_arr[data_arr > 0] = 1
    return df, data_arr

def get_data(data):
    """
    Get data array from the pandas DataFrame
    """
    tups = [tuple(x) for x in data.values]
    data_arr = np.asarray(tups)
    data_arr = data_arr.astype(int)

    # Map all positive values to 1 since any > 0 indicates the disease
    data_arr[data_arr > 0] = 1
    return data_arr

def emp_prob(data_arr, plot=False):
    """
    Given data array, return the empirical probabilities of the data
    """
    num_dis = data_arr.shape[1]
    N = data_arr.shape[0]
    emp_prob = np.zeros(num_dis + 1)
    for vec in data_arr:
        j = sum(vec)
        emp_prob[j] += 1

    emp_prob /= N

    if plot==True:
        plt.plot(np.arange(0, num_dis+1), emp_prob)
        plt.xlabel('Number of diseases')
        plt.ylabel('Proportion of people having x number of diseases')
        plt.show()

    return emp_prob

def curve_fitting(data_arr, plot=False):    
    """
    Function to generate the curve of number of diseases by each patient, and return best 
    matching lambda for the exponential function
    """
    num_dis = data_arr.shape[1]
    x = np.arange(0, num_dis+1)
    y = emp_prob(data_arr)
    def func(x, l):
        return l * np.exp(-l * x) 

    popt, pcov = curve_fit(func, x, y)

    if plot == True:
        plt.figure()
        plt.plot(x, y, 'k*', label="Original Data")
        plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
        plt.xlabel("Number of diseases")
        plt.ylabel("Proportion of people having x number of diseases")
        plt.legend()
        plt.show()

    return popt[0]

def create_subsets(df, n_subs=None, d=None, row=None, topd=None):
    """
    Function to create subsets of data from the whole dataset
    Args:
        df: DataFrame containing the whole dataset
        d: No of diseases
        row: No of patients
        topd: Top d occurring diseases 
        n: No of subsets 
    """
    if topd != None:
        counts = np.sum(df, axis=0)
        counts = counts.sort_values(ascending=False)[:topd]
        cols = counts.index
        topd_data = df[cols]
        new_cols = np.arange(len(topd_data.columns))
        new_cols = [str(i) for i in new_cols]
        topd_data.columns = new_cols
        topd_data.reset_index(drop=True, inplace=True)
        topd_data.to_csv('../dataset/real_subsets/top_'+str(topd)+'d.csv')

    else:
        for i in range(n_subs):
            d_sub = df.sample(n=d, axis=1, replace=False)
            new_cols = np.arange(len(d_sub.columns))
            new_cols = [str(i) for i in new_cols]
            d_sub.columns = new_cols
            d_sub.reset_index(drop=True, inplace=True)
            d_sub.to_csv('../dataset/real_subsets/'+str(d)+'d_'+str(i)+'.csv')

def calc_lambda_range(df, n_subs, d, row=None):
    """
    Function to calculate the lambda range for synthetic data generation 
    """
    subsets = {}
    for i in range(n_subs):
        # subsets[i] = df.sample(n=d, axis=1, replace=False).sample(n=row, axis=0, replace=False)
        subsets[i] = df.sample(n=d, axis=1, replace=False)
    
    lambdas = np.zeros(n_subs)

    for i in range(n_subs):
        data_arr = get_data(subsets[i])
        sum_dis = emp_prob(data_arr)
        lambdas[i] = curve_fitting(data_arr)
        
    avg_lambda = np.mean(lambdas)
    print("Average lambda is: ", avg_lambda)


def create_data():
    df, data_array_full = read_data('../dataset/2010-2014-fy.csv')
    # create_subsets(df, topd=20)
    # create_subsets(df, topd=15)
    # create_subsets(df, topd=10)
    # create_subsets(df, topd=7)
    # create_subsets(df, n_subs=10, d=20)
    create_subsets(df, topd=4)
    create_subsets(df, n_subs=1, d=4)

def main():
    create_data()
    # df, data_array_full = read_data('../dataset/2010-2014-fy.csv')
    # calc_lambda_range(df, n_subs=250, d=20)

main()
