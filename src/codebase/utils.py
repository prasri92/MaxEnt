import pandas as pd
import numpy as np 

"""Perform all utility taks here
Code is taken from the data_exploration ipython notebook.

TODO:
Data loading modules
Writing clean csv files
Removing missing values, extra columns, NaN etc.
Data filering - removing redunant features using thresholding
Functions to deal with specific files
"""

def load_disease_data_perturb(filePath, perturb_prob):
    """ Creates a numpy array from given csv file

    Creates a pandas dataframe from the given csv file and then exports it to 
    a numpy ndarray. Also perfoms the check where any value > 0 is mapped to 1 
    since it corresponds to a particular disease being prevalent.

    Args:
        filePath: Path to the csv file to load the disease data from
        prob: perturbation probability for 0 to 1

    Returns:
        A binary (0-1) numpy ndarray with each row corresponding to a particular 
        person's disease prevalence data. 
    """
    df = pd.read_csv(filePath)
    tups = [tuple(x) for x in df.values]
    data_arr = np.asarray(tups)
    # Map all positive values to 1 since any > 0 indicates the disease
    data_arr[data_arr > 0] = 1
    
    p_mat = np.random.random(size=data_arr.shape)
    mask_p_idx = (p_mat < perturb_prob)
    data_arr[mask_p_idx] = 1 # If there was 1 prev, no problem, but if there was a 0, map it to 1    
    
    return data_arr



def load_disease_data(filePath):
    """ Creates a numpy array from given csv file

    Creates a pandas dataframe from the given csv file and then exports it to 
    a numpy ndarray. Also perfoms the check where any value > 0 is mapped to 1 
    since it corresponds to a particular disease being prevalent.

    Args:
        filePath: Path to the csv file to load the disease data from

    Returns:
        A binary (0-1) numpy ndarray with each row corresponding to a particular 
        person's disease prevalence data. 
    """
    df = pd.read_csv(filePath)
    tups = [tuple(x) for x in df.values]
    data_arr = np.asarray(tups)

    # Map all positive values to 1 since any > 0 indicates the disease
    data_arr[data_arr > 0] = 1
    return data_arr

def clean_preproc_data(filePath):
    """
    Creates a numpy array of the data given the csv file

    Creates a pandas dataframe from the given csv file and exports it to a numpy ndarray. 
    Checks to see if any disease has zero marginal probability and removes it from the dataframe. 
    Readjusts the number of diseases, reindexes the diseases and returns the dataframe. 

    Input Assumption:
    All data that is fed into the file is the form of 0, 1 with the first row being the header row 
    stating the number of the disease (indexed from 0)

    Args:
        filePath: Path to the csv file to load the disease data
    Returns:
        A binary (0-1) numpy ndarray with each row corresponding to a particular 
        person's disease prevalence data. 
    """
    data=pd.read_csv(filePath, error_bad_lines=False)
    
    #Check if any disease does not occur in the dataset at all, if so, that disease has to be removed
    counts = np.sum(data, axis=0)
    to_drop = list(counts[counts==0].index)
    if len(to_drop)!=0:
        print("Disease " + str(to_drop) + " do not occur. Removing them to proceed")
    data.drop(columns=to_drop, inplace=True)
    new_index = np.arange(len(data.columns))
    new_index = [str(i) for i in new_index]
    data.columns = new_index

    return data

def clean_prepoc_data_nonzeros(filePath):
    """
    Creates a numpy array of the data given the csv file

    Creates a pandas dataframe from the given csv file and exports it to a numpy ndarray. 
    Checks to see if any disease has zero marginal probability and removes it from the dataframe. 
    Readjusts the number of diseases, reindexes the diseases and returns the dataframe. 
    Removes all the zero vectors and returns the dataframe 

    Input Assumption:
    All data that is fed into the file is the form of 0, 1 with the first row being the header row 
    stating the number of the disease (indexed from 0)

    Args:
        filePath: Path to the csv file to load the disease data
    Returns:
        data: A binary (0-1) numpy ndarray with each row corresponding to a particular 
        person's disease prevalence data. 
        prob_zeros: Probability of zero vectors in the dataset 
    """
    data=pd.read_csv(filePath, error_bad_lines=False)

    #Check if any disease does not occur in the dataset at all, if so, that disease has to be removed
    counts = np.sum(data, axis=0)
    to_drop = list(counts[counts==0].index)
    if len(to_drop)!=0:
        print("Disease " + str(to_drop) + " do not occur. Removing them to proceed")
    data.drop(columns=to_drop, inplace=True)
    new_index = np.arange(len(data.columns))
    new_index = [str(i) for i in new_index]
    data.columns = new_index

    #Removing all zero vectors from the data
    all_rows = data.shape[0]
    data = data.loc[~(data==0).all(axis=1)]
    non_zero_rows = data.shape[0]
    prob_zeros = (all_rows-non_zero_rows)/all_rows

    return data, prob_zeros

def write_csv_files_2010_14():
    """Get disease data from the complete 2010-14 csv file and write to csv files

    Function to clean and write the fy, sy and merged csv files from the 
    2010-2014 csv file. Only the fy and sy disease prevalences are extracted 
    from the csv file. All other columns are ignored. 

    Args:
        None

    Returns:
        None
    """ 
    bigFilePath = '../data/2010-2014.csv'
    big_df = pd.read_csv(bigFilePath)
    col_list = big_df.columns
    # print col_list
    print ("Total columns:",  len(col_list))

    print ("Extracting relevant column numbers: ")

    # first disease is 'CCCfy1'
    i = 0
    for i in range(len(col_list)):
        if col_list[i] == 'CCCfy1':
            print (i)
            break

    # last disease is 'CCCfy670'
    j = 0
    for j in range(len(col_list)):
        if col_list[j] == 'CCCfy670':
            print (j)
            break

    first_index = i
    fy_last_index = j
    total_first_year = fy_last_index - first_index + 1
    end_index = first_index + 2 * total_first_year 

    # sanity check: end_index should be the one just after
    # the last disease's sy column
    print (first_index, fy_last_index, end_index)
    print (col_list[first_index], col_list[fy_last_index], col_list[end_index-1])

    # disease_list_2years is the list for all the columns for first year (fy)
    # disease prevalence in the dataset
    disease_list_fy = col_list[first_index:fy_last_index+1]
    print ("Fy disease list:", disease_list_fy)
    
    # disease_list_sy is the list for all the columns for second year (sy)
    # disease prevalence in the dataset
    disease_list_sy = col_list[fy_last_index+1:end_index]
    print ("Sy disease list:", disease_list_sy)


    # disease_list_merge is the list for all the columns for first and second year 
    # (fy and sy) disease prevalence in the dataset
    disease_list_merge = col_list[first_index:end_index]
    print ("Merge disease list:", disease_list_merge)


    df_fy = big_df.filter(disease_list_fy, axis=1)
    df_sy = big_df.filter(disease_list_sy, axis=1)
    df_merge = big_df.filter(disease_list_merge, axis=1)

    print ("Printing the shape of the three data frames:")
    print (df_fy.shape, df_sy.shape, df_merge.shape)

    # Drop the rows who have NaN in certain columns
    df_fy = df_fy.dropna()
    df_sy = df_sy.dropna()
    df_merge = df_merge.dropna()

    print ("Printing the shape of the three data frames after dropping the NaN rows:")
    print (df_fy.shape, df_sy.shape, df_merge.shape)


    print ("Saving the cleaned data frames to csv files")
    fy_csv_file = '../data/2010-2014-fy.csv'
    sy_csv_file = '../data/2010-2014-sy.csv'
    merge_csv_file = '../data/2010-2014-merge.csv'

    df_fy.to_csv(fy_csv_file, encoding='utf-8', index=False)
    df_sy.to_csv(sy_csv_file, encoding='utf-8', index=False)
    df_merge.to_csv(merge_csv_file, encoding='utf-8', index=False)

    return



def write_csv_files_toy_data():
    """Clean and write csv files from toy data set
    Function to clean and write the fy, sy and merged csv files from the 
    50Age toy dataset csv file. Only the fy and sy disease prevalences are 
    extracted from the csv file. All other columns are ignored. 

    Args:
        None

    Returns:
        None
    """ 
    # Loading the toy-dataset
    filePath = '../data/Age50_DataExtract.csv'
    df = pd.read_csv(filePath)

    print (df.columns)

    drop_list = ['fyAGE', 'CCCfy98.1']
    df = df.drop(drop_list, axis=1)

    fy_list = col_list[:9]
    sy_list = col_list[9:]

    df_fy = df.filter(fy_list, axis=1)
    df_sy = df.filter(sy_list, axis=1)

    print ("Saving the clean csv files")

    fname_merge = '../data/Age50_DataExtract_merge.csv'
    fname_fy = '../data/Age50_DataExtract_fy.csv'
    fname_sy = '../data/Age50_DataExtract_sy.csv'

    df.to_csv(fname_merge, encoding='utf-8', index=False)
    df_fy.to_csv(fname_fy, encoding='utf-8', index=False)
    df_sy.to_csv(fname_sy, encoding='utf-8', index=False)

    return
