import numpy as np
import pandas as pd 
import sys 
import itertools

path_to_codebase = './codebase/'
sys.path.insert(0, path_to_codebase)
from codebase.utils import clean_preproc_data

def get_emp_prob(cleaneddata):
        """
        Find the empirical probabilities of all 2**k disease combinations
        Args:
            cleaneddata: the pandas Dataframe of all patient vectors
        Returns: 
            emp: probability distribution vector for all combinations 
        """
        #ignore warnings for pandas dataframe handling 
        pd.options.mode.chained_assignment = None  # default='warn'

        data = cleaneddata
        size = data.shape[0]
        diseases = data.shape[1]
        cols = np.arange(diseases)
        data.columns = cols

        # initialize b_eq
        b_eq = []
        
        all_perms = list(itertools.product([0,1], repeat=diseases))

        #ndata = new data 
        ndata = pd.DataFrame()
        ndata[all_perms[0]] = np.logical_not(np.any(data, axis=1))*1
        b_eq.append(np.sum(ndata[all_perms[0]])/size)

        for perm in all_perms[1:]:
            ones = [i for i,x in enumerate(perm) if perm[i]==1]
            sub_data = data[ones]
            sub_data['m'] = np.all(sub_data,axis=1)*1
            t = np.sum(sub_data['m'], axis=0)
            m = t/size
            b_eq.append(m)

        print(b_eq)

# generating synthetic data 
directory = '../dataset/d15/synthetic_data_expt1.csv'

# for synthetic data 
cleaneddata = clean_preproc_data(directory)
get_emp_prob(cleaneddata)