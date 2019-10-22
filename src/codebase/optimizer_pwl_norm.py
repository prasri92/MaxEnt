'''
Optimizer for piecewise likelihood method
'''
from __future__ import division
import itertools
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Optimizer(object):
    """
    Class summary
    Solves the maximum-entropy optimization problem when given an object
    from the ExtractFeatures class which contains the feature partitions and 
    the feature pairs for the constraints. Optimization algorithm uses the 
    `minimize` function from scipy for finding an optimal set of params.
    Attributes:
        feats_obj: Object from the ExtractFeatures class. Has the necessary 
            feature partitions and pairs for the optimization algorithm.
        opt_sol: List with length equal to the number of partitions in the 
            feature graph. Stores the optimal parameters (thetas) for each 
            partitions.
        norm_z: List with length equal to the number of partitions in the feature
            graph. Stores the normalization constant for each of partitions (since
            each partition is considered independent of others).
        N_s: Matrix of shape |c| x 1 which contains the likelihood of each constraint 
            in the original dataset. 
    """

    def __init__(self, features_object):
        # Init function for the class object
        
        self.feats_obj = features_object
        self.N_s = None
        self.opt_sol = None     
        self.norm_z = None
        self.zero_stats = None 

    # Utility function to check whether a tuple (key from constraint dict)
    # contains all the variables inside the given partition.
    def check_in_partition(self, partition, key_tuple):
        flag = True
        for i in key_tuple:
            if i not in partition:
                flag = False
                break
        return flag

    def compute_zero_stats(self):
        """
            Computes the normalization constant for all zero vectors
        """ 
        #initialize the datasets
        N = self.feats_obj.N        
        data_arr = self.feats_obj.data_arr  

        p = 0

        for i in range(N):
            dvec = data_arr[i]
            if sum(dvec) == 0:
                p += 1
        
        self.zero_stats = 1/(p/N)
        print('Zero vector probability is: ', 1/self.zero_stats)
        

    # This function computes the value of N_s to be used in the objective function.
    # N_s denotes the number of rows satisfying the constraints for each partition/ total number of rows
    def compute_data_stats(self, partition, partition_index):
        """
            partition: a list of feature indices which are present in the constraints. 
            If a diseases is not present in any of the constraints, the marginal probability is taken, 
            and the problem is solved as in the normal case (no piecewise likelihood)

            partition_index: the index of the partition in the final array
            ---------------
            Computes
            -   'p_hat': The data stats vector for the given partition
        """ 
        #initialize the datasets
        N = self.feats_obj.N        
        data_arr = self.feats_obj.data_arr  

        # get constraints for that partition
        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict

        #sanity check for the constraint to be present in the partition
        num_feats = len(partition)
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        len_p_hat = num_feats + num_2wayc + num_3wayc + num_4wayc
        p_hat = np.zeros(len_p_hat)

        # Reverse lookup hashmap for the indices in the partition
        # Useful to make thetas and the constraint_sum match up consistently
        # dvec's (data vector) first index corresponds to the first index in the partition
        # with respect to the original vector (before cropping it out for the
        # partition)
        findpos = {elem:i for i,elem in enumerate(partition)}
      
        for i in range(N):
            dvec = data_arr[i, partition]
            tmp_arr = self.util_compute_p_hat(dvec, partition, findpos, twoway_dict, 
                                    threeway_dict, fourway_dict, num_feats, num_2wayc,
                                    num_3wayc, num_4wayc)

            p_hat += (tmp_arr)
        p_hat /= N

        self.N_s[partition_index] = p_hat
        
    # This function computes the N_s (p_hat) array for every datavector that satisfies each constraint
    def util_compute_p_hat(self, dvec, partition, findpos, twoway_dict, threeway_dict, fourway_dict, 
                            num_feats, num_2wayc, num_3wayc, num_4wayc):

        """Function to compute the sum of marginals for a given input vector for all the constraints. 
        Args:
            dvec: vector to compute N_s for. Note that it should be
            the 'cropped' version of the vector with respect to the partition
            supplied i.e only those feature indices.
            partition: a list of feature indices indicating that they all belong
            in a single partition and we only need to consider them for now.
        """ 
        
        # Just the single feature marginal case --> MLE update
        if len(partition) == 1:
            return dvec[0]  # only the marginal constraint applies here

        def check_condition(key, value):
            # key is a tuple of feature indices
            # value is their corresponding required values
            flag = True
            for i in range(len(key)):
                if dvec[findpos[key[i]]] != value[i]:
                    flag = False
                    break
            return flag

        len_p_hat = num_feats + num_2wayc + num_3wayc + num_4wayc
        feat_arr = np.zeros(len_p_hat)

        # Add up constraint_sum for MARGINAL constraints.
        for i in range(num_feats):
            indicator = 1 if dvec[i] == 1 else 0
            feat_arr[i] += indicator

        j = num_feats
        for key,val in twoway_dict.items():
            if self.check_in_partition(partition, key):
                indicator = 1 if check_condition(key,val) else 0
                feat_arr[j] += indicator
                j += 1

        for key,val in threeway_dict.items():
            if self.check_in_partition(partition, key):
                indicator = 1 if check_condition(key,val) else 0
                feat_arr[j] += indicator
                j += 1

        for key,val in fourway_dict.items():
            if self.check_in_partition(partition, key):
                indicator = 1 if check_condition(key,val) else 0
                feat_arr[j] += indicator
                j += 1

        return feat_arr


    # Function to compute Z_c for all constraints
    def compute_zc(self, partition, partition_index):
        '''
        Args:
            partition: List of feature indices indicating that they all belong
            in the same feature partition
        Computes:
            z_c: The corresponding z value for the constraint
        '''

        # Get all the constraints
        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict

        #sanity check for the constraint to be present in the partition
        num_feats = len(partition)
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        # initialize the z_c vector
        len_z_c = num_feats + num_2wayc + num_3wayc + num_4wayc
        z_c = np.zeros(len_z_c)

        j = 0
        for i in range(num_feats):
            z_c[j] = 1/(1 - self.N_s[partition_index][j])
            j += 1

        for i in range(num_2wayc):
            z_c[j] = 3/(1 - self.N_s[partition_index][j])
            j += 1

        for i in range(num_3wayc):
            z_c[j] = 7/(1 - self.N_s[partition_index][j])
            j += 1

        for i in range(num_4wayc):
            z_c[j] = 15/(1 - self.N_s[partition_index][j])
            j += 1

        self.norm_z[partition_index] = z_c

    # Function to analytically compute thetas 
    def compute_theta(self, partition, partition_index):
        '''
        partition: List of feature indices indicating that they all belong
            in the same feature-partition.
        Computes:
            theta: the corresponding thetas for the given partition
        '''
        # get all constraints belonging to that partition
        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict

        #sanity check for the constraint to be present in the partition
        num_feats = len(partition)
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc
        theta = np.zeros(len_theta)

        j = 0
        for i in range(num_feats):
            theta[j] = np.log(self.N_s[partition_index][j]) + np.log(self.norm_z[partition_index][j])
            j += 1

        for i in range(num_2wayc):
            theta[j] = np.log(self.N_s[partition_index][j]) + np.log(self.norm_z[partition_index][j])
            j += 1

        for i in range(num_3wayc):
            theta[j] = np.log(self.N_s[partition_index][j]) + np.log(self.norm_z[partition_index][j])
            j += 1

        for i in range(num_4wayc):
            theta[j] = np.log(self.N_s[partition_index][j]) + np.log(self.norm_z[partition_index][j])
            j += 1

        self.opt_sol[partition_index] = theta


    def solver_optimize(self):
        """Function to perform the optimization
           uses l-bfgsb algorithm from scipy
        """
        parts = self.feats_obj.feat_partitions
        self.N_s = [None for i in parts]
        self.opt_sol = [None for i in parts]
        self.norm_z = [None for i in parts]
        self.compute_zero_stats()

        for i, partition in enumerate(parts):
            self.compute_data_stats(partition, i)
            self.compute_zc(partition, i)
            self.compute_theta(partition, i)

        # print('N_s is: ', self.N_s)
        # print('Z_c is: ', self.norm_z)
        # print('Theta is: ', self.opt_sol)

    def compute_all_prob(self, num_feats):
        '''
        Function to compute probability of given r vector according to the formula 
        p(r, theta) = Product of all constraints (exp(theta * f_c(r))/Z(theta))
        For every r, find the diseases that appear, find all the constraints in which those diseases appear, 
        and use those datapoints only

        '''
        #Generate all vectors
        all_perms = itertools.product([0,1], repeat=num_feats)

        #initialize p(r)
        p_r = np.ones((2**num_feats))

        # get all constraints
        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict

        # get all partitions
        parts = self.feats_obj.feat_partitions

        #Create matrix of diseases and their positions by parsing once
        # have to do so for every partition, and have to take in the marginal constraints for that partition 
        matrix_constraints = []

        for indx, partition in enumerate(parts):
            feats = partition
            two_wayc = [k for k,v in twoway_dict.items() if self.check_in_partition(partition, k)]
            three_wayc = [k for k,v in threeway_dict.items() if self.check_in_partition(partition,k)]
            four_wayc = [k for k,v in fourway_dict.items() if self.check_in_partition(partition,k)]
           
            for key in feats:
                matrix_constraints.append([key])

            for key in two_wayc:
                matrix_constraints.append([i for i in key])

            for key in three_wayc:
                matrix_constraints.append([i for i in key])

            for key in four_wayc:
                matrix_constraints.append([i for i in key]) 

        # print('Constraints matrix', matrix_constraints)
        '''
        # PROCESS AS AND WHEN NECESSARY 
        # function to check if disease is present in the constraint
        def check_constraint(d):
            positions = []
            for elem in matrix_constraints:
                if d in elem[1]:
                    positions.append(elem[0])
            return positions

        # function to check corresponding values of theta, z_c, and f_c(r)
        def generate_params(dis, indices):
            fc_r,theta,z_c=[],[],[]
            partition = indices[0]
            positions = indices[1:]
            for pos in positions:
                # fc = 1 only if both diseases are present in the constraints 


                # fc = 1 if any 1 of diseases in r is present in the constraint
                # fc = 1 if set(dis).issubset(set(matrix_constraints[pos][1])) else 0 

                fc = 1 if set(dis) == set(matrix_constraints[pos][1]) else 0
                fc_r.append(fc)
                theta.append(self.opt_sol[partition][pos])
                z_c.append(self.norm_z[partition][pos])

            return fc_r, theta, z_c
        '''
        def util_compute_indicator(dindx):
            fc_r = []
            for c in matrix_constraints:
                f = 1 if set(dindx) == set((c)) else 0
                fc_r.append(f)

            return fc_r


        #initialize total_prob
        total_prob = 0
        theta = np.concatenate(self.opt_sol)
        z_c = np.concatenate(self.norm_z)
        # CHECK 
        #Z = total normalization constant (product of all z's, including zero??)
        # z = np.log(np.sum(np.exp(z_c)))
        z = self.zero_stats

        for i, rvec in enumerate(all_perms):
            # find the index of diseases in rvec 
            dindx = [ind for ind,val in enumerate(rvec) if val == 1]
            # print('Disease index:' ,dindx)

            if len(dindx) == 0:
                p_r[i] = 1/self.zero_stats

            else:
                # initialize fc_r, theta, z_c
                fc_r = util_compute_indicator(dindx)
                # print('Indicator:', fc_r)
                p_r[i] = np.exp(np.dot(theta, fc_r))/z

            total_prob += p_r[i]

        # Calculate empirical probabilities
        emp_prob = np.zeros(num_feats + 1)
        for vec in self.feats_obj.data_arr:
            j = sum(vec)
            emp_prob[j] += 1
        emp_prob /= self.feats_obj.data_arr.shape[0]

        print('Total probability: ', total_prob)
        return p_r, emp_prob, total_prob