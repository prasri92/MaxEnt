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

    # Utility function to check whether a tuple (key from constraint dict)
    # contains all the variables inside the given partition.
    def check_in_partition(self, partition, key_tuple):
        flag = True
        for i in key_tuple:
            if i not in partition:
                flag = False
                break
        return flag

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
        # get constraints for that partition
        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict

        #sanity check for the constraint to be present in the partition
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        len_p_hat = num_2wayc + num_3wayc + num_4wayc
        p_hat = np.zeros(len_p_hat)

        # Reverse lookup hashmap for the indices in the partition
        # Useful to make thetas and the constraint_sum match up consistently
        # dvec's (data vector) first index corresponds to the first index in the partition
        # with respect to the original vector (before cropping it out for the
        # partition)
        findpos = {elem:i for i,elem in enumerate(partition)}

        N = self.feats_obj.N        
        data_arr = self.feats_obj.data_arr        
        for i in range(N):
            dvec = data_arr[i, partition]
            tmp_arr = self.util_compute_p_hat(dvec, partition, twoway_dict, 
                                    threeway_dict, fourway_dict, findpos)
            
            p_hat += (tmp_arr/N)

        self.N_s[partition_index] = p_hat
        
    # This function computes the N_s (p_hat) array for every datavector that satisfies each constraint
    def util_compute_p_hat(self, dvec, partition, twoway_dict, threeway_dict, fourway_dict, findpos):

        """Function to compute the sum of marginals for a given input vector for all the constraints. 
        Args:
            dvec: vector to compute N_s for. Note that it should be
            the 'cropped' version of the vector with respect to the partition
            supplied i.e only those feature indices.
            partition: a list of feature indices indicating that they all belong
            in a single partition and we only need to consider them for now.
        """ 
        
        #TODO: CHECK WHAT SHOULD BE DONE FOR A SINGLE DISEASE 
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

        #sanity check for the constraint to be present in the partition
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        len_p_hat = num_2wayc + num_3wayc + num_4wayc
        feat_arr = np.zeros(len_p_hat)

        j = 0
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

        # TODO: if partition length is 1 
        if len(partition) == 1:
            pass

        # Get all the constraints
        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict

        #sanity check for the constraint to be present in the partition
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        # initialize the z_c vector
        len_z_c = num_2wayc + num_3wayc + num_4wayc
        z_c = np.zeros(len_z_c)

        j = 0
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

        # TODO: 
        if len(partition) == 1:
            pass

        # get all constraints belonging to that partition
        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict

        #sanity check for the constraint to be present in the partition
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        len_theta = num_2wayc + num_3wayc + num_4wayc
        theta = np.zeros(len_theta)

        j = 0
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

        for i, partition in enumerate(parts):

            # TODO: Check if partition is 1, how to process
            if len(partition) == 1:
                pass

            else:
                self.compute_data_stats(partition, i)
                self.compute_zc(partition, i)
                self.compute_theta(partition, i)

        print('N_s is: ', self.N_s)
        print('Z_c is: ', self.norm_z)
        print('Theta is: ', self.opt_sol)

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

        #sanity check for the constraint to be present in the partition
        num_2wayc = len([1 for k,v in twoway_dict.items()])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items()]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items()]) # num of 4way constraints for the partition
        
        # get all partitions
        parts = self.feats_obj.feat_partitions

        #Create matrix of diseases and their positions by parsing once. 
        matrix_constraints = np.zeros((num_2wayc+num_3wayc+num_4wayc, 2), dtype=object)

        j=0
        for key in twoway_dict.keys():
            matrix_constraints[j][0] = j
            matrix_constraints[j][1] = key
            j+=1
        for key in threeway_dict.keys():
            matrix_constraints[j][0] = j
            matrix_constraints[j][1] = key
            j+=1
        for key in fourway_dict.keys():
            matrix_constraints[j][0] = j
            matrix_constraints[j][1] = key
            j+=1   

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
                # fc = 1 if any 1 of diseases in r is present in the constraint
                # fc = 1 if set(dis).issubset(set(matrix_constraints[pos][1])) else 0 

                # fc = 1 only if both diseases satisfy the constraint 
                fc = 1 if set(dis) == set(matrix_constraints[pos][1]) else 0
                fc_r.append(fc)
                theta.append(self.opt_sol[partition][pos])
                z_c.append(self.norm_z[partition][pos])

            return fc_r, theta, z_c

        #initialize total_prob
        total_prob = 0

        for i, rvec in enumerate(all_perms):
            # find the index of diseases in rvec 
            dindx = [ind for ind,val in enumerate(rvec) if val == 1]

            # initialize fc_r, theta, z_c
            fc_r = []
            theta = []
            z_c = []

            # for each disease present in rvec, find the partition in which that disease belongs to, 
            # find the partition number, index for theta and Z(theta) within that partition 
            # find the index in the two_way, three_way and four_way
            indices = {}
            for d in dindx:
                indices[d] = [ind for ind,val in enumerate(parts) if d in val]
                indices[d].extend(check_constraint(d))

            for k,v in indices.items():
                f,t,z = generate_params(dindx, v)
                fc_r.extend(f)
                theta.extend(t)
                z_c.extend(z)

            for j in range(len(theta)):
                p_r[i] *= (np.exp(theta[j]*fc_r[j]))/z_c[j]

            total_prob += p_r[i]

        print('Probabilities are: ', p_r)
        print('Total probability: ', total_prob)
        return p_r




        





            


