from __future__ import division
import itertools
from collections import defaultdict

import numpy as np
from scipy.optimize import fmin_l_bfgs_b as spmin_LBFGSB


"""
- opt steps --- numpy dot products
"""


class RobustOptimizer(object):
    """ Class summary
    Solves the maximum-entropy optimization problem when given an object
    from the ExtractFeatures class which contains the feature paritions and 
    the feature pairs for the constraints. Optimization algorithm uses the 
    `fmin_l_bfgs_b` function from scipy for finding an optimal set of params.


    Attributes:
        feats_obj: Object from the ExtractFeatures class. Has the necessary 
            feature partitions and pairs for the optimization algorithm.
        opt_sol: List with length equal to the number of partitions in the 
            feature graph. Stores the optimal parameters (thetas) for each 
            partitions.
        norm_z: List with length equal to the number of partitions in the feature
            graph. Stores the normalization constant for each of partitions (since
            each partition is considered independent of others).
    """

    def __init__(self, features_object, lamb):
        # Init function for the class object
        
        self.feats_obj = features_object
        self.opt_sol = None     
        self.norm_z = None
        self.scaling_factor = lamb
        

    # Utility function to check whether a tuple (key from constraint dict)
    # contains all the variables inside the given partition.
    def check_in_partition(self, partition, key_tuple):
        flag = True
        for i in key_tuple:
            if i not in partition:
                flag = False
                break
        return flag


    # This function computes the inner sum of the 
    # optimization function objective    
    # could split thetas into marginal and specials
    def compute_constraint_sum(self, thetas, rvec, partition):
        """Function to compute the inner sum for a given input vector. 
        The sum is of the product of the theta (parameter) for a particular
        constraint and the indicator function for that constraint and hence the
        sum goes over all the constraints. Note that probability is
        not calculated here. Just the inner sum that is exponentiated
        later.

        Args:
            thetas: list of the maxent paramters
            
            rvec: vector to compute the probability for. Note that it should be
            the 'cropped' version of the vector with respect to the partition
            supplied i.e only those feature indices.

            partition: a list of feature indices indicating that they all belong
            in a single partition and we only need to consider them for now.

        """ 

        # thetas is ordered as follows: 
        # (1) all the marginal constraints
        # (2) all the two-way constraints
        # (3) all the three-way constraints
        # (4) all the four-way constraints

        # Just the single feature marginal case --> MLE update
        if len(partition) == 1:
            constraint_sum = 0.0
            if rvec[0] == 1:
                constraint_sum += thetas[0]
            
            # print(partition,  constraint_sum)
            return constraint_sum

        
        constraint_sum = 0.0
        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict
       
        # Sanity Checks for the partition and the given vector
        num_feats = len(partition)  # number of marginal constraints
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        assert len(rvec) == num_feats        
        assert len(thetas) == num_feats + num_2wayc + num_3wayc + num_4wayc
        
        # CHECKING WITH 1 since BINARY FEATURES
        # Add up constraint_sum for MARGINAL constraints.
        for i in range(num_feats):
            indicator = 1 if rvec[i] == 1 else 0
            constraint_sum += thetas[i] * indicator
        
        # Reverse lookup hashmap for the indices in the partition
        # Useful to make thetas and the constraint_sum match up consistently
        # rvec's first index corresponds to the first index in the partition
        # with respect to the original vector (before cropping it out for the
        # partiton)
        findpos = {elem:i for i,elem in enumerate(partition)}
        
        def check_condition(key, value):
            # key is a tuple of feature indices
            # value is their corresponding required values
            flag = True
            for i in range(len(key)):
                if rvec[findpos[key[i]]] != value[i]:
                    flag = False
                    break
            return flag

        j = 0
        twoway_offset = num_feats
        for key,val in twoway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                constraint_sum += thetas[twoway_offset + j] * indicator
                j += 1

        j = 0
        threeway_offset = twoway_offset + num_2wayc
        for key,val in threeway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                constraint_sum += thetas[threeway_offset + j] * indicator
                j += 1

        j = 0
        fourway_offset = threeway_offset + num_3wayc
        for key,val in fourway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                constraint_sum += thetas[fourway_offset + j] * indicator
                j += 1


        # Thetas is still a contiguous across the marginals and the 2way, 3way
        # and the 4way constraints for a given partiton

        return constraint_sum


    # normalization constant Z(theta)
    # assuming binary features for now.
    def binary_norm_Z(self, thetas, partition):
        """Computes the normalization constant Z(theta) for a given partition

        Args:
            thetas: The parameters for the given partition

            partition: List of feature indices indicating that they all belong
            in the same feature-partition.
        """
        norm_sum = 0.0       
        num_feats = len(partition)
       
        if num_feats == 1:
            norm_sum = 1 + np.exp(thetas[0])
            return norm_sum

        # Create all permuatations of a vector belonging to that partition
        all_perms = itertools.product([0, 1], repeat=num_feats)
        for vec in all_perms:
            tmpvec = np.asarray(vec)
            tmp = self.compute_constraint_sum(thetas, tmpvec, partition)
            norm_sum += np.exp(tmp)
        
        return norm_sum


    def solver_optimize(self):
        """Function to perform the optimization
           uses l-bfgsb algorithm from scipy
        """
        parts = self.feats_obj.feat_partitions
        solution = [None for i in parts]
        norm_sol = [None for i in parts]

        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict

        for i,partition in enumerate(parts):

            if len(partition) == 1:     # just use the MLE          
                N = self.feats_obj.N
                data_arr = self.feats_obj.data_arr
                feat_col = partition[0]
                mle_count = 0

                for j in range(N):
                    rvec = data_arr[j, feat_col]
                    if rvec == 1:
                        mle_count += 1
                
                mle = (mle_count * 1.0)/N                
                theta_opt = np.log(mle/(1-mle))                
                
                # Storing like this to maintain consistency with other
                # partitions optimal solutions
                optimThetas = [theta_opt]
                solution[i] = [optimThetas]  # conv to list to maintain consistency
                norm_sol[i] = self.binary_norm_Z(optimThetas, partition)                
                # print(partition, mle, 1-mle, theta_opt, norm_sol[i])
            
            else:           
                num_feats = len(partition)  # number of marginal constraints
                num_2wayc = len([1 for k in twoway_dict if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
                num_3wayc = len([1 for k in threeway_dict if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
                num_4wayc = len([1 for k in fourway_dict if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition

                # length1 = len(partition)
            
                # # number of 'extra' constraints for that partition
                # length2 = len([(k,v) for k,v in topK_pairs_dict.items() 
                #             if (k[0] in partition and k[1] in partition)])
                theta_len = num_feats + num_2wayc + num_3wayc + num_4wayc
                initial_val = np.random.rand(theta_len)

                def func_objective(thetas):
                    objective_sum = 0.0
                    N = self.feats_obj.N        
                    data_arr = self.feats_obj.data_arr

                    # THIS CAN SPED UP BY EFFICIENT NUMPY OPERATIONS
                    # GO OVER ALL THE DATA
                    for i in range(N):
                        rvec = data_arr[i, partition]
                        inner_constraint_sum = self.compute_constraint_sum(thetas, rvec, partition)
                        objective_sum += inner_constraint_sum

                    subtraction_term = N * np.log(self.binary_norm_Z(thetas, partition))
                    objective_sum -= subtraction_term

                    # scaling_factor = 4.0 # make it a float
                    regularization = np.linalg.norm(thetas, 2) * (1.0/self.scaling_factor)
                    objective_sum -= regularization

                    return (-1 * objective_sum) # SINCE MINIMIZING IN THE LBFGS SCIPY FUNCTION


                optimThetas = spmin_LBFGSB(func_objective, x0=initial_val,
                                        fprime=None, approx_grad=True, 
                                        disp=True)

                solution[i] = optimThetas
                norm_sol[i] = self.binary_norm_Z(optimThetas[0], partition)

        self.opt_sol = solution
        self.norm_z = norm_sol

        return (solution, norm_sol)





    def prob_dist(self, rvec):
        """
        Function to compute the probability for a given input vector
        """
        
        prob_product = 1.0
        log_prob = 0.0
        parts = self.feats_obj.feat_partitions
        solution = self.opt_sol
        norm_sol = self.norm_z

        # partition will be a set of indices in the i-th parition        
        for i, partition in enumerate(parts):
            tmpvec = rvec[partition]
            term_exp = self.compute_constraint_sum(solution[i][0], tmpvec, partition)
            log_prob += term_exp - np.log(norm_sol[i])            
            prob_product *= (1.0/norm_sol[i]) * np.exp(term_exp)
        
        return np.exp(log_prob)
        # return prob_product


    def compare_marginals(self):        
                
        N = self.feats_obj.N        
        data_arr = self.feats_obj.data_arr
        num_feats = data_arr.shape[1]

        # all_perms is a generator. So it doesnt store everything in memory all
        # at once!! Very useful for enumerations like this
        all_perms = itertools.product([0, 1], repeat=num_feats)

        mxt_probs = np.zeros(num_feats)
        emp_probs = np.zeros(num_feats)

        for tvec in all_perms:
            vec = np.asarray(tvec)
            for j in range(num_feats):
                if vec[j] == 1:
                    # mxt_dict[j] += self.prob_dist(vec)
                    mxt_probs[j] += self.prob_dist(vec)
        
        for vec in data_arr:
            emp_probs += vec
        
        emp_probs /= N

        return mxt_probs, emp_probs


    def compare_constraints_2way(self):        

        N = self.feats_obj.N        
        data_arr = self.feats_obj.data_arr
        num_feats = data_arr.shape[1]               
        
        all_perms = itertools.product([0, 1], repeat=num_feats)
        pair_dict = self.feats_obj.two_way_dict
        mxt_dict = defaultdict(float)
        emp_dict = defaultdict(float)

        for tvec in all_perms:
            vec = np.asarray(tvec)
            for key,val in pair_dict.items():
                if vec[key[0]] == val[0] and vec[key[1]] == val[1]:
                    mxt_dict[(key,val)] += self.prob_dist(vec)
        
        
        for vec in data_arr:
            for key,val in pair_dict.items():
                if vec[key[0]] == val[0] and vec[key[1]] == val[1]:
                    emp_dict[(key,val)] += 1.0

        for k in emp_dict:
            emp_dict[k] /= N

        return mxt_dict, emp_dict


    def compare_constraints_3way(self):        

        N = self.feats_obj.N        
        data_arr = self.feats_obj.data_arr
        num_feats = data_arr.shape[1]               
        
        all_perms = itertools.product([0, 1], repeat=num_feats)
        pair_dict = self.feats_obj.three_way_dict
        mxt_dict = defaultdict(float)
        emp_dict = defaultdict(float)

        for tvec in all_perms:
            vec = np.asarray(tvec)
            for key, val in pair_dict.items():
                if vec[key[0]] == val[0] and vec[key[1]] == val[1] and vec[key[2]] == val[2]:
                    mxt_dict[(key,val)] += self.prob_dist(vec)
        
        
        for vec in data_arr:
            for key,val in pair_dict.items():
                if vec[key[0]] == val[0] and vec[key[1]] == val[1] and vec[key[2]] == val[2]:
                    emp_dict[(key,val)] += 1.0

        for k in emp_dict:
            emp_dict[k] /= N

        return mxt_dict, emp_dict


    def compare_constraints_4way(self):        

        N = self.feats_obj.N        
        data_arr = self.feats_obj.data_arr
        num_feats = data_arr.shape[1]               
        
        all_perms = itertools.product([0, 1], repeat=num_feats)
        pair_dict = self.feats_obj.four_way_dict
        mxt_dict = defaultdict(float)
        emp_dict = defaultdict(float)

        for tvec in all_perms:
            vec = np.asarray(tvec)
            for key, val in pair_dict.items():
                if vec[key[0]] == val[0] and vec[key[1]] == val[1] and vec[key[2]] == val[2] and vec[key[3]] == val[3]: 
                    mxt_dict[(key,val)] += self.prob_dist(vec)
        
        
        for vec in data_arr:
            for key,val in pair_dict.items():
                if vec[key[0]] == val[0] and vec[key[1]] == val[1] and vec[key[2]] == val[2] and vec[key[3]] == val[3]:
                    emp_dict[(key,val)] += 1.0

        for k in emp_dict:
            emp_dict[k] /= N

        return mxt_dict, emp_dict



    def transition_prob(self, rv1, rv2):
        # rv1 and rv2 are the first and second year's disease 
        # prevalence respectively
        given_rvec = np.append(rv1, rv2)       
        norm_prob = 0   # g_a(r)
        num_feats2 = len(rv2)        

        # generator/iterator
        rv2_perms = itertools.product([0, 1], repeat=num_feats2)
        for v2 in rv2_perms:
            tmp_v2 = np.asarray(v2)
            tmp = np.append(rv1, tmp_v2)
            norm_prob += self.prob_dist(tmp)
        
        trans_prob = self.prob_dist(given_rvec)/norm_prob

        return trans_prob
