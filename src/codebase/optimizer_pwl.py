'''
Optimizer for piecewise likelihood method
'''
from __future__ import division
import itertools
from collections import defaultdict

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
            in the original dataset
    """

    def __init__(self, features_object):
        # Init function for the class object
        
        self.feats_obj = features_object
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
        
        self.zero_stats = p/N
        print('Zero vector probability is: ', self.zero_stats)

    # This function computes the value of N_s to be used in the objective function.
    # N_s denotes the number of rows satisfying the constraints for each partition
    def compute_data_stats(self, partition):
        """
            partition: a list of feature indices which are present in the constraints. 
            If a diseases is not present in any of the constraints, the marginal probability is taken, 
            and the problem is solved as in the normal case (no piecewise likelihood)
            ---------------
            Returns:
            -   'N_s': The data stats vector for the given partition
        """ 
        N = self.feats_obj.N
        data_arr = self.feats_obj.data_arr

        #initialize N_s
        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict

        #sanity check for the constraint to be present in the partition
        num_feats = len(partition)
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc
        N_s = np.zeros(len_theta)

        # Reverse lookup hashmap for the indices in the partition
        # Useful to make thetas and the constraint_sum match up consistently
        # dvec's (data vector) first index corresponds to the first index in the partition
        # with respect to the original vector (before cropping it out for the
        # partition)
        findpos = {elem:i for i,elem in enumerate(partition)}
     
        for i in range(N):
            dvec = data_arr[i, partition]
            tmp_arr = self.util_compute_ns(dvec, partition, twoway_dict, 
                                    threeway_dict, fourway_dict, findpos, len_theta,
                                    num_feats, num_2wayc, num_3wayc, num_4wayc)
            
            N_s += tmp_arr
        N_s /= N

        return N_s
        
    # This function computes the N_s array for every datavector that satisfies each constraint
    def util_compute_ns(self, dvec, partition, twoway_dict, threeway_dict, fourway_dict, findpos,
                    len_theta, num_feats, num_2wayc, num_3wayc, num_4wayc):

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

        len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc
        feat_arr = np.zeros(len_theta)

        j = 0
        # Add up constraint_sum for MARGINAL constraints.
        for i in range(num_feats):
            indicator = 1 if dvec[i] == 1 else 0
            feat_arr[j] += indicator
            j += 1
        
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


    #Function to compute Theta_c * F_c(s) constraint matrix
    def util_compute_fcs(self, partition):
        """ partition: List of feature indices indicating that they all belong
            in the same feature-partition.
        """
        #initialize the indicator function matrix
        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict

        #sanity check for the constraint to be present in the partition
        num_feats = len(partition)
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc
        #As we will have a maximum of 4 way constraints, for now the indicator function matrix has been hard-coded to have length 2^4
        fcs = np.zeros((len_theta, 2**4))

        #Indicator matrix is fixed, for 2 way, 3 way and 4 way constraints
        j = 0
        for i in range(num_feats):
            fcs[j][1] = 1
            j += 1

        for i in range(num_2wayc):
            fcs[j][3] = 1
            j += 1

        for i in range(num_3wayc):
            fcs[j][7] = 1
            j += 1

        for i in range(num_4wayc):
            fcs[j][15] = 1
            j += 1

        return fcs

    # Function to compute the normalization constant Z(theta)
    def compute_norm_z(self, thetas, partition, fcs):
        """
        Computes the normalization constant Z(Theta) for all constraints c.
        Args:
            thetas: The parameters for a partition
            partition: The list of feature indices indicating that these belong to a partition
        Returns:
            Z(theta): vector of length thetas 
        """

        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict

        #sanity check for the constraint to be present in the partition
        num_feats = len(partition)
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        norm_z = np.zeros(len(thetas))

        #Compute Z(theta) for each 2 way, 3 way and 4 way constraint
        j = 0
        for i in range(num_feats):
            num_total_vectors = (2**1)-1
            # norm_z[j] = num_total_vectors + np.exp(thetas[i])
            norm_z[j] = num_total_vectors + np.exp(thetas[j])
            j += 1

        for i in range(num_2wayc):
            num_total_vectors = (2**2)-1
            # norm_z[j] = num_total_vectors + np.exp(thetas[i+num_feats])
            norm_z[j] = num_total_vectors + np.exp(thetas[j])
            j += 1

        for i in range(num_3wayc):
            num_total_vectors = (2**3)-1
            # norm_z[j] = num_total_vectors + np.exp(thetas[i+num_feats+num_2wayc])
            norm_z[j] = num_total_vectors + np.exp(thetas[j])
            j += 1

        for i in range(num_4wayc):
            num_total_vectors = (2**4)-1
            # norm_z[j] = num_total_vectors + np.exp(thetas[i+num_feats+num_2wayc+num_3wayc])
            norm_z[j] = num_total_vectors + np.exp(thetas[j])
            j += 1

        return norm_z

    def solver_optimize(self):
        """Function to perform the optimization
           uses l-bfgsb algorithm from scipy
        """
        parts = self.feats_obj.feat_partitions
        solution = [None for i in parts]
        norm_sol = [None for i in parts]

        for i, partition in enumerate(parts):
            if len(partition) == 1:
                # in every case check what happens if the length of the partition is 1 
                pass 
            else:
                N_s = self.compute_data_stats(partition)
                print('N_s: ', N_s)
                constraint_matrix = self.util_compute_fcs(partition)
                len_theta = N_s.shape[0]
                a = np.random.RandomState(seed=1)
                initial_val = a.rand(len_theta)

                def func_objective(thetas):
                    objective_sum = 0.0
                    theta_fcs = thetas.reshape(-1,1) * constraint_matrix
                    z = self.compute_norm_z(thetas, partition, constraint_matrix)
                    log_norm_z = np.log(z).reshape(-1,1)

                    maximize_matrix = N_s.reshape(-1,1) * (theta_fcs - log_norm_z)
                    objective_sum = maximize_matrix.sum()

                    return (-1 * objective_sum)

                optimThetas = minimize(func_objective, x0 = initial_val, method='L-BFGS-B',
                    options={'disp':False, 'maxcor':20, 'ftol':2.2e-10, 'maxfun':100000})

                # Check if the LBFGS-B converges, if doesn't converge, then return error message
                if optimThetas.status != 0:
                    print(optimThetas.message)
                    return None
                
                solution[i] = optimThetas.x
                norm_sol[i] = self.compute_norm_z(optimThetas.x, partition, constraint_matrix)

                # print('Thetas are: ', optimThetas)

        self.opt_sol = solution
        self.norm_z = norm_sol
        return (solution, norm_sol)

    """

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
                fc = 1 if set(dis).issubset(set(matrix_constraints[pos][1])) else 0 
                fc_r.append(fc)
                theta.append(self.opt_sol[partition][pos])
                z_c.append(self.norm_z[partition][pos])

            return fc_r, theta, z_c

        #initialize total_prob
        total_prob = 0

        for i,rvec in enumerate(all_perms):
            # find the index of diseases in rvec 
            dindx = [i for i,val in enumerate(rvec) if val == 1]

            # initialize fc_r, theta, z_c
            fc_r = []
            theta = []
            z_c = []

            # for each disease present in rvec, find the partition in which that disease belongs to, 
            # find the partition number, index for theta and Z(theta) within that partition 
            # find the index in the two_way, three_way and four_way
            indices = {}
            for d in dindx:
                indices[d] = [i for i,val in enumerate(parts) if d in val]
                indices[d].extend(check_constraint(d))

            for k,v in indices.items():
                f,t,z = generate_params(dindx, v)
                fc_r.extend(f)
                theta.extend(t)
                z_c.extend(z)

            for j in range(len(theta)):
                p_r[i] *= (np.exp(theta[j]*fc_r[j]))/z_c[j]

            total_prob += p_r[i]

        print('Total probability', total_prob)
        return p_r
    """

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

            for key in twoway_dict.keys():
                matrix_constraints.append([i for i in key])

            for key in threeway_dict.keys():
                matrix_constraints.append([i for i in key])

            for key in fourway_dict.keys():
                matrix_constraints.append([i for i in key]) 

        def util_compute_indicator(dindx):
            fc_r = []
            for c in matrix_constraints:
                f = 1 if set(dindx).issubset(set(c)) else 0
                fc_r.append(f)

            return fc_r


        #initialize total_prob
        total_prob = 0
        theta = np.concatenate(self.opt_sol)
        z_c = np.concatenate(self.norm_z)
        self.compute_zero_stats()
        z = np.prod(z_c)*self.zero_stats

        print('Thetas :', theta)
        print('Z_c :', z_c)


        for i, rvec in enumerate(all_perms):
            # find the index of diseases in rvec 
            dindx = [ind for ind,val in enumerate(rvec) if val == 1]

            if len(dindx) == 0:
                p_r[i] = 1/self.zero_stats

            else:
                # initialize fc_r, theta, z_c
                fc_r = util_compute_indicator(dindx)
                p_r[i] = np.exp(np.dot(theta, fc_r))/z
                total_prob += p_r[i]

        print('Probabilities are: ', p_r)
        print('Total probability: ', total_prob)
        return p_r





        





            

