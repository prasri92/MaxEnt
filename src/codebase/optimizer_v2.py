from __future__ import division
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b as spmin_LBFGSB
from scipy.optimize import fmin_tnc as spmin_tnc
from scipy.optimize import linprog

"""
TODO:
- documentation
- dealing with unconstrained/problematic optimizations
- explicitly pass the function gradient as instead of approx_grad = True
  which calcs it numerically. (could lead to faster code!)
"""


class Optimizer(object):
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

    def __init__(self, features_object):
        # Init function for the class object
        
        self.feats_obj = features_object
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

        # CHECKING WITH 1 since BINARY FEATURES
        # Add up constraint_sum for MARGINAL constraints.
        for i in range(num_feats):
            indicator = 1 if rvec[i] == 1 else 0
            constraint_sum += thetas[i] * indicator

        # 2-way constraints
        j = 0
        twoway_offset = num_feats
        for key,val in twoway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                constraint_sum += thetas[twoway_offset + j] * indicator
                j += 1

        # 3-way constraints
        j = 0
        threeway_offset = twoway_offset + num_2wayc
        for key,val in threeway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                constraint_sum += thetas[threeway_offset + j] * indicator
                j += 1

        # 4-way constraints
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



    # This function computes the constraint array    
    # for the entire dataset. Then that array can 
    # be used over and over again
    def compute_data_stats(self, partition):
        """
            partition: a list of feature indices indicating that they all belong
            in a single partition and we only need to consider them for now.
        """ 

        # thetas is ordered as follows: 
        # (1) all the marginal constraints
        # (2) all the two-way constraints
        # (3) all the three-way constraints
        # (4) all the four-way constraints       

        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict
       
        # Sanity Checks for the partition and the given vector
        num_feats = len(partition)  # number of marginal constraints
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        # assert len(rvec) == num_feats        
        # assert len(thetas) == num_feats + num_2wayc + num_3wayc + num_4wayc
        len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc
        data_stats_vector = np.zeros(len_theta)
       
        # Reverse lookup hashmap for the indices in the partition
        # Useful to make thetas and the constraint_sum match up consistently
        # rvec's first index corresponds to the first index in the partition
        # with respect to the original vector (before cropping it out for the
        # partiton)
        findpos = {elem:i for i,elem in enumerate(partition)}

        N = self.feats_obj.N        
        data_arr = self.feats_obj.data_arr        
        for i in range(N):
        # for i in range(1, N):
            rvec = data_arr[i, partition]
            # print("RVEC", rvec)
            tmp_arr = self.util_compute_array(rvec, partition, twoway_dict, 
                                    threeway_dict, fourway_dict, findpos,
                                    num_feats, num_2wayc, num_3wayc, num_4wayc)
            
            data_stats_vector += tmp_arr

            # objective_sum += inner_constraint_sum
        return data_stats_vector



    # This function computes the constraint array for
    # a single vector
    def util_compute_array(self, rvec, partition,
                    twoway_dict, threeway_dict, fourway_dict, findpos,
                    num_feats, num_2wayc, num_3wayc, num_4wayc):

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
            return rvec[0]  # only the marginal constraint applies here
        
        
        def check_condition(key, value):
            # key is a tuple of feature indices
            # value is their corresponding required values
            flag = True
            for i in range(len(key)):
                if rvec[findpos[key[i]]] != value[i]:
                    flag = False
                    break
            return flag

        len_theta = len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc
        feat_arr = np.zeros(len_theta)

        # CHECKING WITH 1 since BINARY FEATURES
        # Add up constraint_sum for MARGINAL constraints.
        for i in range(num_feats):
            indicator = 1 if rvec[i] == 1 else 0
            feat_arr[i] = indicator
            # constraint_sum += thetas[i] * indicator

        # 2-way constraints
        j = 0
        twoway_offset = num_feats
        for key,val in twoway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                feat_arr[twoway_offset + j] = indicator
                # constraint_sum += thetas[twoway_offset + j] * indicator
                j += 1

        # 3-way constraints
        j = 0
        threeway_offset = twoway_offset + num_2wayc
        for key,val in threeway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                feat_arr[threeway_offset + j] = indicator
                # constraint_sum += thetas[threeway_offset + j] * indicator
                j += 1

        # 4-way constraints
        j = 0
        fourway_offset = threeway_offset + num_3wayc
        for key,val in fourway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                feat_arr[fourway_offset + j] = indicator
                # constraint_sum += thetas[fourway_offset + j] * indicator
                j += 1

        # Thetas is still a contiguous across the marginals and the 2way, 3way
        # and the 4way constraints for a given partiton
        # print('feat_arr', feat_arr)
        return feat_arr



   # assuming binary features for now.
    def util_constraint_matrix(self, partition):
        """ partition: List of feature indices indicating that they all belong
            in the same feature-partition.
        """
        # norm_sum = 0.0       
        # num_feats = len(partition)       
        # if num_feats == 1:
        #     norm_sum = 0.0
        #     norm_sum = 1 + np.exp(thetas[0])
        #     return np.log(norm_sum)                   
        
        # thetas is ordered as follows: 
        # (1) all the marginal constraints
        # (2) all the two-way constraints
        # (3) all the three-way constraints
        # (4) all the four-way constraints       

        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict       
        
        num_feats = len(partition)  # number of marginal constraints
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc
        data_stats_vector = np.zeros(len_theta)
       
        # Reverse lookup hashmap for the indices in the partition
        # Useful to make thetas and the constraint_sum match up consistently
        # rvec's first index corresponds to the first index in the partition
        # with respect to the original vector (before cropping it out for the
        # partiton)
        findpos = {elem:i for i,elem in enumerate(partition)}       

        # Create all permuatations of a vector belonging to that partition
        all_perms = list(itertools.product([0, 1], repeat=num_feats))[1:]
        num_total_vectors = 2**(num_feats)-1
        # print('length of all_perms', len(all_perms)==num_total_vectors)


        # all_perms = itertools.product([0, 1], repeat=num_feats)
        # num_total_vectors = 2**(num_feats)
        constraint_mat = np.zeros((num_total_vectors, len_theta))        
                

        for i, vec in enumerate(all_perms):
            tmpvec = np.asarray(vec)
            # tmp = self.compute_constraint_sum(thetas, tmpvec, partition)
            tmp_arr = self.util_compute_array(tmpvec, partition, twoway_dict, 
                                    threeway_dict, fourway_dict, findpos,
                                    num_feats, num_2wayc, num_3wayc, num_4wayc)
            constraint_mat[i,:] = tmp_arr
        
        # TEST to check if the constraint matrix being zero gives bad gradient function
        # print("constraint_mat before\n", constraint_mat)
        # constraint_mat[num_total_vectors-1,:] = np.zeros(len_theta)
        # print("constraint_mat after\n", constraint_mat)
        return constraint_mat

    def build_constraint_matrix(self, num_feats):
        """
        Builds the primal constraint matrix to solve the linprog for detection of zero vectors
        """
        A_eq = np.zeros([2**num_feats-1, 2**num_feats-1])
        d = list(itertools.product([0,1], repeat=num_feats))[1:]
        
        i,j=0,0
        for dis in d:
            index_d = [i for i,val in enumerate(dis) if val==1]
            r = list(itertools.product([0,1], repeat=num_feats))[1:]
            j=0
            for rvec in r:
                index_r = [i for i,val in enumerate(rvec) if val==1]
                if(all(x in index_d for x in index_r)):
                    A_eq[j][i]=1
                j+=1
            i+=1

        return A_eq

    def build_constraint_matrix_approx(self, A_exact):
        """
        Builds the primal constraint matrix to solve the linprog for detection of zero vectors (approximate method)
        """
        A_eq = np.concatenate((A_exact,A_exact), axis=1)
        return A_eq

    def exact_zero_detection(self, cleaneddata):
        """
        An exact iterative method for the detection of zero probabilities of the "r" vector of a patient.
        Considers all vectors to be zero vectors at first, and then after running through the lin prog, 
        returns all actual zero vectors
        Args:
            partition: The set of all diseases present in the partition
            constraint_mat: Set of constraints to be satisfied 
        Returns: 
            all actual zero vectors
        """
        parts = self.feats_obj.feat_partitions
        for i in parts:
            indices = list(i)
            num_feats = len(i)
            #objective function = maximize summation of p(r)
            # f = (-1 *np.arange(1, 2**num_feats))
            f = (-1 * np.ones((2**num_feats)-1))

            '''
            Constraints are of the form A_eq @ x == b_eq
            where A_eq is the matrix including the information for the marginals, 2 way,
            3 way and 4 way constraints
            b_eq is the sum of probabilities according to the maximum entropy
            '''
            
            #Build the primal constraint matrix
            A_eq = self.build_constraint_matrix(num_feats)

            #To count the occurrences of only diseases present in the partition, transform incoming data
            #accordingly

            indices = [str(i) for i in indices]

            data = cleaneddata[indices]
            size = data.shape[0]
            diseases = data.shape[1]
            cols = np.arange(diseases)
            data.columns = cols

            b_eq = []
            
            all_perms = list(itertools.product([0,1], repeat=diseases))

            ndata = pd.DataFrame()

            for perm in all_perms[1:]:
                ones = [i for i,x in enumerate(perm) if perm[i]==1]
                sub_data = data[ones]
                sub_data['m'] = np.all(sub_data,axis=1)*1
                t = np.sum(sub_data['m'], axis=0)
                m = t/size
                b_eq.append(m)

            print('parts', i, 'b_eq', b_eq)

            #upper bounds on x
            A_ub = np.identity((2**num_feats)-1)
            b_ub = np.ones((2**num_feats)-1)
            
            res = linprog(f, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)#,options = {"presolve":False})
            res = {'status': res.status, 'message':res.message, 'x': res.x if res.success else None}
            if res['status']!=0:
                print('LP to find zero vectors: ',res['message'])
            else:
                print('Solving the linear program', res['x'])

    def approximate_zero_detection(self, cleaneddata):
        """
        An approximate method for detection of zero atoms. 
        Maximize v_b which lies between 0 and epsilon, setting epsilon to 0.0001
        subject to the constraints p(r) = v_b + w_b where w_b + v_b = p(r)
        Args:
            partition: The set of all diseases present in the partition
            constraint_mat: Set of constraints to be satisfied
        Returns:
            zero vectors
        """
        parts = self.feats_obj.feat_partitions
        for i in parts:
            indices = list(i)
            num_feats = len(i)
            #objective function - maximize v_b (which lies between 0 and epsilon, epsilon = 0.0001
            v = np.ones((2**num_feats)-1)
            w = np.zeros((2**num_feats)-1)
            c = np.concatenate((v,w), axis=0)
            f = (-1 * c)

            '''
            Constraints are of the form A_eq @ x == b_eq
            where A_eq is the matrix including the information for the marginals, 2 way, 3 way and 4 way constraints
            b_eq is the sum of probabilities according to maximum entropy

            Upper bound constraints are imposed in the form A_ub @ x == b_ub
            where now x = v + w 
            '''

            A_eq = self.build_constraint_matrix_approx(self.build_constraint_matrix(num_feats))

            #To count the occurrences of only diseases present in the partition, transform incoming data
            #accordingly

            indices = [str(i) for i in indices]

            data = cleaneddata[indices]
            size = data.shape[0]
            diseases = data.shape[1]
            cols = np.arange(diseases)
            data.columns = cols

            b_eq = []
            
            all_perms = list(itertools.product([0,1], repeat=diseases))

            ndata = pd.DataFrame()

            for perm in all_perms[1:]:
                ones = [i for i,x in enumerate(perm) if perm[i]==1]
                sub_data = data[ones]
                sub_data['m'] = np.all(sub_data,axis=1)*1
                t = np.sum(sub_data['m'], axis=0)
                m = t/size
                b_eq.append(m)

            print('parts', i, 'b_eq', b_eq)

            #upper bounds on x
            A_ub = np.identity(((2**num_feats)-1)*2)
            v_ub = np.array([0.001]*((2**num_feats)-1))
            w_ub = np.ones(2**num_feats-1)
            b_ub = np.concatenate((v_ub,w_ub), axis=0)

            res = linprog(f, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, options={"disp": False})
            res = {'message': res.message, 'status':res.status, 'x': res.x if res.success else None}
            if res['status']!=0:
                print('LP to find zero vectors: ', res['message'])
            else:
                print('Solving the linear program', res['x'][:2**num_feats])


    # normalization constant Z(theta)
    # assuming binary features for now.
    def log_norm_Z(self, thetas, partition, constraint_mat):
        """Computes the log of normalization constant Z(theta) for a given partition
        Uses the log-sum-exp trick for numerical stablility
        Args:
            thetas: The parameters for the given partition
            partition: List of feature indices indicating that they all belong
            in the same feature-partition.
        """
        norm_sum = 0.0       
        num_feats = len(partition)
       
        if num_feats == 1:
            norm_sum = 0.0
            norm_sum = 1 + np.exp(thetas[0])
            return np.log(norm_sum)            
        
        num_total_vectors = 2**(num_feats)-1
        # num_total_vectors = 2**(num_feats)
        inner_array = np.dot(constraint_mat, thetas)
        
        log_norm = 0.0
        a_max = np.max(inner_array)
        inner_array -= a_max
        log_norm = a_max + np.log(np.sum(np.exp(inner_array)))

        return log_norm


    # normalization constant Z(theta)
    # assuming binary features for now.
    # Still KEEP it around for len(part) == 1 case
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
        # all_perms = itertools.product([0, 1], repeat=num_feats)
        all_perms = list(itertools.product([0, 1], repeat=num_feats))[1:]

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

        # twoway_dict = self.feats_obj.two_way_dict
        # threeway_dict = self.feats_obj.three_way_dict
        # fourway_dict = self.feats_obj.four_way_dict

        for i, partition in enumerate(parts):

            if len(partition) == 1:     # just use the MLE          
                N = self.feats_obj.N
                data_arr = self.feats_obj.data_arr
                feat_col = partition[0]
                mle_count = 0

                for j in range(N):
                    rvec = data_arr[j, feat_col]
                    if rvec == 1:
                        mle_count += 1
                
                if mle_count == 0:
                    print('Zero mle for :', feat_col)
                mle = (mle_count * 1.0)/N                
                theta_opt = np.log(mle/(1-mle))                
                
                # Storing like this to maintain consistency with other
                # partitions optimal solutions
                optimThetas = [theta_opt]
                solution[i] = [optimThetas]  # conv to list to maintain consistency
                norm_sol[i] = self.binary_norm_Z(optimThetas, partition)                
                # print(partition, mle, 1-mle, theta_opt, norm_sol[i])
            
            else:         
                datavec_partition = self.compute_data_stats(partition)
                c_matrix_partition = self.util_constraint_matrix(partition)
                len_theta = datavec_partition.shape[0]
                initial_val = np.random.rand(len_theta)


                def func_objective(thetas):
                    objective_sum = 0.0
                    N = self.feats_obj.N        
                    # data_arr = self.feats_obj.data_arr

                    # # THIS CAN BE SPED UP BY EFFICIENT NUMPY OPERATIONS
                    # for i in range(N):
                    #     rvec = data_arr[i, partition]
                    #     inner_constraint_sum = self.compute_constraint_sum(thetas, rvec, partition)
                    #     objective_sum += inner_constraint_sum
                    
                    theta_term = np.dot(datavec_partition, thetas)
                    # norm_term = -1 * N * np.log(self.binary_norm_Z(thetas, partition))                    
                    norm_term = -1 * N * self.log_norm_Z(thetas, partition, c_matrix_partition)
                    objective_sum = theta_term + norm_term

                    return (-1 * objective_sum) # SINCE MINIMIZING IN THE LBFGS SCIPY FUNCTION

            
                optimThetas = spmin_LBFGSB(func_objective, x0=initial_val,
                                        fprime=None, approx_grad=True, 
                                        disp=True, epsilon=1e-18)
                # optimThetas = spmin_tnc(func_objective, x0=initial_val,
                #                         fprime=None, approx_grad=True, 
                #                         disp=True)
                
                solution[i] = optimThetas
                

                # norm_sol[i] = self.binary_norm_Z(optimThetas[0], partition)

                norm_sol[i] = np.exp(self.log_norm_Z(optimThetas[0], partition, c_matrix_partition))
                inn_arr = np.dot(c_matrix_partition, optimThetas[0])
                inn_arr = np.exp(inn_arr)
                if norm_sol[i] != 0:
                    inn_arr /= norm_sol[i]
                else:
                    print('Numerical error with exponent')
                total_prob = np.sum(inn_arr, axis=0)
                print('thetas', optimThetas)
                print('Partition num, Total prob: ', i, total_prob)

        self.opt_sol = solution
        self.norm_z = norm_sol
        return (solution, norm_sol)


    def prob_dist(self, rvec):
        """
        Function to compute the probability for a given input vector
        """        
        np.seterr(all='raise')
        log_prob = 0.0
        parts = self.feats_obj.feat_partitions
        solution = self.opt_sol
        norm_sol = self.norm_z

        # partition will be a set of indices in the i-th parition        
        for i, partition in enumerate(parts):
            tmpvec = rvec[partition]
            term_exp = self.compute_constraint_sum(solution[i][0], tmpvec, partition)
            try:
                part_logprob = term_exp - np.log(norm_sol[i])
                log_prob += part_logprob
                part_prob = np.exp(part_logprob)
            except:
                #TODO: CHECK WHAT IS THE CORRECT WAY TO HANDLE THIS EXCEPTION, GIVING INCORRECT RESULTS NOW
                #TODO: FIX 
                #Does it happen even for machine precision? 
                part_logprop = term_exp - 1.0
                log_prob += part_logprop
                return 0.0
                # part_prob = np.exp(part_logprop)
            # print('partition, prob: ', i, part_prob)            
            # prob_product *= (1.0/norm_sol[i]) * np.exp(term_exp)
        
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
