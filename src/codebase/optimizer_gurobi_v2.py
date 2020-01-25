from __future__ import division
import itertools
from collections import defaultdict
from gurobipy import *
import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b as spmin_LBFGSB
from scipy.optimize import fmin_tnc as spmin_tnc
from scipy.optimize import linprog
from scipy.optimize import minimize
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
        self.zero_indices = {}
        

    # Utility function to check whether a tuple (key from constraint dict)
    # contains all the variables inside the given partition.
    def check_in_partition(self, partition, key_tuple):
        flag = True
        for i in key_tuple:
            if i not in partition:
                flag = False
                break
        return flag


    # This function computes the inner sum of the optimization function objective    
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
        # (2) 0 diseases constraints
        # (3) all the two-way constraints
        # (4) all the three-way constraints
        # (5) all the four-way constraints

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
        
        # assert len(rvec) == num_feats        
        # assert len(thetas) == num_feats + num_2wayc + num_3wayc + num_4wayc + 1
        
       
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

        # zero vector constraint
        j = 0
        zero_offset = num_feats
        indicator = 1 if sum(rvec) == 0 else 0
        constraint_sum += thetas[zero_offset + j] * indicator
        # j += 1
        # indicator = 1 if sum(rvec) == 1 else 0
        # constraint_sum += thetas[zero_offset + j] * indicator
        # j += 1
        # indicator = 1 if sum(rvec) == 2 else 0
        # constraint_sum += thetas[zero_offset + j] * indicator
        # j += 1
        # indicator = 1 if sum(rvec) == 3 else 0
        # constraint_sum += thetas[zero_offset + j] * indicator
        # j += 1
        # indicator = 1 if sum(rvec) == 4 else 0
        # constraint_sum += thetas[zero_offset + j] * indicator

        # 2-way constraints
        j = 0
        twoway_offset = num_feats + 1 
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
        # (2) 0 and 1 disease constraints
        # (3) all the two-way constraints
        # (4) all the three-way constraints
        # (5) all the four-way constraints     

        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict
       
        # Sanity Checks for the partition and the given vector
        num_feats = len(partition)  # number of marginal constraints
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        # assert len(rvec) == num_feats        
        # assert len(thetas) == num_feats + num_2wayc + num_3wayc + num_4wayc + 1
        len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc + 1
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
        # (2) zero vector constraints
        # (3) all the two-way constraints
        # (4) all the three-way constraints
        # (5) all the four-way constraints

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

        len_theta = len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc + 1
        feat_arr = np.zeros(len_theta)

        # CHECKING WITH 1 since BINARY FEATURES
        # Add up constraint_sum for MARGINAL constraints.
        for i in range(num_feats):
            indicator = 1 if rvec[i] == 1 else 0
            feat_arr[i] = indicator
            # constraint_sum += thetas[i] * indicator

        # zero vector constraint
        j = 0
        zero_offset = num_feats
        indicator = 1 if sum(rvec) == 0 else 0
        feat_arr[zero_offset + j] = indicator
        # j += 1
        # indicator = 1 if sum(rvec) == 1 else 0
        # feat_arr[zero_offset + j] = indicator
        # j += 1
        # indicator = 1 if sum(rvec) == 2 else 0
        # feat_arr[zero_offset + j] = indicator
        # j += 1
        # indicator = 1 if sum(rvec) == 3 else 0
        # feat_arr[zero_offset + j] = indicator
        # j += 1
        # indicator = 1 if sum(rvec) == 4 else 0
        # feat_arr[zero_offset + j] = indicator


        # 2-way constraints
        j = 0
        twoway_offset = num_feats + 1
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
        print("\n")
        print('Partition for calculation:', partition)       

        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict       
        
        num_feats = len(partition)  # number of marginal constraints
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc + 1
        data_stats_vector = np.zeros(len_theta)
       
        # Reverse lookup hashmap for the indices in the partition
        # Useful to make thetas and the constraint_sum match up consistently
        # rvec's first index corresponds to the first index in the partition
        # with respect to the original vector (before cropping it out for the
        # partiton)
        findpos = {elem:i for i,elem in enumerate(partition)}       

        # Create all permuatations of a vector belonging to that partition
        # all_perms = list(itertools.product([0, 1], repeat=num_feats))[1:]
        # num_total_vectors = 2**(num_feats)-1
        # print('length of all_perms', len(all_perms)==num_total_vectors)

        all_perms = itertools.product([0, 1], repeat=num_feats)
        
        #Use a generator to yield only non zero vecotrs
        def non_zero_vectors(self, partition):
            for i, vec in enumerate(all_perms):
                if (vec not in self.zero_indices[tuple(partition)]):
                    yield vec

        N = self.feats_obj.N

        #all_perms = itertools.product([0, 1], repeat=num_feats)
        num_total_vectors = 2**(num_feats)
        if len(self.zero_indices)!=0 and tuple(partition) in self.zero_indices.keys():
            num_total_vectors -= len(self.zero_indices[tuple(partition)])
        constraint_mat = np.zeros((num_total_vectors, len_theta))        
                
        for i, vec in enumerate(non_zero_vectors(self, partition)):
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
        #A_eq = np.zeros([2**num_feats, 2**num_feats])
        vectors = list((itertools.product([0,1], repeat=num_feats)))
        l=tuplelist([])
        
        for i, d in enumerate(vectors):
            if i==0:
                continue
            diseases_d = set([i for i,val in enumerate(d) if val==1])

            for j, r in enumerate(vectors):
                if j==0:
                    continue
                diseases_r = set([i for i,val in enumerate(r) if val==1])
                if len(diseases_d)>=len(diseases_r):
                    if diseases_r.issubset(diseases_d):
                        l+=[(i,j)]
                

            #     #########################################################################
            #     if(j!=0 and all(x in index_d for x in index_r)): #changed all to any
            #         l+=[(j,i)]
            #         #A_eq[j][i]=1
            #     j+=1
            # i+=1
        #A_eq[0] = np.zeros([2**num_feats])
        #A_eq[0][0]=1
        #print('A_eq', A_eq)
        l.append((0,0))
        # print('THE CONSTRAINT MATRIX IS: ', l)
        return l

    def build_constraint_matrix_approx(self, A_exact):
        """
        Builds the primal constraint matrix to solve the linprog for detection of zero vectors (approximate method)
        """
        A_eq = np.concatenate((A_exact,A_exact), axis=1)
        return A_eq


    def approximate_zero_detection(self, cleaneddata):
        """
        An approximate method for detection of zero atoms. 
        Maximize v_b which lies between 0 and epsilon, setting epsilon to 0.0001
        subject to the constraints p(r) = v_b + w_b where w_b + v_b = p(r)
        Args:
            partition: The set of all diseases present in the partition
            constraint_mat: Set of constraints to be satisfied
        Returns:
            The LP solution vector probabilities 
        """
        print("\nDetecting zero vectors")
        parts = self.feats_obj.feat_partitions
        for i in parts:
            indices = list(i)
            num_feats = len(i)
            #objective function - maximize v_b (which lies between 0 and epsilon, epsilon = 0.0001
            v = np.ones(2**num_feats)
            w = np.zeros(2**num_feats)
            c = np.concatenate((v,w), axis=0)
            f = (-1 * c)

            '''
            Constraints are of the form A_eq @ x == b_eq
            where A_eq is the matrix including the information for the marginals, 2 way, 3 way and 4 way constraints
            b_eq is the sum of probabilities according to maximum entropy
            Upper bound constraints are imposed in the form A_ub @ x == b_ub
            where now x = v + w 
            '''
            
            #ignore warnings 
            pd.options.mode.chained_assignment = None  # default='warn'

            A_eq = self.build_constraint_matrix_approx(self.build_constraint_matrix(num_feats))
            
            '''
            To find marginal probabilities and constraint probabilities, the data is transformed
            into smaller subsets, and their individual probabilities are summed. 
            Each subset represents the diseases and constraints  
            '''
            indices_str = [str(i) for i in indices]

            data = cleaneddata[indices_str]
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
                print('ones:', ones)
                sub_data = data[ones]
                sub_data['m'] = np.all(sub_data,axis=1)*1
                t = np.sum(sub_data['m'], axis=0)
                m = t/size
                b_eq.append(m)

            print('Diseases', i, 'Marginal Probabilities', b_eq)
         
            #Imposing upper bounds on x, where x = v + w
            A_ub = np.identity((2**num_feats)*2)
            v_ub = np.array([0.01]*(2**num_feats))
            w_ub = np.ones(2**num_feats)
            b_ub = np.concatenate((v_ub,w_ub), axis=0)

            #Solving the linear program using simplex method
            res = linprog(f, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, options={"disp": False})
            res = {'message': res.message, 'status':res.status, 'x': res.x if res.success else None}
            #Flag to check if there are any zero vectors
            flag = 0
            zero_vectors = []
            if res['status']!=0:
                print('LP to find zero vectors is unsuccessful: ',res['message'])
            else:
                print('Solving the linear program gives us the vector probabilities: \n', res['x'][:2**num_feats])
                #Check if any of the vectors are zero 
                for vector, lp_prob in enumerate(res['x'][:2**num_feats]):
                    if lp_prob == 0:
                        flag = 1
                        # print("The zero vectors are: " + format(vector, "0"+str(diseases)+"b"))
                        zero_vectors.append(format(vector,"0"+str(diseases)+"b"))

            if flag==1:
                print("Eliminate zero vectors\n")
                print("The zero vectors are:", zero_vectors)
                print()

    def exact_zero_detection(self, cleaneddata):
        """
        An exact iterative method for the detection of zero probabilities of the "r" vector of a patient.
        Considers all vectors to be zero vectors at first, and then after running through the lin prog, 
        returns all actual zero vectors
        Args:
            partition: The set of all diseases present in the partition
            constraint_mat: Set of constraints to be satisfied 
        Returns: 
            The LP solution to all vectors
        """

        print("\nDetecting zero vectors")
        model=Model('Zero_atom_detection') #Initialized Gurobi model


        parts = self.feats_obj.feat_partitions
        non_single_parts=[p for p in parts if len(p)!=1]

        for nsp in non_single_parts:
            indices = list(nsp)
            num_feats = len(nsp)
            print("\n")
            print("Diseases:", indices)
            print("Number of diseases:", num_feats)
            self.zero_indices[tuple(indices)] = list()

            '''
            Constraints are of the form A_eq @ x == b_eq
            where A_eq is the matrix including the information for the marginals, 2 way,
            3 way and 4 way constraints
            b_eq is the sum of probabilities according to the maximum entropy
            Upper bound constraints are imposed in the form A_ub @ x == b_ub
            The upper bound constraints say that the bounds of x are [0,1]
            '''

            #ignore warnings for pandas dataframe handling 
            pd.options.mode.chained_assignment = None  # default='warn'

            '''
            To find marginal probabilities and constraint probabilities, the data is transformed
            into smaller subsets, and their individual probabilities are summed. 
            Each subset represents the diseases and constraints  
            '''

            indices_str = [str(i) for i in indices]
            # print("indices:", indices)
            # print("indices_str", indices_str)

            #TODO: check why the error is present? 
            data = cleaneddata[indices_str]
            # data = cleaneddata[indices]

            size = data.shape[0]
            diseases = data.shape[1]
            cols = np.arange(diseases)
            data.columns = cols

            '''Add all gurobi variables to our model, and define their coefficients in the
            objective function

            FOR REFERENCE ONLY:
            i) addVars uses a tuplelist object to store all possible vectors (easy querying).
            ii) Use a dictionary comprehension for retrieving the corresponding variables
            that are key value pairs. 
            
            For example: 
            vals = { k : v.X for k,v in l.items() }'''
            
            x=model.addVars(itertools.product([0,1], repeat=diseases), ub=1.0, lb=0.0, obj=1.0)


            '''Develop the b_eq matrix and add the corresponding constraints in the following order:
            (1) Marginal constraints
            (2) Zero vector constraint
            (3) Two way constraints
            (4) Three way constraints
            (5) Four way constraints'''

            
            '''Create a dictionary 'linking' corresponding disease (indices) to newly 
            assigned column labels(cols). This will be useful while imposing constraints
            for our LP'''
            
            link=dict(zip(indices, cols))


            ''' Use q for querying variables from the gurobi tuple list object'''
            q=tuple(['*']*diseases)
            
            # initialize b_eq
            
            b_eq = []
                                    
            '''Marginal constraints'''
            for i in indices:
                col=link[i]
                t_m=np.sum(data[col], axis=0)
                p_m=t_m/size
                b_eq.append(p_m)

                #temporarily used for querying
                q_m=['*']*diseases 
                q_m[col]=1                
                model.addConstr(x.sum(*q_m)==p_m)

                
            '''Zero constraint'''
            #ndata = new data 
            ndata = pd.DataFrame()
            q_zero=[0]*diseases
            ndata[(*q_zero,)] = np.logical_not(np.any(data, axis=1))*1
            p_zero=np.sum(ndata[(*q_zero,)])/size            
            b_eq.append(p_zero)
            model.addConstr(x.sum(*q_zero)==p_zero)


            '''Two-way constraints'''
            for key in self.feats_obj.two_way_dict:
                if key[0] not in nsp:
                    continue

                else:                                        
                    #Find the corresponding columns in nsp for each disease in the constraint
                    constraint_indices=list(key)
                    constraint_cols=[link[i] for i in constraint_indices] 
                    sub_data=data[constraint_cols]
                    sub_data['m']=np.all(sub_data, axis=1)*1                    
                    t_two=np.sum(sub_data['m'], axis=0)
                    p_two=t_two/size
                    b_eq.append(p_two)                    
                    q_two=list(q)                 
                    for col in constraint_cols:
                        q_two[col]=1                                        
                    model.addConstr(x.sum(*q_two)==p_two)


            '''Three-way constraints'''        
            for key in self.feats_obj.three_way_dict:
                if key[0] not in nsp:
                    continue

                else:                                        
                    #Find the corresponding columns in nsp for each disease in the constraint
                    constraint_indices=list(key)
                    constraint_cols=[link[i] for i in constraint_indices] 
                    sub_data=data[constraint_cols]
                    sub_data['m']=np.all(sub_data, axis=1)*1
                    t_three=np.sum(sub_data['m'], axis=0)
                    p_three=t_three/size
                    b_eq.append(p_three)                    
                    q_three=list(q)                 
                    for col in constraint_cols:
                        q_three[col]=1                
                    model.addConstr(x.sum(*q_three)==p_three)

            '''Four-way constraints'''        
            for key in self.feats_obj.four_way_dict:
                if key[0] not in nsp:
                    continue

                else:                                        
                    #Find the corresponding columns in nsp for each disease in the constraint
                    constraint_indices=list(key)
                    constraint_cols=[link[i] for i in constraint_indices] 
                    sub_data=data[constraint_cols]
                    sub_data['m']=np.all(sub_data, axis=1)*1
                    t_four=np.sum(sub_data['m'], axis=0)
                    p_four=t_four/size
                    b_eq.append(p_four)                    
                    q_four=list(q)                 
                    for col in constraint_cols:
                        q_four[col]=1                
                    model.addConstr(x.sum(*q_four)==p_four)                    

            #Add probability sum constraint
            coeffs=[1]*len(x)
            sum_vars=[x[i] for i in x.keys()]
            expr=gurobipy.LinExpr(coeffs, sum_vars)
            model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=1.0)

            model.Modelsense=GRB.MAXIMIZE #maximize the objective             
            model.write("LP.lp")
            model.optimize()
            zero_vectors = set({k for k,v in x.items() if v.X==0})
            non_zero_vectors = set({k for k,v in x.items() if v.X!=0})

            # if after the first iteration, there are no zero vectors move on to the next cluster
            if len(zero_vectors) == 0:
                print('No zero vectors in this cluster')
                continue

            # Else, use the iterative method to check for more zero vectors 
            it_no = 0
            while len(zero_vectors)!=0:
                it_no += 1
                prev_zero_vectors=zero_vectors#Take note of the zero vectors
                model.reset() #reset the solved model to an unsolved state
                
                #Change the objective function
                model.setObjective(0.0) #First, remove the objective function                                        
                for vec in zero_vectors: #Now add variables corresponding to zero atoms only
                    x[vec].Obj=1.0
                
                #Solve the model
                model.update()
                model.Modelsense=GRB.MAXIMIZE
                model.write("LP.lp")
                model.optimize()
                non_zero_vectors = set({ k for k,v in x.items() if v.X!=0})                
                
                zero_set=prev_zero_vectors-non_zero_vectors
                print('ZERO SET for iteration: ', it_no, ' : ', len(zero_set))
               
                #Stopping condition
                if zero_set==set():
                    print("There are no zero vectors in cluster")
                    break
                
                elif zero_set==prev_zero_vectors:
                    print("There are zero vectors")
                    self.zero_indices[tuple(indices)] = list(zero_set)
                    # print("Gurobi variable number:", {v for k,v in x.items() if v.X==0})
                    break

                zero_vectors=zero_set

            '''Reset the same Gurobi model for use in the next cluster iteration if required'''
            model.setObjective(0.0) #reset the objective function to zero
            model.remove(model.getConstrs()) #remove all constraints            
            model.remove(model.getVars()) #remove all variables
        print('Zero vectors are:', self.zero_indices)


    def analyze_zero_atoms(self, cleaneddata):
        '''
        Check empirical dataset vs. zero atoms as assigned by the LP
        '''
        for indices in self.feats_obj.feat_partitions:
            zero_vectors = self.zero_indices[tuple(indices)]

            indices_str = [str(i) for i in indices]
            data = cleaneddata[indices_str]

            def convert_to_int(p_row):
                string = p_row.tolist()
                string = [str(i) for i in string]
                string = ''.join(string)
                string = int(string)
                return int(string, base=2)

            data['Number'] = data.apply(convert_to_int, axis=1)

            print(data)                
            


    """
    #old method
    def exact_zero_detection(self, cleaneddata): 
        '''
        An exact iterative method for the detection of zero probabilities of the "r" vector of a patient.
        Considers all vectors to be zero vectors at first, and then after running through the lin prog, 
        returns all actual zero vectors
        Args:
            partition: The set of all diseases present in the partition
            constraint_mat: Set of constraints to be satisfied 
        Returns: 
            The LP solution to all vectors
        '''
        print("\nDetecting zero vectors")
        model=Model('Zero_atom_detection') #Initialized Gurobi model

        '''
        First, we compute the b_eq matrix. To find marginal probabilities and constraint probabilities, the data is transformed
        into smaller subsets, and their individual probabilities are summed. 
        Each subset represents the diseases and constraints  
        '''

        parts = self.feats_obj.feat_partitions
        non_single_parts=[p for p in parts if len(p)!=1]

        for nsp in non_single_parts:
            indices = list(nsp)
            num_feats = len(nsp)
            print("\n")
            print("Diseases:", indices)
            print("Number of diseases:", num_feats)

            '''
            Constraints are of the form A_eq @ x == b_eq
            where A_eq is the matrix including the information for the marginals, 2 way,
            3 way and 4 way constraints
            b_eq is the sum of probabilities according to the maximum entropy
            Upper bound constraints are imposed in the form A_ub @ x == b_ub
            The upper bound constraints say that the bounds of x are [0,1]
            '''

            #ignore warnings for pandas dataframe handling 
            pd.options.mode.chained_assignment = None  # default='warn'

            '''
            To find marginal probabilities and constraint probabilities, the data is transformed
            into smaller subsets, and their individual probabilities are summed. 
            Each subset represents the diseases and constraints  
            '''
            #TODO: some experiments don't work with STR indices
            indices_str = [str(i) for i in indices]
            # print("indices:", indices)
            # print("indices_str", indices_str)

            data = cleaneddata[indices_str]
            # data = cleaneddata[indices]

            size = data.shape[0]
            diseases = data.shape[1]
            cols = np.arange(diseases)
            data.columns = cols        

            # initialize b_eq
            b_eq = []
            all_perms = list(itertools.product([0,1], repeat=diseases))

            #FIX 1 - Remove all the permutations for which constraints are not present
            # Consider whether or not to use tuplelist 

            #ndata = new data 
            ndata = pd.DataFrame()
            ndata[all_perms[0]] = np.logical_not(np.any(data, axis=1))*1            
            b_eq.append(np.sum(ndata[all_perms[0]])/size)

            for perm in all_perms[1:]:
                ones = [i for i,x in enumerate(perm) if perm[i]==1]
                sub_data = data[ones]
                sub_data['m'] = np.all(sub_data,axis=1)*1
                #print("sub data:", sub_data)
                t = np.sum(sub_data['m'], axis=0)
                m = t/size
                b_eq.append(m)

            # print('Remove vectors from the b_eq matrix with zero marginal probabilities: 
            #Done before first iteration of zero atom detection')
            # remove_indices = []
            
            # for i_beq, val in enumerate(b_eq):
            #     if val == 0.0:
            #         remove_indices.append(i_beq)

            J=[]

            all_perms = list(itertools.product([0,1], repeat=diseases))
            
            permdict={} #dictionary for storing permutations and their corresponding number 

            for i_perms,perm in enumerate(all_perms):
                 # print('Vector: ', perm, ' Empirical Probability: ', b_eq[ind])
                 # permdict[ind]=perm
                 J.append(i_perms)

            # #print('length of all perms:', len(all_perms))

    

            # #Delete rows of A_eq
            # b_eq = np.delete(np.array(b_eq), remove_indices, axis=0)

            # J=sorted(list(np.delete(np.array(J), remove_indices, axis=0))) #J is the list of all vectors
            # #print('length of J:', len(J))
            
            #Build the primal constraint matrix and objective function in Gurobi
            
            x={} #Empty dictionary to store all variables in the model
            l = self.build_constraint_matrix(diseases) #build constraint matrix
            # print('SHAPE OF CONSTRAINT MATRIX:', l)

            # Making the variables for the LP matrix
            binarystrings={}
            for n in range(len(all_perms)):
                x[n]=model.addVar(ub=1.0, lb=0.0, obj=1.0) #each of the probabilities for all possible vectors
                binarystrings[n]=format(n, '0'+str(diseases)+'b')
                        #Linear Program using simplex method

            model.Modelsense=GRB.MAXIMIZE #maximize the objective
            # print('BINARY STRINGS:', binarystrings)   
            #print('l:', l)
            #print('J:', J)

            #print('l:', l[0:100])

            #Initialize constraint dictionary
            #print("Variable dictionary:", x)
            #Add all respective marginal constraints
            for j in range(len(b_eq)):
                J_j=J[j]
                L=l.select('*',J_j)
                # print("J_j, L:", J_j, L)                
                variables=[x[p] for p, q in L] #if i in J]
                coeffs=[1]*len(variables)
                expr=gurobipy.LinExpr(coeffs, variables)                
                model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=b_eq[j]) 
            
            #Add probability sum constraint
            coeffs=[1]*len(x)
            sum_vars=[x[i] for i in x.keys()]
            expr=gurobipy.LinExpr(coeffs, sum_vars)
            model.addConstr(lhs=expr, sense=gurobipy.GRB.EQUAL, rhs=1.0)

            model.write("LP.lp")
            model.optimize()
            varslist=model.getVars()
            # print('varslist:', varslist)
            v=[]
            for vrbls in varslist:
                v.append(vrbls.x)
            # print('FINAL VARIABLES:', v)   
            zero_indices=[i for i, Vars in enumerate(v) if Vars==0]
            # print('ZERO INDICES:', zero_indices)
            non_zero_indices=[i for i, Vars in enumerate(v) if Vars!=0]
            # for ind in zero_indices:
            #     print('Zero vector:', binarystrings[J[ind]])
            
            #If no zero vectors in the first iteration itself, break the loop and keep a note
            if zero_indices==[]:
                self.zero_indices[tuple(indices)]=zero_indices
                model.setObjective(0.0) #reset the objective function to zero
                model.remove(model.getConstrs()) #remove all constraints            
                model.remove(model.getVars()) #remove all variables
                continue            
            #print('solution:', v)

            #Added the iterative method
            it_no=0
            while len(zero_indices)!=0:
                # print("Iteration number:", it_no) #Print iteration number

                # print("No. of zero vectors:", len(zero_indices))
                
                # for ind in zero_indices: #print zero indices
                #     print('Zero vector:', binarystrings[ind])
                
                model.reset() #reset the solved model to an unsolved state
                
                #Change the objective function

                model.setObjective(0.0) #First, remove the objective function
                
                for ind in zero_indices: #Now add variables corresponding to zero atoms only
                    x[ind].Obj=1.0
                
                #Solve the model
                model.update()
                model.Modelsense=GRB.MAXIMIZE
                model.write("LP.lp")
                model.optimize()                
                varslist=model.getVars()
                v=[]
                for vrbls in varslist:
                    v.append(vrbls.x)
                prev_zero_indices=zero_indices    
                #print("Vars list:", v)
                zero_indices=[i for i, Vars in enumerate(v) if Vars==0]
                #print("zero indices:", zero_indices)                
                non_zero_indices=[i for i, Vars in enumerate(v) if Vars!=0]

                it_no+=1

                if prev_zero_indices==zero_indices:
                    self.zero_indices[tuple(indices)]=zero_indices
                    zero_indices=[]
                    print("There are zero vectors in cluster")
                    #print("Cluster:", tuple(indices))
                    # print("Zero vectors:", self.zero_indices[tuple(indices)])
                    break
            
            if len(self.zero_indices.values())==0:
                zero_indices=[]
                print("There are no zero vectors in the entire model")

            #it_no+=1    

            '''Reset the Gurobi model for use in the next cluster iteration if required'''
            model.setObjective(0.0) #reset the objective function to zero
            model.remove(model.getConstrs()) #remove all constraints            
            model.remove(model.getVars()) #remove all variables
        
        print("All zero indices:", self.zero_indices)
    """

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
        
        num_total_vectors = 2**(num_feats)
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
        all_perms = itertools.product([0, 1], repeat=num_feats)
        # all_perms = list(itertools.product([0, 1], repeat=num_feats))[1:]

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
                # print("DATAVECTOR PARTITION", datavec_partition)
                c_matrix_partition = self.util_constraint_matrix(partition)
                # print("CONSTRAINT MATRIX PARTITION", c_matrix_partition)
                len_theta = datavec_partition.shape[0] 
                a = np.random.RandomState(seed=1)
                initial_val = a.rand(len_theta)


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

            
                # optimThetas = spmin_LBFGSB(func_objective, x0=initial_val,
                #                         fprime=None, approx_grad=True, 
                #                         disp=True, epsilon=1e-08) 
                optimThetas = minimize(func_objective, x0=initial_val, method='L-BFGS-B',
                    options={'disp':False, 'maxcor':20, 'ftol':2.2e-10, 'maxfun':100000})

                # Check if the LBFGS-B converges, if doesn't converge, then return error message
                if optimThetas.status != 0:
                    print(optimThetas.message)
                    return None
                
                solution[i] = optimThetas
                
                # norm_sol[i] = self.binary_norm_Z(optimThetas.x, partition)

                norm_sol[i] = np.exp(self.log_norm_Z(optimThetas.x, partition, c_matrix_partition))
                inn_arr = np.dot(c_matrix_partition, optimThetas.x)
                inn_arr = np.exp(inn_arr)
                inn_arr /= norm_sol[i]
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
        log_prob = 0.0
        parts = self.feats_obj.feat_partitions
        solution = self.opt_sol
        norm_sol = self.norm_z

        # partition will be a set of indices in the i-th parition        
        for i, partition in enumerate(parts):
            tmpvec = rvec[partition]

            if len(partition) == 1:
                term_exp = self.compute_constraint_sum(solution[i][0], tmpvec, partition)
            else:
                #convert zero indices to list of binary
                zeros_list = []
                for zero_atom in self.zero_indices[tuple(partition)]:
                    zeros_list.append(zero_atom)
                # for zero_atom in self.zero_indices[tuple(partition)]:
                #     zeros_list.append(format(zero_atom, '0'+str(len(partition))+'b'))

                # print('Zeros_list', zeros_list)
                zero_vec = tuple(tmpvec)
                # zero_vec = tmpvec.tolist()
                # zero_vec = ''.join(map(str,zero_vec))
                # print('Zeros_vec', zero_vec)

                if zero_vec in zeros_list:
                    return 0
                else:       
                    term_exp = self.compute_constraint_sum(solution[i].get('x'), tmpvec, partition)
                
            part_logprob = term_exp - np.log(norm_sol[i])
            log_prob += part_logprob
            # part_prob = np.exp(part_logprob)
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