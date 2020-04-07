from __future__ import division
from collections import defaultdict, OrderedDict
import operator
import numpy as np

from pyitlib import discrete_random_variable as drv

"""TODO
- better documentation
    - give math formulas for the variables defined in the functions
    - where applicable, explain the code woriking with one line comments
"""

class ExtractFeatures(object):
    """ Class summary
    Extract the relevant feature pairs from a given numpy data array to form
    the constraints for the maximum-entropy optimization algorithm. Currently it
    has methods to deal with discrete binary data arrays.

    Give extended description of extraction procedure here (see math classes
    implementation in python for reference on documentation)

    Attributes:
        data_arr: A numpy array (binary) for the disease prevalence data
        ent_estimator: String indicating which entropy estimator to use from the
            the `pyitlib` library. Default is 'JAMES-STEN'
        K: Total number of constraints to find for maxent optimization
        N: Total number of training examples in the dataset
        L_measure_dict: Dict to store the value of normalized L-measures 
            between different feature pairs.
        feats_pairs_dict: Dict to store the the top K feature pairs along with 
            their values to be used for the constraints.
        feat_graph: Dict to store the transitive graph induced by the feature
            pairs in feats_pairs_dict. Adjacency list representation is used.
        feat_partitions: List to store the partitions (connected components)
            found in the feature graph. Made up of lists containing indices for 
            each partition which have the feature(column) numbers.
    """

    def __init__(self, dataArray, entropy_estimator='JAMES-STEIN', topK=5, Mu=None):
        self.data_arr = dataArray
        self.ent_estimator = entropy_estimator
        self.K = topK   # number of feature pairs to extract
        self.N = self.data_arr.shape[0] # Number of training data examples
        
        self.L_measure_dict = {}

        #order the values for use in piecewise likelihood
        self.two_way_dict = OrderedDict()
        self.three_way_dict = OrderedDict()
        self.four_way_dict = OrderedDict() 

        # self.two_way_dict = {}  
        # self.three_way_dict = {}
        # self.four_way_dict = {}
        
        self.feat_graph = {}             
        self.feat_partitions = [] 

        self.Mu = Mu
        self.supportdict = {}  


    def set_two_way_constraints(self, ext_two_way_dict):
        self.two_way_dict = ext_two_way_dict


    def set_three_way_constraints(self, ext_three_way_dict):
        self.three_way_dict = ext_three_way_dict
    

    def set_four_way_constraints(self, ext_four_way_dict):
        self.four_way_dict = ext_four_way_dict

    def set_supports(self, ext_supportdict):
        self.supportdict=ext_supportdict


    def compute_discrete_Lmeasure(self):
        """Function to compute the un-normalized L-measure between the all the 
        discrete feature pairs. The value for all the possible pairs is stored
        in the L_measures dict. Auxiliary values like the mutual information
        (I_mutinfo) are also in their respective dicts for all the possible pairs.        
        This method sets the `feats_pairs_dict` class attribute.

        Args:
            None
        
        Returns:
            None
        """
        # TAKE note: the function expects the array to be in a transpose form
        indi_entropies = drv.entropy(self.data_arr.T, estimator=self.ent_estimator)
        # indi_entropies = drv.entropy(self.data_arr.T)
        num_rand = self.data_arr.shape[1]  # Number of random variables (feature columns)
        assert num_rand == len(indi_entropies)

        L_measures = {}     # Dictionary storing the pairwise L-measures
        I_mutinfo = {}      # Dictionary storing the pairwise mutual information
        # mu_vals = {}        # Dictionary storing the pairwise MU values

        for i in range(num_rand):
            for j in range(i+1, num_rand):
                key = (i, j)    # since 0-indexed
                h_i = indi_entropies[i]
                h_j = indi_entropies[j]
    
                # mu_ij = self.get_discrete_mu(i, j)            

                # Potential error: I_ij may come out negative depending on the estiamtor   
                I_ij = drv.information_mutual(self.data_arr.T[i], self.data_arr.T[j], estimator=self.ent_estimator)                
                W_ij = min(h_i, h_j)
                
                num = (-2.0 * I_ij * W_ij)
                den = (W_ij - I_ij)
                eps = 1e-9   # epsilon value for denominator
                inner_exp_term = num/(den + eps)                              
                # removing numerical errors by upper bounding exponent by 0
                inner_exp_term = min(0, inner_exp_term)
                
                L_measures[key] = np.sqrt(1 - np.exp(inner_exp_term))
                I_mutinfo[key] = I_ij                

                # print(I_ij, W_ij, num, den)
                # print(key, L_measures[key], inner_exp_term)
                # print('\n')

        
        self.L_measure_dict = L_measures
        return


    def compute_topK_feats(self):   
        """ Function to compute the top-K feature pairs and their corresponding 
        feature assignment from amongst all the pairs. Approximate computation: 
        Select the top K pairs based on their L_measures values. For each pair 
        just select the highest scoring feature assignment. Score is calculated
        by $\delta(x_i, y_j)$. 
        
        This method sets the `feats_pairs_dict` class attribute.

        Args:
            None

        Returns:
            None 
        """

        # First, run the method for setting the Lmeasures dictionary with 
        # appropriate values.        
        # self.compute_discrete_norm_Lmeasure() # Only use it for multi-discrete
        
        print("Computing the L_measures between the feature pairs")
        self.compute_discrete_Lmeasure() # Use it for binary case
        
        print("Sorting the L_measures")
        # This sorted list will also be useful in approximate partitioning 
        # by dropping the lowest L(x,y) pairs of edges in the feat-graph.
        sorted_list = sorted(self.L_measure_dict.items(), 
                                key=operator.itemgetter(1),
                                reverse=True)


        # Just consider the top-K pairs of features first. This will ensure that
        # you will get at least K exact feature pairs (x_i, y_j) from the list.
        # each entry is a tuple of (key, value). We just want the keys
        topK_keys = [item[0] for item in sorted_list[:self.K]]        
        val_dict = {}        

        print("Computing the topK pairs")
        # tuple_list = []
        for k_tuple in topK_keys:
            i = k_tuple[0]
            j = k_tuple[1]    
            
            # Do this for computing when multi-valued discrete features 
            # involved. Not needed for binary.
            # set_xi = set(self.data_arr[:,i])
            # set_yj = set(self.data_arr[:,j])
            set_xi = [0,1]
            set_yj = [0,1]

            # CHOOSING JUST A SINGLE maxima PAIR of values
            # Can update to include multiple later on
            maxima = 0.0    
            for xi in set_xi:
                for yj in set_yj:
                    b_i = self.data_arr[:,i] == xi
                    b_j = self.data_arr[:,j] == yj
                    n_i = sum(b_i) # CAN BE pre-fetched
                    n_j = sum(b_j) # CAN be pre-fetched
                    # n_i = counts[(i, xi)]
                    # n_j = counts[(j, yj)]
                    n_ij = sum(b_i & b_j)
                    
                    # print(i,j, xi, yj, n_i, n_j, n_ij)
                    delta_ij = np.abs( (n_ij / self.N) * np.log((n_ij * self.N) / (n_i * n_j)) )

                    if delta_ij > maxima :
                        maxima = delta_ij
                        val_dict[k_tuple] = (xi, yj)

        print(k_tuple, (xi, yj), maxima)
        
        # set the two_way_dict to the val dict
        # self.two_way_dict = val_dict
        self.set_two_way_constraints(val_dict)
        


    def util_add_edges(self, graph, edge_tup):
        # graph is the dictionary for the partition-graph
        for t in edge_tup:
            for t_ot in edge_tup:
                if t != t_ot:
                    graph[t].add(t_ot)

    def util_remove_edges(self, graph, edge_tup):
        for t in edge_tup:
            for t_ot in edge_tup:
                if t!=t_ot:
                    if t_ot in graph[t]:
                        graph[t].remove(t_ot)

   
    def create_partition_graph(self):
        """Function to create a graph out of the feature pairs (constraints)
        Two nodes (feature indices) have an edge between them if they appear in 
        a constraint together. The graph is an adjacency list representation
        stored in the graph dictionary.
        
        This method sets the class attribute `feat_graph` to the graph dict

        Args: 
            None

        Returns:
            None
        """
        graph = {}  # undirected graph
        num_feats = self.data_arr.shape[1]

        # init for each node an empty set of neighbors
        for i in range(num_feats):
            graph[i] = set()

        # print("Creating the feature graph")
        # create adj-list representation of the graph
        
        for tup_2way in self.two_way_dict.keys():
            # Here tup_2way is a tuple of feature indices            
            # print("Added edge for:", tup_2way)
            self.util_add_edges(graph, tup_2way)

        if len(self.three_way_dict) != 0:
            for tup_3way in self.three_way_dict.keys():
                # Here tup_3way is a triplet of feature indices
                # print("Added edge for:", tup_3way)
                self.util_add_edges(graph, tup_3way)
        # else:
        #     print("No 3 way constraints specified")

        if len(self.four_way_dict) != 0:
            for tup_4way in self.four_way_dict.keys():
                # Here tup_3way is a triplet of feature indices
                # print("Added edge for:", tup_4way)
                self.util_add_edges(graph, tup_4way)
        # else:
            # print("No 4 way constraints specified")            

        print()
        self.feat_graph = graph
        # return graph
    

    def partition_features(self):        
        """Function to partition the set of features (for easier computation).
        Partitoning is equivalent to finding all the connected components in 
        the undirected graph of the features indices as their nodes.         
        This method find the partitions as sets of feature indices and stores 
        them in a list of lists with each inner list storing the indices 
        corresponding to a particular partition.

        This method sets the class attribute `feats_partitions` which is list of
        lists containing the partition assignments.

        Args:
            None
        
        Returns:
            None
        """
        self.create_partition_graph()
        print("Partioning the feature graph", end=' ')

        def connected_components(neighbors):
            seen = set()
            def component(node):
                nodes = set([node])
                while nodes:
                    node = nodes.pop()
                    seen.add(node)
                    nodes |= neighbors[node] - seen
                    yield node
            for node in neighbors:
                if node not in seen:
                    yield component(node)

        partitions = []
        print("and Finding the connected components:")
        for comp in connected_components(self.feat_graph):
            partitions.append(list(comp))

        if self.Mu!=None: #if max. cluster size specified
            self.Mu=int(self.Mu) #makes sure the value of Mu is specified by the user
            # print("Checking if clusters are smaller than maximum specified size")        
            approx_clusters=[i for i in partitions if len(i)>self.Mu] #list of clusters that need to be approximated
            self.feat_partitions=list(filter(lambda x: len(x)<self.Mu, partitions)) 
            if len(approx_clusters)==0: #checks if all cluster sizes are below th threshold
                print("All clusters are smaller than the maximum threshold, hence the exact solution holds")
                self.feat_partitions=partitions
            else:    
                print("Approximating the following clusters through forced partitioning:\n", approx_clusters)

                
                """ This section creates the dictionary 'edgetracker' associating each edge 
                with a cluster index in approx_clusters"""
                edgetracker={}
                for i in self.two_way_dict:
                    x, y = i
                    counter=0
                    for cluster in approx_clusters:
                        if x in cluster:
                            edgetracker[(x, y)]=counter                                                    
                        else:
                            counter+=1
        
                for i in self.three_way_dict:
                    x, y, z = i
                    counter=0
                    for cluster in approx_clusters:
                        if x in cluster:
                            edgetracker[(x, y, z)]=counter                                                    
                        else:
                            counter+=1
                    
                for i in self.four_way_dict:
                    x, y, z, a = i
                    counter=0
                    for cluster in approx_clusters:
                        if x in cluster:
                            edgetracker[(x, y, z, a)]=counter
                        else:
                            counter+=1
                
                """ Calculate marginal probabilities and load into a dictionary 
                for each disease in approx_clusters"""
                marginalsvals=np.sum(self.data_arr, axis=0)
                size = self.data_arr.shape[0]
                marginalsvals.tolist()   
                marginalsvals = marginalsvals/size
                marginals={} #dict stores all marginal probabilities of diseases in approx_clusters
                for a in approx_clusters:
                    for j in a:
                          marginals[j]=marginalsvals[j]    
                approximated=[] #list to store the approximated cluster pieces from the original cluster
                broken_edges=[] #list to store all edges in the graph that were broken
        
                """Loop inside approx_clusters with the forced partitioning implementation 
                for each cluster"""
                clusterindex=0 #initial value of cluster index
                while clusterindex in range(len(approx_clusters)):
                    edges=[k for k, v in edgetracker.items() if v==clusterindex] #all edges associated with a cluster that needs approximation
                    T=[tuple(e) for e in edges] #convert to tuple               
                    edge_supportvalues=[self.supportdict[x] for x in T] #get corresponding support values for edges
                    edge_supportdict=dict(zip(T, edge_supportvalues)) #store these values with corresponding edges in the dictionary
                    N=[i for i in approx_clusters[clusterindex]] #all nodes in the given cluster, stored in a list. 'N' in the paper
                    N_n=[frozenset([i]) for i in N] #N_i in the paper, initialized and stored as a sequential list of frozensets N0, N1, N2 etc
                    T_n=[set([i]) for i in N_n]  #T_i in the paper, initialized stored as a sequential list of set of frozensets T0, T1, T2 etc
                    """Calculate delta values and store in a dictionary"""
                    deltadict={} #a dictionary to store delta values, calculated from support and marginal probabilities
                    
                    """Delta values are calculated for each edge using supports
                    from edge_supportdict. as detailed in page 64 section 5.2"""
                    for e in edge_supportdict:
                        l=np.empty((1, len(e)))
                        counter=0
                        for i in e:                        
                            l[0, counter]=marginals[i]
                            counter+=1
                        product_marginals=np.prod(l)    
                        delta_e=min(edge_supportdict[e]/product_marginals, product_marginals/edge_supportdict[e])
                        deltadict[e]=delta_e
                    T=sorted(deltadict.keys(), key=lambda d:deltadict[d]) #List of edges sorted in ascending value of deltas for sequential processing

        
                    """ This is the start of the forced partitioning algorithm, Loop inside T to process the edges in descending values of supports, keep track of indices for N_n to perform set unions"""
                    i=0
                    while len(T)!=0: #line 5 in figure 5: T!=Phi
                        X=T[-1]  #this is the edge with the highest delta; Line 5 in figure 5: Select X belong to T
                        T.pop() #remove X from T, line 7 in figure 5
                        minX=min(X) #this is the smallest index in edge Xl line 9 in figure 5
                        nlist=[] #this stores the indices of all elements in N_n that is also in X        
        
                        
                        for x in X:
                              for n in N_n:
                                  if x in n:
                                      nlist.append(N_n.index(n)) 
        
                        nsets=[N_n[i] for i in nlist] 
                        tsets=[T_n[i] for i in nlist]          
                        Nunion=frozenset().union(*nsets) #line 11, this is union of N_j
                        Tunion=set().union(*tsets) #line 11, this is the Union of T_j        
                        #minindex=min(nlist) #this is the smallest element in X intersect N_n
                        if len(Nunion)<=self.Mu: #line 8, max. size of approx. clusters
                            for i in X:
                                if i==minX:
                                    for n in N_n:
                                        if i in n:
                                            indxmin=N_n.index(n) #this is index in N_n corresponding to the smallest disease index                                            
                                            N_n[indxmin]=Nunion #line 11: assign N_union to the smallest index
                                            T_n[N_n.index(Nunion)]=set.union(Tunion, set(X)) #line 11: assign T_Union Union X corresponding to indxmin
                                    
                                else: 
                                    for n in N_n:
                                         if i in n and minX not in n:
                                             indx=N_n.index(n)
                                             N_n[indx]=frozenset([]) #line 13: N_i=Phi
                                             del N_n[indx]
                        
                        else:
                            broken_edges.append(X)

                    # print("BROKEN EDGES", broken_edges)
                                                       
                    N_n= [list(N) for N in N_n if N!={}] #filter out empty cluster to feed the optimizer
                    for L in N_n:
                            approximated.append(L)             
                    clusterindex+=1
                self.feat_partitions.extend(approximated)
#                print("The clusters after approximation are:", self.feat_partitions)
                
                """ Remove all edges that were broken from the graph, twowaydict, threewaydict and fourwaydict"""
                for edge in broken_edges:
                    self.util_remove_edges(self.feat_graph, edge)
                    if len(edge)==2:
                        del self.two_way_dict[edge]
                    if len(edge)==3:
                        del self.three_way_dict[edge]
                    if len(edge)==4:
                        del self.four_way_dict[edge]
                
                """Re-add edges from two way dict, three way dict and four way dict"""
                for tup_2way in self.two_way_dict.keys():
                    self.util_add_edges(self.feat_graph, tup_2way)

                if len(self.three_way_dict) != 0:
                    for tup_3way in self.three_way_dict.keys():
                        self.util_add_edges(self.feat_graph, tup_3way)

                if len(self.four_way_dict) != 0:
                    for tup_4way in self.three_way_dict.keys():
                        self.util_add_edges(self.feat_graph, tup_4way)


        else:
            self.feat_partitions = partitions
        
        return 