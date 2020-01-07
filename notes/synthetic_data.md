# Synthetic Data Generation 

## Cluster generation 
Parameters that are considered are: 
1. Number of clusters (C): The number of clusters are chosen to reasonably divide diseases into multiple loosely correlated (largely uniform) clusters. 
| Diseases | Clusters |
|----------|----------|
| 4        | 2        |
| 7        | 2        |
| 10 	   | 3        |
| 15       | 3        |
Table (1): Assigned cluster size 
2. Number of clusters a disease can belong to (m): 
In the overlapping case, we see that a single disease can be picked from multiple clusters. The number of clusters a disease can belong to, is picked from a uniform distribution. 
3. Relative sizes of the clusters:
In the overlapping case, we can use another pmf to determine the relative sizes of the clusters. We control this by defining $\gamma$, which chooses each of the clusters in m with a probability p. (We set p = 0.5 and keep this distribution uniform). 

### Possible variations 
1. Number of clusters can be larger for diseases, such as 
| Diseases | Clusters |
|----------|----------|
| 4        | 2        |
| 7        | 3        |
| 10       | 4        |
| 15       | 6        |
Table(2): Variations in cluster size
We can test how increasing the number of clusters change the true disease distribution for the patient. We keep the other parameters (the number of clusters a disease can belong to, and relative size of the clusters as drawn from a uniform distribution.) 

## Patient Generation 
The parameters for generating a patient vector are:
1. Number of diseases a patient has (n): The number of diseases a patient vector has is chosen according to a truncated exponential distribution (with an exponential value of [0.8,1.2,1.6,2.0,2.4]. Note that this translates to a $\lambda$ of [0.42, 0.5, 0.62, 0.83, 1.25]). (Choice of $\lambda$ is motivated by observation of exponent in real-dataset) 
2. Choice of disease cluster for 'in-cluster' diseases (k): The choice of cluster depends on a truncated zipfian distribution (Current choice of 0.0 for generating zipfian distribution). 
3. Number of 'in-cluster' diseases (C): According to binomial(n,q1) (currently q1=0.75)


### Possible variations
1. Vary the distribution from which choice of cluster for 'in-cluster' diseases are picked. (zipf value can be in the range [0.0, 2.0, 4.0])
2. q1 = 0.25, 0.5, 0.75

## Experimental evaluation
### Experiment 1

#### Evaluation method 1
Generate synthetic data with the following variations - (1) Change in cluster size (2) Change in zipfian parameter for choice of in-cluster diseases (3) Choice of binomial(n,q1) for number of in-cluster diseases (4) Choice of binomial(n,q2) for number of exclusive diseases. 
Evaluate probability distribution after running maximum entropy with previously specified rules (for choosing number of constraints, etc.). Compare range of KL divergences between maximum entropy (M) and true distribution (R) to see if there is any change in behavior. 

For each variation, take average of 15 datasets to achieve comparable performance. 

#### Evaluation method 2
Generate synthetic data with the following variations - (1) Change in cluster size (2) Change in zipfian parameter for choice of in-cluster diseases (3) Choice of binomial(n,q1) for number of in-cluster diseases (4) Choice of binomial(n,q2) for number of exclusive diseases. 

##### Set 1
Cluster size as in Table 2 
q1 = q2 = p = 0.5 
z = 0.0 

(can use as ground truth for testing effect of varying cluster size)

#### Set 2
Cluster size as in Table 1
zipfian 0.0 2.0 4.0 
q1 = q2 = p = 0.25 0.5 0.75 
dataset_2 - dataset_10

#### Set 3
Cluster size as in Table 2
zipfian 0.0 2.0 4.0 
q1 = q2 = p = 0.25 0.5 0.75 
dataset_11 - dataset_19

#### Set 4
Cluster size as in Table 1
zipfian 0.0 
q1 = q2 = p = 0.5
dataset_20 - dataset_25

Find the KL divergence between the first and all other generated samples. 

Compare KL divergence of the true distribution of synthetic data (with uniform probability and lesser number of clusters) to other variations. If there exists a large KL divergence, we can assume the distributions are very different and then perform the rest of the experiments. 

Find where the greatest variations are by comparing what changes give different results. 

20 datasets with the first cluster values, and rest all set to uniform probabilities. (can be used as a baseline for all experiments)

Sum of probabilities remains constant, and only some probabilties vary. Can compare for some small distributions, say 10 

Final conclusions can be drawn by averaging over these variations.

## Conclusions (Subjective)
1. Max KL divergence is when all the parameters change, and for a very exponential distribution, we find the KL divergence to be 0.9. 
2. Sometimes, the interactions make the synthetic data generation process counteract against each other. 
3. Plot of all 9 different combinations 
4. Same parameters repeated multiple times will have the same true distribution but different synthetic data.  

Combinations - 
1. Cluster size (config 1, config 2)
2. z = [0.0, 2.0, 4.0]
3. q1 = [0.25, 0.5, 0.75]

18 possible combinations, 2 of each as there is variance in the cluster? 
