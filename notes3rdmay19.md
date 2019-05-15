# Discussion 
1. Choose the parameters for different synthetic data, and find the synthetic patient data (same number of rows as existing patient data) 
2. Find which fits the best, and fix those parameters

# Regularization 
1. two different regularizations schemes because we cannot do cross-validation 

# Numeric Trick
1. Log trick for really small probabilities (take log of the probabilities and sum them up and then take exponent of it)


# TO DISCUSS
1. Truncated Exponential with an upper and lower bound set. Check how to control the scale variable as well as how to set b. (Current hack, set alpha = 2 * num_diseases so that b = 1/lambda = 2 or lambda = 0.5)
