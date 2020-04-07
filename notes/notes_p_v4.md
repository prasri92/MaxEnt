# Discussion (3/4/2020)

## Highlights 
1. Learned Supports - discuss the tuning parameters in the synthetic data
2. Learned relation wrt power divergence measure with 3 factors (number of diseases, fitted exponent and dataset size) without removing zero vectors 
3. Improved the results by a small margin 
4. Used polynomial regression with degree 2 for final output 
5. Observed slightly better results in a small setting, have to test for many different datasets. 
6. Ran the regularization experiment (with previously used values of box width) and found that regularization does indeed help. 
7. Caveat - Power divergence (divide by zero error (replaced zeros by 1e-300))

## Things to do - 3/18
1. Which box constraint method performs better? Try for normal support (not the learned support) (box-width with previously specified width or with the CI)
2. Single width - after trying out a range of widths, we find that the single width parameter of W = 1 is best, implying that A_i = B_i = 1/L where L is the size of the dataset. (rq2.2.py tries different values of W to select the best width) 
3. rq2.1.py - box-constraints with width parameter W = 1 (chosen width is much better, plot results side by side) W=1 is better than W=2 also. Also play around with the smaller widths, see the implications on the probabilities that are assigned to the vectors. 
4. rq2.3.py - box-constraints with CI of 95% as the A_i and B_i (not as good as using the width parameter)
5. What is robust optimizer doing to the zeros? Run an experiment where we find the zeros using the normal optimizer, then compute the probabilities with the normal optimizer and the robust optimizer so we can compare the results. Done, compare the value of the prob assigned to zero vectors. 
6. Generate data with large dataset size to find the learned support
7. Learn the support 