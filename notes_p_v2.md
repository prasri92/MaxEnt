# Chronic Disease Modeling - Pracheta B A
# July 30, 2019

List of things to-do
1. Clean up code in main_kl_test.py main() function. Move all data cleaning and processing methods to load_disease_data function. Perform the following checks - 
	If removing zeros, perform zero vector removal on the dataset 
	Check for diseases that do not occur, if there is such a case, remove that disease from the entire dataframe 
2. Once zero atom detection is achieved, modify code to ensure that the zero atoms are removed from the probability calculations. Modify the code for all_perms to run for all probabilities, except certain vectors. Print out statement to check if there are zero atoms present in the LP 
3. In the zero removal case, the total probability calculation for the entire dataset is wrong, correct the same. 
4. Once the zero diseases are removed, and the number of diseases are determined, update the visualization code to dynamically take in the number of diseases to be input to the plots 
5. In the zero removal case (non-zeros), plot f to see why the function is ill-behaved, to see which of the two mentioned reasons is giving an abnormal behavior 
6. In zero removal case, and otherwise, if there is an error, print correct error messages and do not display probabilities
7. Run code for many diseases and few constraints and for 20 diseases to print the output of LP 
8. Add zero constraints for 4, 10 diseases and check
9. Support value is too large for some cases, write exception handling code

Features added in main_kl_test 
1. load disease data, clean it
2. print zero vectors for LP solution(exact, approx)
3. add constraint for 0 diseases

Features added in main_kl_test_v2
1. load disease data, clean it 
2. print zero vectors LP solution (exact, approx)

Features checked
1. zero removal probabilities seem to be correct (some numerical errors experienced)
2. check non zeros 
3. run experiments for 4, 10 diseases 
4. plot them 

Done
1. Merging approximate maxent
