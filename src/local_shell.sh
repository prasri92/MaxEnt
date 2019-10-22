#!bin/bash
for i in 3 13 23
do
for j in 0.01 0.02 0.03 0.05 0.1 0.2
do
python main_maxent_perturb.py $i $j 
done 
done

