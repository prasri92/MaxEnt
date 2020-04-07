#!/bin/bash

# Check effectiveness of robust methods 
for i in 10 20 30 40 50 60
do 
	for j in 4 7 10 15
	do
		for k in 1 2 3 4 5
		do 
			for l in 0 1
			do 
				echo "PROCESSING DATASET: $i DISEASES: $j FILE NUMBER: $k PERT: $l"
				sbatch expt_sbatch_robust.sh $j $k $l $i
			done
		done
	done
done