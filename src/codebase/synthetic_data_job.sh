#!/bin/bash

for i in {1..10}
do 
	for l in DISEASES_4 DISEASES_7 DISEASES_10 DISEASES_15
	do	
		sbatch synthetic_data_job_2.sh $l $i 
	done
done