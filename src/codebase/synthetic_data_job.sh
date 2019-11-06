#!/bin/bash
i=20
for j in 0.0 2.0 4.0
do
	for k in 0.25 0.5 0.75
	do
		for l in DISEASES_4 DISEASES_7 DISEASES_10 DISEASES_15
		do	
			sbatch synthetic_data_job_2.sh $l $i $k $j 
		done
		let "i+=1"
	done
done