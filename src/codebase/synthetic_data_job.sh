#!/bin/bash

# EARLIER VERSION
# confvar=0
# for ((dataset=26;dataset<=61;dataset++))
# do
# 	if [ $((dataset%2)) -eq 0 ]
# 	then
# 		let "confvar+=1"
# 	fi
# 	for l in DISEASES_4 DISEASES_7 DISEASES_10 DISEASES_15
# 	do	
# 		# echo $l $confvar $dataset
# 		sbatch synthetic_data_job_2.sh $l $confvar $dataset
# 	done
# done

# FOR DISEASES = 4 - 16
# for ((dataset=1;dataset<101;dataset++))
# do
# 	for i in 4 6 8 10 12 14 16
# 	do
# 		echo $i $dataset
# 		sbatch synthetic_data_job_2.sh $i $dataset
# 	done
# done

# FOR DISEASES 18-20, FIRST CALL THIS, TO CREATE PARAMETERS
# for larger datasets, place each generation as a separate process
# for ((dataset=1;dataset<101;dataset++))
# do 
# 	for i in 18 20
# 	do
# 		echo $i $dataset
# 		sbatch synthetic_data_job_3.sh $i $dataset 
# 	done
# done 

# FOR DISEASES 18-20, CALL NEXT TO START THE PROCESS
# Calls a separate thread for each of the 10 datasets to be generated
for ((dataset=1;dataset<101;dataset++))
do 
	for i in 18 20
	do
		echo $i $dataset
		for j in {1..10}
		do
			sbatch synthetic_data_job_4.sh $i $dataset $j
		done
	done 
done 

# cd ../../
# for i in {21..40}
# do
# 	cd data
# 	mkdir dataset_s$i
# 	cd dataset_s$i
# 	for k in 4 6 8 10 12 14 16 18 20
# 	do 
# 		mkdir d$k
# 	done 
# 	cd ../../output
# 	mkdir output_s$i 
# 	cd output_s$i
# 	for j in 4 6 8 10 12 14 16 18 20
# 	do 
# 		mkdir d$j 
# 		var="d$j"_expt1.2
# 		mkdir $var
# 	done 
# 	cd ../../
# done