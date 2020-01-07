#!/bin/bash

confvar=0
for ((dataset=26;dataset<=61;dataset++))
do
	if [ $((dataset%2)) -eq 0 ]
	then
		let "confvar+=1"
	fi
	for l in DISEASES_4 DISEASES_7 DISEASES_10 DISEASES_15
	do	
		# echo $l $confvar $dataset
		sbatch synthetic_data_job_2.sh $l $confvar $dataset
	done
done


# cd ../../
# for i in {26..61}
# do
# 	mkdir dataset_s$i
# 	cd dataset_s$i
# 	for k in 4 7 10 15
# 	do 
# 		mkdir d$k
# 	done 
# 	cd ../
# 	mkdir output_s$i 
# 	cd output_s$i
# 	for j in 4 7 10 15
# 	do 
# 		mkdir d$j 
# 		var="d$j"_expt1.2
# 		mkdir $var
# 	done 
# 	cd ../
# done