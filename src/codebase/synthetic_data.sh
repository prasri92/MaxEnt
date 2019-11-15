#!bin/bash

for i in 1 2 3 4 5
do 
	for j in DISEASES_4 DISEASES_7 DISEASES_10 DISEASES_15
	do
		python synthetic_data.py $j $i 
	done
done

# i=11
# for j in 0.0 2.0 4.0
# do
# 	for k in 0.25 0.5 0.75
# 	do
# 		for l in DISEASES_4 DISEASES_7 DISEASES_10 DISEASES_15
# 		do	
# 			python synthetic_data.py $l $i $k $j
# 		done
# 		let "i+=1"
# 	done
# done

# to create directory
# cd ../../
# for j in 20 21 22 23 24 25
# do
# 	mkdir dataset_s$j
# 	cd dataset_s$j
# 	for k in 4 7 10 15
# 	do
# 		mkdir d$k
# 	done
# 	cd ../
# done
 
# cd ../../
# for j in 20 21 22 23 24 25
# do
# 	cd output_s$j
# 	for k in 4 7 10 15
# 	do
# 		newdir="expt1.2"
# 		mkdir $newdir
# 	done
# 	cd ../
# done
	
# for i in 20 21 22 23 24 25
# do 
# 	for j in 4 7 10 15
# 	do
# 		for k in 1 2 3 4 5
# 		do
# 			python rq1.2.py $j $k $i
# 		done
# 	done
# done  

