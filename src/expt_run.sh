#!/bin/bash

# for i in 4 6 8 10 12
# do 
# 	for j in {1..100}
# 	do
# 		counter=1
# 		echo $i $j $counter 
# 		for k in 0.001 0.01 0.019 0.028 0.037 0.046 0.055 0.064 0.073 0.082 0.091 0.1
# 		do 
# 			sbatch expt_sbatch.sh $i $j $counter $k
# 			let "counter+=1" 
# 		done
# 	done
# done

# for i in 14
# do 
# 	for j in {1..100}
# 	do
# 		for k in {1..10}
# 		do 
# 			counter=1
# 			echo $i $j $counter $k
# 			for l in 0.001 0.0045 0.0080 0.0116 0.0151 0.0187 0.0222 0.0258 0.0293 0.0329 0.0364 0.04
# 			do 
# 				# echo $i $j $counter $l $k
# 				sbatch expt_sbatch_2.sh $i $j $counter $l $k
# 				let "counter+=1" 
# 			done
# 		done
# 	done
# done

# for i in 16
# do 
# 	for j in {1..100}
# 	do
# 		for k in {1..10}
# 		do 
# 			counter=1
# 			echo $i $j $counter $k
# 			for l in 0.001 0.0045 0.0080 0.0116 0.0151 0.0187 0.0222 0.0258 0.0293 0.0329 0.0364 0.04
# 			do 
# 				# echo $i $j $counter $l $k
# 				sbatch expt_sbatch_2.sh $i $j $counter $l $k
# 				let "counter+=1" 
# 			done
# 		done
# 	done
# done

# for i in 18
# do 
# 	for j in {1..100}
# 	do
# 		for k in {1..10}
# 		do 
# 			counter=1
# 			echo $i $j $counter $k 
# 			for l in 0.001 0.0036 0.0062 0.0089 0.0115 0.0141 0.0168 0.0194 0.0220 0.0247 0.0273 0.03
# 			do 
# 				# echo $i $j $counter $l $k
# 				sbatch expt_sbatch_2.sh $i $j $counter $l $k
# 				let "counter+=1" 
# 			done
# 		done
# 	done
# done

for i in 20
do 
	for j in {1..100}
	do
		for k in {1..10}
		do 
			counter=1
			echo $i $j $counter $k 
			for l in 0.005 0.0072 0.0095 0.0118 0.01409 0.0163 0.0186 0.02090 0.02318 0.02545 0.02772 0.03
			do 
				# echo $i $j $counter $l $k
				sbatch expt_sbatch_2.sh $i $j $counter $l $k
				let "counter+=1" 
			done
		done
	done
done
