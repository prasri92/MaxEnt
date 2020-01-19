#!/bin/bash

# cd ../ 
# for i in {26..61}
# do
# 	cd output_s$i
# 	mkdir expt1.2
# 	cd ../
# done


# for i in 4 7 10 15
# do 
# 	for j in 1 2 3 4 5
# 	do
# 		for k in {26..61}
# 		do
# 			sbatch expt_sbatch.sh $i $j $k 
# 		done
# 	done
# done

# for i in 4 7 10 15
# do 
# 	for j in {1..61}
# 	do
# 		python vis_rq1.2.py $i $j
# 	done
# done

# for i in 10 15
# do 
# 	for j in {1..5}
# 	do
# 		for k in 0 1
# 		do
# 			python rq2.1.py $i $j $k
# 		done
# 	done
# done

for i in 4 7 10 
do
	for j in {1..5}
	do
		python rq1.2.py $i $j 
	done
done