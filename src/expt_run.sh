#!/bin/bash
# for i in {1..19}
# do
# 	for j in 4 7 10 15
# 	do
# 		for k in 1 2 4 5
# 		do
# 			python rq3.2.py $j $k $i 
# 		done
# 	done
# done

for i in 4 7 10 15
do 
	for j in 1 2 3 4 5
	do
		python rq1.1.py $i $j
	done
done