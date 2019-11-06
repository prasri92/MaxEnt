#!bin/bash
i=11
for j in 0.0 2.0 4.0
do
	for k in 0.25 0.5 0.75
	do
		for l in DISEASES_4 DISEASES_7 DISEASES_10 DISEASES_15
		do	
			python synthetic_data.py $l $i $k $j
		done
		let "i+=1"
	done
done

# to create directory
# cd ../../
# for j in 11 12 13 14 15 16 17 18 19
# do
# 	mkdir dataset_s$j
# 	cd dataset_s$j
# 	for k in 4 7 10 15
# 	do
# 		mkdir d$k
# 	done
# 	cd ../
# done
 

		
