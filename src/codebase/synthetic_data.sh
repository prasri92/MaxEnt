#!bin/bash
i=1
for j in 0.8 1.6 2.4 3.2 4.0
do
	for k in 0.0 1.0 2.0 3.0 4.0 
	do
		python synthetic_data.py $j $k $i
		let "i+=1" 
	done
done
