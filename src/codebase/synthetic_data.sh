#!bin/bash
i=3
for j in 0.8 2.4 4.0
do
	for k in 2.0 
	do
		python synthetic_data.py $j $k $i
		let "i+=10" 
	done
done
