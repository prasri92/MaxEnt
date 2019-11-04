#!bin/bash
for i in 4 7 10 15
do
for j in 3 
do
for k in 0 1
do
python rq2.2.py $i $j $k
done
done 
done