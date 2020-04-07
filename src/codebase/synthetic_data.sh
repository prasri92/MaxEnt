#!/bin/bash

# create dataset and output directories 
#cd ../../
#mkdir -p output
#mkdir -p dataset
#cd output
#for i in 4 7 10 15
#do 
#	mkdir d$i
#done
#cd ../dataset
#for i in 4 7 10 15
#do 
#	mkdir d$i
#done

cd ../../
for i in 20 30 40 50 60
do
	cd output_s$i
	for j in 4 7 10 15 
	do
		dis=d$j
		expt=expt2.2
		name=${dis}_${expt}
		mkdir $name
	done
	cd ../
done

# generate synthetic data 
# for d in DISEASES_4 DISEASES_7 DISEASES_10 DISEASES_15
# do
# 	python synthetic_data.py $d
# done
