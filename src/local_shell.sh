#!bin/bash
for ((i=1;i<=25;i++))
do
python main_kl_test.py $i >> outfiles/out_$i.txt
done 

