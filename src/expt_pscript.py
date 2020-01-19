#!/usr/bin/env python3
import subprocess	

for i in ['4','7','10']:
	for j in ['1', '2', '3', '4', '5']:
		# for k in ['0', '1']:
		print("###################################################")
		print("PROCESSING: ", i, j)
		print("###################################################")
		subprocess.call(['python', 'rq3.2.py', i, j], shell=True)
