#!/usr/bin/env python3
import subprocess	

for i in ['4','7','10']:
	for j in range(30,62):
		print("###################################################")
		print("PROCESSING: ", i, str(j))
		print("###################################################")
		subprocess.call(['python', 'rq1.1.1.py', i, str(j)], shell=True)
