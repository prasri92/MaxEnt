import pandas as pd 
import numpy as np 

def main(dis_num):
	data = {}
	for i in range(20,26):
		data[i] = pd.read_csv('../output_s'+str(i)+'/expt1.2/d'+str(dis_num)+'.csv', error_bad_lines=False)
		data[i] = data[i].sort_values("KL Divergence").groupby("Lambda", as_index=False).first()
		data[i]['Dataset']=str(i)
	
	result = pd.concat(data, ignore_index=True)
	result = result.sort_values('Lambda')
	result.to_csv('d'+str(dis_num)+'_rq4.2.csv')

main(15)