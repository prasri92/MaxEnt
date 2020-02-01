import pandas as pd 
import csv

data = pd.DataFrame(columns=['#_diseases','file_num','tot_zeros','zv_nzemp','zv_zemp','nzv_zemp','dataset_num'])

for i in range(1,62):
	for j in ['4','7','10']:
		file_name = 'dataset_s'+str(i)+'_d'+j+'_app.csv'
		d = pd.read_csv(file_name)
		d['dataset_num'] = i
		data = data.append(d, ignore_index=True)

data.to_csv('zero_results_approx.csv')