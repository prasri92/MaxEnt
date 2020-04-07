import pandas as pd 
import csv

data = pd.DataFrame(columns=['#_diseases','file_num','tot_zeros','zv_nzemp','zv_zemp','nzv_zemp','time_taken_sec','dataset_num'])

for i in range(50, 56):
	for j in ['15']:
		file_name = 'dataset_s'+str(i)+'_d'+j+'.csv'
		d = pd.read_csv(file_name)
		d.rename(columns={'t':'time_taken_sec'},inplace=True)
		d['dataset_num'] = i
		print(d)
		data = data.append(d, ignore_index=True)

data.to_csv('zero_results.csv')