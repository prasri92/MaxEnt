import pandas as pd 

data = pd.DataFrame(columns=['#_diseases','cluster_size','dataset_size','p','q1','z','local_file_num','exponent','beta','dataset_num'])

for i in range(1,101):
	for j in range(4,21,2):
		file_name = 'dataset_s'+str(i)+'_d'+str(j)+'.csv'
		d = pd.read_csv(file_name)
		d['dataset_num'] = i
		data = data.append(d, ignore_index=True)

data.to_csv('parameters.csv')
