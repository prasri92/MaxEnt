import pandas as pd 
import numpy as np 
import csv
import matplotlib.pyplot as plt
import sys

def vis_1(diseases, dataset_num, file_num):
	# Vis 1: Plot the counts of data 
	file_name = '../dataset_s'+str(dataset_num)+'/d'+str(diseases)+'/synthetic_data_expt'+str(file_num)+'.csv'
	d = pd.read_csv(file_name)
	d['sum'] = d.sum(axis=1)
	counts_1 = d['sum'].value_counts().sort_index()
	counts_1.plot(kind='bar')
	plt.title("Empirical counts of number of diseases per patient")
	plt.xlabel('Number of Diseases')
	plt.ylabel('Counts')
	plt.show()

def vis_2(diseases, dataset_num, file_num):
	# Vis 2: Plot the counts of unique combination of rows 
	file_name = '../dataset_s'+str(dataset_num)+'/d'+str(diseases)+'/synthetic_data_expt'+str(file_num)+'.csv'
	d = pd.read_csv(file_name)
	d['val'] = d[d.columns[:]].apply(
	    lambda x: ''.join(x.dropna().astype(str)),
	    axis=1
	)
	# d['val'] = d['val'].apply(lambda x: int(x, base=2))
	counts_2 = d['val'].value_counts().sort_index()
	counts_2.plot(kind='bar')
	plt.title("Empirical counts of unique disease combinations in the data\nTotal Combinations : "+str(2**int(diseases))+"\nCombinations Present : "+str(len(counts_2)))
	plt.xlabel('Unique combinations')
	plt.ylabel('Counts')
	plt.show()

if __name__ == '__main__':
	dataset_num = sys.argv[2]#57
	file_num = sys.argv[3] #2
	diseases = sys.argv[1] #7
	vis = sys.argv[4] #1
	if vis == str(1):
		vis_1(diseases, dataset_num, file_num)
	elif vis == str(2):
		vis_2(diseases, dataset_num, file_num)
	
