# -*- coding: utf-8 -*-
"""
We implement a script for tuning the performance of LWS algorithm over any specified range of threshold values

@authors:
Nikos Mylonas   myloniko@csd.auth.gr
Stamatis Karlos stkarlos@csd.auth.gr
Grigorios Tsoumakas greg@csd.auth.gr
"""

import os
import numpy as np
import pandas as pd
import pickle


from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score as acc

import seaborn as sns
import matplotlib.pyplot as plt

def read_pickles(path, df, cmax, space = np.arange(0.65, 0.91, 0.01)):
		
		os.chdir(path)
		simulated_max , simulated_mean, simulated_max3 = {}, {}, {}

		for th in space:
				print('th : ', np.round(th,2))
		
				simulated_max[str(np.round(th,2))] = []
				simulated_mean[str(np.round(th,2))] = []
				simulated_max3[str(np.round(th,2))] = []
				
				for i in df.keys():
					q = []
					for c in range(1, cmax+1):
						_ = df[i][c]
					
						for k in _:
							q.append(k[0])
							
					d = pd.DataFrame()
					d['label1'] = q
				
					l, m, n = [], [], []
					if d.max()[0] > th:
						l.append(1)
					else:
						l.append(0)
					if d.mean()[0] > th:
						m.append(1)
					else:
						m.append(0)
					if d.nlargest(3,'label1').mean()[0] > th:
						n.append(1)
					else:
						n.append(0)
					 
					simulated_max[str(np.round(th,2))].append(l)
					simulated_mean[str(np.round(th,2))].append(m)
					simulated_max3[str(np.round(th,2))].append(n)
			
		return  simulated_max , simulated_mean, simulated_max3

def export_tuned_results(space, y_test_edited, path, path_save, export_csv = False):
	
	dd = {}
	for metric in ['max','mean','max3']:
			
			dd['metric=' + metric] = {}
			prec, rec, f1_macro, f1_pos = [], [], [], []
			if metric == 'max':
					simulated = simulated_max
			elif metric == 'mean':
					simulated = simulated_mean
			else:
					simulated = simulated_max3
			
			for th1 in space:
					predictions = np.array(simulated[str(np.round(th1,2))])
					f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
					f1_pos.append(f1_score(y_test_edited, predictions, average = 'binary' , pos_label=1))
					prec.append(precision_score(y_test_edited, predictions))
					rec.append(recall_score(y_test_edited, predictions))
					
			print(max(prec), max(rec) , max(f1_macro) , max(f1_pos))
			
			dd['metric=' + metric]['prec'] = max(prec)
			dd['metric=' + metric]['rec'] = max(rec)
			dd['metric=' + metric]['f1_macro'] = max(f1_macro)
			dd['metric=' + metric]['f1_pos'] = max(f1_pos)
			dd['metric=' + metric]['pos_prec'] = space[np.argmax(prec)]
			dd['metric=' + metric]['pos_rec'] = space[np.argmax(rec)]
			dd['metric=' + metric]['pos_f1_macro'] = space[np.argmax(f1_macro)]
			dd['metric=' + metric]['pos_f1_pos'] = space[np.argmax(f1_pos)]

	os.chdir(path_save)
	if export_csv:
		pd.DataFrame.from_dict(dd).to_csv('SETN2020_' + name + 'tuned.csv')
	os.chdir(path)

	return

#%%	cell of main code 	
path = '..\Amulet-Setn\LWS'   
os.chdir(path)

names = [ 'density_Biomineralization_train_ratio_1_1_test_ratio' , 'density_Chlorophyceae_train_ratio_1_1_test_ratio' , 'density_Cytoglobin_train_ratio_1_1_test_ratio']
mesh = ["Biomineralization" , "Chlorophyceae" , "Cytoglobin"]

# define the examined space of values for the th parameter
space = np.arange(0.65, 0.91, 0.01)

for name in names:
    
    with open(name + ".pickle", "rb") as f:
    				density_funk, predictions, y_test_edited, time = pickle.load(f)
    
    simulated_max , simulated_mean, simulated_max3 = read_pickles(path, density_funk, cmax = 3, space = space)
    
    y_test_edited = np.array(y_test_edited)
    
    export_tuned_results(space, y_test_edited, path, path + '\\tuning', True)


#%% Get the confusion matrix for any desired threshold/mode/MeSH term

os.chdir(path + "\\tuning")

predictions_mode1 = simulated_max['0.8']
naming = 'cytoFurn'

print(confusion_matrix(y_test_edited, predictions_mode1))

results = []   

results.append(f1_score(y_test_edited, predictions_mode1, average = 'macro'))
results.append(f1_score(y_test_edited, predictions_mode1, average = 'binary' , pos_label=1))
results.append(precision_score(y_test_edited, predictions_mode1))
results.append(recall_score(y_test_edited, predictions_mode1))

results_df = pd.DataFrame()
results_df['scores'] = results

results_df.index = ['f1_macro','f1_pos', 'prec', 'recall']
results_df.to_csv(naming + '.csv')