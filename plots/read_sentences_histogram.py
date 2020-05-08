# -*- coding: utf-8 -*-
"""


@authors:
Nikos Mylonas   myloniko@csd.auth.gr
Stamatis Karlos stkarlos@csd.auth.gr
Grigorios Tsoumakas greg@csd.auth.gr
"""


import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
import numpy as np

from matplotlib.colors import LogNorm
from sklearn import mixture
from sklearn.metrics import confusion_matrix


def change_labels(y_labels, label):

	new_y=[]
	for y in y_labels:
		if(y.__contains__(label)):
			new_y.append(1)
		else:
			new_y.append(0)

	return new_y


def manipulate_on_sentence_level(df, y, case, label, ratio, saveplot = False):
	
	length = []
	max_similarity = []
	reject = []
	c = 0
	for i in df.keys():
		if df[i] == []:
			reject.append(c)
			continue
		else:
			length.append(len(df[i]))
			max_similarity.append(max(df[i]))
		c += 1
		
	df_max = pd.DataFrame()
	df_max['length'] = length
	df_max['label'] = np.delete(y, reject)
	df_max['max_similarity'] = max_similarity
	
		
	f, ax = plt.subplots(1, 1)
	
	negative = df_max[ df_max.iloc[:,1] == 0]
	positive = df_max[ df_max.iloc[:,1] == 1]
	sns.kdeplot(positive['max_similarity'], shade=True, color="r")
	sns.kdeplot(negative['max_similarity'], shade=True, color="b")
	ax.legend(['positive','negative'])
		
	if saveplot:
		plt.savefig(label + '_density_plot_max_variant' + '_' + case + '_' + ratio + '.png' , dpi = 300)
	
	return df_max


def manipulate_on_instance_level(df,  y, case, label_name, ratio, saveplot = False):
	
	labels = []
	max_similarity_total = []
	reject = []
	c = 0
	for i in df.keys():
		
		if df[i] == []:
			reject.append(c)
			c+=1
			continue
		
		label = y[c]
		
		c += 1
		
		for j in df[i]:
			max_similarity_total.append(j)
			labels.append(label)
		
	
	df = pd.DataFrame()
	df['max_similarity_total'] = max_similarity_total
	df['labels'] = labels
	
	
	f, ax = plt.subplots(1, 1)
	
	negative = df[ df.iloc[:,1] == 0]
	positive = df[ df.iloc[:,1] == 1]
	sns.kdeplot(positive['max_similarity_total'], shade=True, color="r")
	sns.kdeplot(negative['max_similarity_total'], shade=True, color="b")
	ax.legend(['positive','negative'])
	
	if saveplot:
		plt.savefig(label_name + '_density_plot_total' + '_' + case + '_' + ratio + '.png' , dpi = 300)
		
	return df
	
def bring_pickle(label, path):
	
	current_path = os.getcwd()
	os.chdir(path)
	names = os.listdir(os.getcwd())
	print(names)
	for name in names:
		if label in name:
			break
	
	if '1_1' in name:
		ratio = '1_1'
	else:
		ratio = '1_4'
    
	with open(name, "rb") as f:
		df_train, df_test, y_train, y_test, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sentences_train, total_calls_test, total_sentences_test], time_computation = pickle.load(f)
	f.close()
    
	os.chdir(current_path)
	
	return df_train, df_test, y_train, y_test, total_calls_train, total_sentences_train, total_calls_test, total_sentences_test, time_computation, ratio



def check_empty_lists(l):
	counter = 0
	for i in l:
		if i == []:
			counter +=1
	return counter


def gmms():


	for _ in labels:

		# fit a Gaussian Mixture Model with two components
		clf = mixture.GaussianMixture(n_components=2, covariance_type='full')#, init_params = 'kmeans')
		X = df[_][0].max_similarity
		#X = df[_][2].max_similarity_total
		
		X_train = np.array(X)
		clf.fit(X_train.reshape(-1,1))
		
		print(clf.means_ , np.mean(clf.means_))
		
		#X_test = df[_][3].max_similarity_total
		X_test = df[_][1].max_similarity

		#clf.predict(np.array(X_test).reshape(-1,1))
		#clf.predict(np.array(X_test).reshape(-1,1)) == df_max.label
		
		#np.count_nonzero( clf.predict(np.array(X_test).reshape(-1,1)) == df[_][3].labels )
		print(X.shape, X_test.shape)
		#print(confusion_matrix(np.array(df[_][3].labels), clf.predict(np.array(X_test).reshape(-1,1))))
		print(confusion_matrix(np.array(df[_][1].label), clf.predict(np.array(X_test).reshape(-1,1))))
	return
		

#%%
path = r'C:\Users\stam\Documents\git\Amulet-Setn\bioBERT embeddings profile per sentence'


labels = ["Biomineralization" , "Chlorophyceae" , "Cytoglobin"]
df = {}
mplot = True

for _ in labels:
	
	df_train_profile, df_test_profile, y_train, y_test, total_calls_train, total_sentences_train, total_calls_test, total_sentences_test, time_computation, ratio = bring_pickle(_, path)
	
	print('Train instances: ', check_empty_lists(df_train_profile) , len(df_train_profile))
	print('Test instances: ', check_empty_lists(df_test_profile), len(df_test_profile))

	s = 0
	for i in df_train_profile.keys():
		s += len(df_train_profile[i])
	
	print(_, " : ", s, " calls of bioBERT during training")
	print(_, " : ",  sum(total_sentences_train), " cases of bioBERT during training")

	s = 0
	for i in df_test_profile.keys():
		s += len(df_test_profile[i])
	
	print(_, " : ", s, " calls of bioBERT during test")
	print(_, " : ",  sum(total_sentences_test), " cases of bioBERT during test")

	
	#df_max_train = manipulate_on_sentence_level(df_train_profile, change_labels(y_train, _), 'train', _, ratio, mplot)
	df_max_test = manipulate_on_sentence_level(df_test_profile, change_labels(y_test, _), 'test', _, ratio, mplot)

	#df_train = manipulate_on_instance_level(d_train_profile, y_train, 'train', _, ratio, mplot)
	df_test = manipulate_on_instance_level(df_test_profile, change_labels(y_test, _), 'test', _, ratio, mplot)

	#df[_] = [df_max_train, df_max_test, df_train, df_test]
	



