# -*- coding: utf-8 -*-
"""
We implement a script for tuning the performance of LWS algorithm over any specified range of threshold values

@author: stama
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
				print('th : ', th)
		
				simulated_max[str(np.round(th,2))] = []
				simulated_mean[str(np.round(th,2))] = []
				simulated_max3[str(np.round(th,2))] = []
				
				for i in df.keys():
					q = []
					for c in range(1, cmax+1):
						_ = df[i][c]
					
						for k in _:
							q.append(k[0])
							
					#d1 = pd.DataFrame.from_dict(df[i][c])
					d = pd.DataFrame()
					#d = pd.concat([d1], axis =1)
			
					#d.columns = ['label1']
					d['label1'] = q
					#print(d.shape)
					
					
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

def export_tuned_results(space, y_test_edited, export_csv = False):
    
    dd = {}
    for metric in ['max','mean','max3']:
    		
    		dd['metric=' + metric] = {}
    		accuracy, prec, rec, f1_macro, f1_weighted = [], [], [], [], []
    		if metric == 'max':
    				simulated = simulated_max
    		elif metric == 'mean':
    				simulated = simulated_mean
    		else:
    				simulated = simulated_max3
    		
    		for th1 in space:
    				predictions = np.array(simulated[str(np.round(th1,2))])
    				f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
    				f1_weighted.append(f1_score(y_test_edited, predictions, average = 'weighted'))
    				accuracy.append(acc(y_test_edited, predictions))
    				prec.append(precision_score(y_test_edited, predictions))
    				rec.append(recall_score(y_test_edited, predictions))
                    
    		print(max(accuracy) , max(prec), max(rec) , max(f1_macro) , max(f1_weighted))
            
    		dd['metric=' + metric]['acc'] = max(accuracy)
    		dd['metric=' + metric]['prec'] = max(prec)
    		dd['metric=' + metric]['rec'] = max(rec)
    		dd['metric=' + metric]['f1_macro'] = max(f1_macro)
    		dd['metric=' + metric]['f1_weighted'] = max(f1_weighted)
    		dd['metric=' + metric]['pos_acc'] = space[np.argmax(accuracy)]
    		dd['metric=' + metric]['pos_prec'] = space[np.argmax(prec)]
    		dd['metric=' + metric]['pos_rec'] = space[np.argmax(rec)]
    		dd['metric=' + metric]['pos_f1_macro'] = space[np.argmax(f1_macro)]
    		dd['metric=' + metric]['pos_f1_weighted'] = space[np.argmax(f1_weighted)]

    if export_csv:
        pd.DataFrame.from_dict(dd).to_csv('furn_' + name + '.csv')
    
    return


def manipulate_on_sentence_level(ddf1, saveplot = False):
    
    length = []
    max_similarity = []
    
    for i in ddf1.keys():
        length.append(len(ddf1[i]))
        max_similarity.append(max(ddf1[i]))
        
    df_max = pd.DataFrame()
    df_max['length'] = length
    df_max['label'] = y_test_edited
    df_max['max_similarity'] = max_similarity
    
    

    f, ax = plt.subplots(1, 1)
    
    negative = df_max[ df_max.iloc[:,1] == 0]
    positive = df_max[ df_max.iloc[:,1] == 1]
    sns.kdeplot(positive['max_similarity'], shade=True, color="r")
    sns.kdeplot(negative['max_similarity'], shade=True, color="b")
    ax.legend(['positive','negative'])
        
    if saveplot:
        plt.savefig('Biomineralization_density_plot.png' , dpi = 150)
    
    return df_max
    #df.iloc[:,2].plot()


def manipulate_on_instance_level(ddf1, saveplot = False):
    
    labels = []
    max_similarity_total = []
    counter = -1
    for i in ddf1.keys():
        counter += 1
        label = y_test_edited[counter]
        for j in ddf1[i]:
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
        plt.savefig('Biomineralization_density_plot_total.png' , dpi = 150)
        
    return df


def check():

    #%% plots of distributions
    df_max = manipulate_on_sentence_level(ddf1)
    df = manipulate_on_instance_level(ddf1)


    #%% gmms

    from matplotlib.colors import LogNorm
    from sklearn import mixture
    from sklearn.metrics import confusion_matrix

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full', init_params = 'kmeans')
    X = df.max_similarity_total
    X_train = np.array(X)
    clf.fit(X_train.reshape(-1,1))

    print(clf.means_)

    X_test = df_max.max_similarity
    #clf.predict(np.array(X_test).reshape(-1,1))
    #clf.predict(np.array(X_test).reshape(-1,1)) == df_max.label
    np.count_nonzero( clf.predict(np.array(X_test).reshape(-1,1)) == df_max.label )

    confusion_matrix(np.array(df_max.label), clf.predict(np.array(X_test).reshape(-1,1)))


#%%		
path = r'C:\Users\stam\Documents\git\Amulet-Setn\LWS'   
os.chdir(path)

names = [ 'density_Biomineralization_train_ratio_1_1_test_ratio_mixed_furnnew' , 'density_Chlorophyceae_train_ratio_1_1_test_ratio_mixed_furnnew' , 'density_Cytoglobin_train_ratio_1_1_test_ratio_mixed_furnnew']
mesh = ["Biomineralization" , "Chlorophyceae" , "Cytoglobin"]

names = names[1:2]
for name in names:
    space =  np.arange(0.65, 0.91, 0.01)
    
    with open(name + ".pickle", "rb") as f:
    				density_funk, predictions, y_test_edited = pickle.load(f)
    
    simulated_max , simulated_mean, simulated_max3 = read_pickles(path, density_funk, cmax = 3, space = space)
    
    y_test_edited = np.array(y_test_edited)
    
    export_tuned_results(space, y_test_edited, True)


# simulated_max['0.77'][0:10] == predictions[0:10]

