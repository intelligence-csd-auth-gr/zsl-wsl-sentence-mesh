# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 19:03:38 2020

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

def read_pickles(path, ddf1, space = np.arange(0.65, 0.91, 0.01)):
		
		os.chdir(path)
		simulated_max , simulated_mean, simulated_max3, simulated_yolo = {}, {}, {}, {}

		for th1 in space:
				print('th : ', th1)
		
				simulated_max[str(np.round(th1,2))] = []
				simulated_mean[str(np.round(th1,2))] = []
				simulated_max3[str(np.round(th1,2))] = []
				simulated_yolo[str(np.round(th1,2))] = []
				
				for i in ddf1.keys():
						d1 = pd.DataFrame.from_dict(ddf1[i])
						d = pd.concat([d1], axis =1)
			
						d.columns = ['label1']
						#print(d.shape)
			
						l, m, n, k = [], [], [], []
						if d.max()[0] > th1:
							l.append(1)
						else:
							l.append(0)
						if d.mean()[0] > th1:
							m.append(1)
						else:
							m.append(0)
						if d.nlargest(3,'label1').mean()[0] > th1:
							n.append(1)
						else:
							n.append(0)
						if (d.label1.sum() - d.label1.max() - d.label1.min())/ (d.shape[0] - 2) > th1:
							k.append(1)
						else:
							k.append(0)
					 
						simulated_max[str(np.round(th1,2))].append(l)
						simulated_mean[str(np.round(th1,2))].append(m)
						simulated_max3[str(np.round(th1,2))].append(n)
						simulated_yolo[str(np.round(th1,2))].append(k)


		return  simulated_max , simulated_mean, simulated_max3, simulated_yolo

def export_tuned_results(space, y_test_edited, export_csv = False):
	
	dd = {}
	for metric in ['max','mean','max3','yolo']:
			
			dd['metric=' + metric] = {}
			prec, rec, f1_macro, f1_pos = [], [], [], []
			if metric == 'max':
					simulated = simulated_max
			elif metric == 'mean':
					simulated = simulated_mean
			elif metric == 'max3':
					simulated = simulated_max3
			else:
				simulated = simulated_yolo
						
			
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

	if export_csv:
		pd.DataFrame.from_dict(dd).to_csv(name + '_tuned.csv')
	
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
#%%		
path = r'C:\Users\stam\Documents\git\Amulet-Setn\DCbio-SentenceMax'

os.chdir(path)

names = [ 'SETN2020_DCbioSentenceMax_results_Biomineralization_train_ratio_1_1_test_ratio' , 'SETN2020_DCbioSentenceMax_results_Chlorophyceae_train_ratio_1_1_test_ratio' , 'SETN2020_DCbioSentenceMax_results_Cytoglobin_train_ratio_1_1_test_ratio']
mesh = ["Biomineralization" , "Chlorophyceae" , "Cytoglobin"]

name = names[2]
space = np.arange(0.65, 0.91, 0.01)

for name in names:
    
    with open(name + ".pickle", "rb") as f:
    				density, y_test_edited = pickle.load(f)
    
    simulated_max , simulated_mean, simulated_max3, simulated_yolo = read_pickles(path, density, space = space)
    
    y_test_edited = np.array(y_test_edited)
    
    export_tuned_results(space, y_test_edited, True)



#%%
    

#predictions_mode1 = simulated_max['0.77']
#naming = 'bio'

#predictions_mode1 = simulated_max['0.77']
#naming = 'chloro'

predictions_mode1 = simulated_max['0.77']
naming = 'cyto'

print(confusion_matrix(y_test_edited, predictions_mode1))


results_mode1 = []   

results_mode1.append(f1_score(y_test_edited, predictions_mode1, average = 'macro'))
results_mode1.append(f1_score(y_test_edited, predictions_mode1, average = 'binary' , pos_label=1))
results_mode1.append(precision_score(y_test_edited, predictions_mode1))
results_mode1.append(recall_score(y_test_edited, predictions_mode1))

results = pd.DataFrame()
results['scores'] = results_mode1

results.index = ['f1_macro','f1_pos', 'prec', 'recall']
results.to_csv(naming + '.csv')
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


#%% still not ready

# https://github.com/Zuu97/Unsupervised-Machine-Learning-From-Strach
# https://medium.com/@isurualagiyawanna/unsupervised-machine-learning-with-gaussian-mixture-models-ce2993e7061c
from scipy.stats import multivariate_normal


def initialize_param( X, K = 2, f = 1):
    N = X.shape[0]
    Mu = np.zeros((K, f))
    Cov = np.zeros((K,f, f))
    pi = np.ones(K)/K

    for k in range(K):
        idx = np.random.choice(N)
        Mu[k] = X[idx]
        Cov[k] = np.eye(f)
    Mu = Mu
    Cov = Cov
    return Mu, Cov, pi
    
    
    
def calculate_parameters(K, f, N, Mu, Cov, pi):
        Total_cost = []
        while True:
            R = np.zeros((N, K))
            for k in range(K):
                for n in range(N):
                    R[n,k] = pi[k] * multivariate_normal.pdf(X[n], Mu[k], Cov[k])

            cost = np.log(R.sum(axis=1)).sum()
            R = R / R.sum(axis=1, keepdims=True)

            R = R
            for k in range(K):
                Nk = R[:,k].sum()
                pi[k] = Nk / N
                Mu_k = R[:,k].dot(X)/Nk
                Mu[k] = Mu_k

                delta = X - Mu_k
                Rdelta = np.expand_dims(R[:,k], -1) * delta
                Cov[k] = Rdelta.T.dot(delta) / Nk + np.eye(f)*smoothing
            Total_cost.append(cost)
            if len(Total_cost) > 2:
                if np.abs(Total_cost[-2] - Total_cost[-1]) < minimal_cost_defference:
                    plt.plot(Total_cost)
                    plt.title("Total cost")
                    plt.show()

                    random_colors = np.random.random((K, 3))
                    colors = R.dot(random_colors)
                    plt.scatter(X[:,0], X[:,1], c=colors)
                    plt.title('GMM for K='+str(K))
                    plt.savefig(str(K)+'.png')
                    plt.show()
                    break
                
minimal_cost_defference = 0.1
smoothing = 1e-2
Mu, Cov, pi = initialize_param(X , 2, 1)
calculate_parameters(2, 1, X.shape[0], Mu, Cov, pi)