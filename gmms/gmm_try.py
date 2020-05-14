# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:12:11 2020

@author: stam
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

from matplotlib.colors import LogNorm
from sklearn import mixture
from sklearn.metrics import confusion_matrix



def call_gmm(X, y, naming):
    
    os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\gmms')
    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full', init_params = 'kmeans', random_state = 23)
   
    X_train = np.array(X)
    X_train = np.round(X_train,4)
    clf.fit(X_train.reshape(-1,1))
    
    print(clf.means_)
    
    X_test = y
    X_test = np.round(X_test,4)
    
    #np.count_nonzero( clf.predict(np.array(X_test).reshape(-1,1)) == df_max.label )
    ground_truth = np.array(df_max.label)
    predictions = clf.predict(np.array(X_test).reshape(-1,1))
    
    results_gmm = []
    print(confusion_matrix(ground_truth, predictions))
    
    results_gmm.append(f1_score(ground_truth, predictions, average = 'binary', pos_label = 1))
    results_gmm.append(precision_score(ground_truth, predictions))
    results_gmm.append(recall_score(ground_truth, predictions))
    results_gmm.extend([clf.means_[0], clf.means_[1], np.round(np.mean(clf.means_), 3)])
    results = pd.DataFrame()
    results['scores'] = results_gmm

    results.index = ['f1_pos', 'prec', 'recall', 'center0', 'center1', 'average of centers']
    results.to_csv(naming + '.csv')
    
    return clf.means_



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


path = r'C:\Users\stam\Documents\git\Amulet-Setn\DCbio-SentenceMax'

os.chdir(path)

names = [ 'SETN2020_DCbioSentenceMax_results_Biomineralization_train_ratio_1_1_test_ratio' , 'SETN2020_DCbioSentenceMax_results_Chlorophyceae_train_ratio_1_1_test_ratio' , 'SETN2020_DCbioSentenceMax_results_Cytoglobin_train_ratio_1_1_test_ratio']
mesh = ["Biomineralization" , "Chlorophyceae" , "Cytoglobin"]

space = np.arange(0.65, 0.91, 0.01)
counter = 0
for name in names:
    
    with open(name + ".pickle", "rb") as f:
    				density, y_test_edited = pickle.load(f)
        
    y_test_edited = np.array(y_test_edited)
    

    df_max = manipulate_on_sentence_level(density)
    df = manipulate_on_instance_level(density)
    
    
    #X = df.max_similarity_total
    X = df_max.max_similarity
    y = df_max.max_similarity
    print(name, X.shape)
    centers = call_gmm(X,y, mesh[counter])
    print(np.mean(centers))
    
    counter+=1
    os.chdir(path)



