# -*- coding: utf-8 -*-
"""
Plot distribution diagrams per examined MeSH term as it concerns the sentence-based proposed approach

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

def manipulate_on_word_level(d, w, y_test_edited, label, path_save, saveplot = False):
    
    length = []
    max_similarity = []
    
    for i in d.keys():
        length.append(len(d[i][w]))
        max_similarity.append(max(d[i][w])[0])
        
    df_max = pd.DataFrame()
    df_max['length'] = length
    df_max['label'] = y_test_edited
    df_max['max_similarity'] = max_similarity
    
    f, ax = plt.subplots(1, 1)
    
    negative = df_max[ df_max.iloc[:,1] == 0]
    positive = df_max[ df_max.iloc[:,1] == 1]
    
    if label == 'Cytoglobin':
        sns.kdeplot(positive['max_similarity'], shade=True, bw = 0.05, color="r")
    else:
        sns.kdeplot(positive['max_similarity'], shade=True, color="r")
        
    sns.kdeplot(negative['max_similarity'], shade=True, color="b")
    ax.legend(['positive','negative'])
    plt.title(label + ' using_ ' + str(w) + ' _words')
    
    os.chdir(path_save)
    if saveplot:
        plt.savefig(label + '_density_plot_using ' + str(w) + ' window size.png' , dpi = 150)
    
    return df_max
	
def bring_y_test(label, path):
	
	current_path = os.getcwd()
	os.chdir(path)
    
	names = os.listdir(os.getcwd())
	#print(names)
	
	for name in names:
		if label in name and '.pickle' in name:
			break
	
	with open(name , "rb") as f:
		density, y_test_edited, _, _ = pickle.load(f)
	
	os.chdir(current_path)
	
	return density, y_test_edited
    
#%% cell of the main code
    
path = '..\Amulet-Setn\LWS'
path_save = '..\Amulet-Setn\plots'


labels = ["Biomineralization" , "Chlorophyceae" , "Cytoglobin"]

for _ in labels:
	
    os.chdir(path)
    print('Exmined MeSH term: ', _)

    name = 'density_' + _ + '_train_ratio_1_1_test_ratio.pickle'
    
    with open(name, "rb") as f:
        density, predictions, y_test_edited, time_LWS  = pickle.load(f)
    f.close()
    
    s = 0
    for i in density.keys():
        for j in density[i].keys():
             s += len(density[i][j])
    
    print(_, " : ", s, " calls of bioBERT")
    
    # needs bw on positive plot of cyto for j = 1
    for j in density[i].keys():
        manipulate_on_word_level(density, j, y_test_edited, _, path_save, saveplot = True)
        
#%% Print the number of bioBERT calls for the proposed sentence-based and the compared state-of-the-art word-based method

path_y = '..Amulet-Setn\DCbio-SentenceMax'   

for _ in labels:
    
    print('Exmined MeSH term: ', _)

    density, y_test_edited = bring_y_test(_, path)

    name = 'density_' + _ + '_train_ratio_1_1_test_ratio.pickle'
    
    with open(name, "rb") as f:
        d, predictions, y_test, time_LWS = pickle.load(f)
    
    s = 0
    for i in d.keys():
        for j in d[i].keys():
             s += len(d[i][j])
    
    print(_, " : ", s, " calls of bioBERT with Funk")
    
    s = 0
    for i in density.keys():
        s += len(density[i])
    
    print(_, " : ", s, " calls of bioBERT with mode1")
    
print('End')
