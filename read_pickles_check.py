# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:43:46 2020

@author: stam
"""

import os

import pickle

#%%
os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\bioBERT embeddings')

name = 'bioBERTemb_train_ratio_1_4_test_ratio_Biomineralization'

with open(name +  '.pickle' , 'rb') as f:
			x_train_bert, x_test_bert, time_preprocess1, x_train_new, x_test_new, y_train_new, y_test_new = pickle.load(f)
f.close()

#%%

os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\bioBERT embeddings profile per sentence')

name = 'bioBERT_profile_per_sentence_Biomineralization_train_ratio_1_1_test_ratio'

with open(name +  '.pickle' , 'rb') as f:
    	x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2 = pickle.load(f)
f.close()

#%%
os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\DCbio-SentenceMax')


name = 'SETN2020_DCbioSentenceMax_results_Biomineralization_train_ratio_1_1_test_ratio'


with open(name +  '.pickle' , 'rb') as f:
    	density, y_test_edited1 = pickle.load(f)
f.close()

#%%

os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\LWS')


name = 'density_Biomineralization_train_ratio_1_1_test_ratio.pickle'
    
with open(name, "rb") as f:
    density, predictions, y_test_edited, time_LWS  = pickle.load(f)
f.close()