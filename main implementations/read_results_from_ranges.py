# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:03:26 2020

@author: SpaceHorizon
"""

import os
import numpy as np
import pandas as pd

mode = 4

if mode == 3:
    os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\WDCbio(bioBERT)')
    text = 'mode_3_after_range.csv'
else:
    os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\WDCbio(tfidf)')
    text = 'mode_4_after_range.csv'

labels = ["Biomineralization" , "Chlorophyceae" , "Cytoglobin"]
scenario = ['train_ratio_1_4_test_ratio',  'train_ratio_1_1_test_ratio']
learners = ['SVC', 'LinearSVC', 'LR']

space = np.arange(0.73, 0.83, 0.01)
results = {}
for i in os.listdir(os.getcwd()):
    for lea in learners:
        #print('Learner: ', lea)
        if lea in i:
            for sc in scenario:
                #print('Scenario: ', sc)
                if sc in i:
                    for label in labels:
                        if label in i:
                            print(i)
                            x = pd.read_csv(i)
                            y = x.f1_macro
                            results[lea + '_' + label + '_' + sc] = [y.max(), space[y.idxmax()]]


#os.chdir(r'C:\Users\SpaceHorizon\OneDrive\auth\comet')
pd.DataFrame.from_dict(results).to_csv(text)