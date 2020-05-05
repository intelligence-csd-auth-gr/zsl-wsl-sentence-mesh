# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:10:43 2020

@author: SpaceHorizon
"""

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt



def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score 

for i in simulated_max.keys():
    predictions = np.array(simulated_max[i])
    print(compute_roc(y_test_edited, predictions)[2])
    
#%%
predictions = np.array(simulated_max['0.79'])
predictions = np.array(simulated_max['0.77'])


sm_mode1 = simulated_max
sm_funk = simulated_max

predictions_mode1 = np.array(sm_mode1['0.77'])
predictions_funk = np.array(sm_funk['0.79'])
y_test_edited_mode1 = np.array(y_test_edited)
y_test_edited_funk = np.array(y_test_edited)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test_edited_mode1, predictions_mode1)
confusion_matrix(y_test_edited_funk, predictions_funk)




#%%

path = r'C:\Users\SpaceHorizon\OneDrive\auth\comet'   
os.chdir(path)


mesh = ["Biomineralization" , "Chlorophyceae" , "Cytoglobin"]

label = mesh[0]

with open('BERT_per_sentence_' + label +  '_' + 'train_ratio_1_1_test_ratio_mixed' + '_save_embs.pickle', 'rb') as f:
	x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2 = pickle.load(f)

f.close()


#%%

def change_labels(y_labels):

	new_y=[]
	for y in y_labels:
		if(y.__contains__(label)):
			new_y.append(1)
		else:
			new_y.append(0)

	return new_y

def check_lists(l,k):
    for i in range(0, len(k)):
        if l[i] != k[i]:
            return 'No'
    return 'Yes'
#%%
y_test_pickle = change_labels(y_test_new)
#%%

import pandas as pd

df = pd.DataFrame()
df['GrountTruth'] = y_test_pickle
df['Mode1'] = predictions_mode1
df['Funk'] = predictions_funk
df['y_text'] = y_test_new
df['x_tex'] = x_test_new

df.to_csv('examine_' + label + '.csv')
