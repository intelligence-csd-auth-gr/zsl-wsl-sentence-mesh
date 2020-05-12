# -*- coding: utf-8 -*-
"""
Implementation of the proposed Dataless and Weakly Supervised Learning algorithms as they are published on SETN2020 along with the corresponding baseline and the LWS algorithm, enriched with the bioBERT embeddings as it is described on:  

@authors:
Nikos Mylonas	myloniko@csd.auth.gr
Stamatis Karlos	stkarlos@csd.auth.gr
Grigorios Tsoumakas	greg@csd.auth.gr

This work is part of AMULET research program funded by ...
"""



import os
import random
import re
import time
import torch
import copy
import pickle

import numpy as np
import pandas as pd
from scipy.spatial import distance

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score as acc
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier as Grad
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import AdaBoostClassifier as ADA
from biobert_embedding.embedding import BiobertEmbedding

import time
import spacy

spacy.load('en_core_web_sm')
nlp = spacy.load("en_core_web_sm") #nlp = spacy.load('en')

import warnings
warnings.filterwarnings("ignore")

def prepare_X_Y(txt):

		items=[]
		with open(txt) as text:
				for line in text:
						items.append(line[2:-2])

		X,Y = [], []
		random.shuffle(items)
		counter = 0
		for item in items:
				if(item.__contains__(" labels: #")):
						counter += 1
						X.append(item.split(" labels: #")[0])
						Y.append(item.split(" labels: #")[1])
		if counter != len(items):
				print('Check message inside prepare_X_Y: ', counter, ' ', items)

		return X,Y

	
def prepare_Y_01():

		y_train_occ = []
		for i in range(0,len(x_train)):

			if( x_train[i].__contains__(label) or x_train[i].__contains__(label.lower()) ):
						y_train_occ.append(y_train[i]+"#"+label)
			else:
						y_train_occ.append(y_train[i])

		return y_train_occ

def my_predictor(biobert, x_test, threshold = 0.76):

	y_pred=[0 for i in range(len(x_test))]
	for i in range(0,len(x_test)):
		if((1-distance.cosine(np.array(biobert.sentence_vector(label)),
																	np.array(biobert.sentence_vector(x_test[i][:1000]))))>threshold):
			y_pred[i]=1
		else:
			y_pred[i]=0

	return y_pred

def change_labels(y_labels, label):

	new_y=[]
	for y in y_labels:
		if(y.__contains__(label)):
			new_y.append(1)
		else:
			new_y.append(0)

	return new_y

def get_embeddings(biobert, x_set, measuretime = True, abstractLen = 1000):

	if measuretime:
		start = time.time()
	new_x=[]
	
	for i in range(0,len(x_set)):
		new_x.append(np.array(biobert.sentence_vector(x_set[i][:abstractLen])))
		#new_x.append(np.array(torch.stack(biobert.word_vector(label))))

	if measuretime:
		end = time.time()
		print(np.round(end - start,3), ' seconds')

	return new_x

def augment_y_with_embeddings(biobert, label, x_train, y_train, command, threshold, qx = None):
 
	label_embedding= np.array(torch.stack(biobert.word_vector(label)))
	#print('1: ',label_embedding.shape, '2: ', x_train[0].shape)
	y_all= {}
	reject_th = []


	for th in threshold:
		counter = 0

		print('th: ', threshold)

		y_all[th] = []
		y = []
		if command == 1:

			for i in range(0, len(x_train)):
				#command1
				# na ta valo tora
				#array_embedding = detect_embedding(x_train[i][0:1000], qx)
				#total_sim=(1-distance.cosine(label_embedding, array_embedding))  

				# ayto trexei tiora
				total_sim=(1-distance.cosine(label_embedding, np.array(biobert.sentence_vector(x_train[i][:1000]))))  

				#print('total sim: ' , total_sim)

				if(total_sim>th):
						y.append( y_train[i]+"#"+label )
						counter += 1
				else:
						y.append( y_train[i] )

			if counter == 0 or counter == len(x_train):
				reject_th.append(th)		
			y_all[th] = y

		elif command == 2:

				for i in range(0, len(x_train)):
					
					#command2
					total_sim=(1-distance.cosine(label_embedding,x_train[i])) 
					#print('total sim: ' , total_sim)

				
					if(total_sim>th):
							y.append( y_train[i]+"#"+label )
							counter += 1
					else:
							y.append( y_train[i] )
			 
				if counter == 0 or counter == len(x_train):
					reject_th.append(th)	
				y_all[th] = y

		else:
				return 'Error'
			
	return y_all, reject_th

def occurence(x_train, y_train, label):

	y_train_occ = []
	for i in range(0,len(x_train)):
			
			if( x_train[i].__contains__(label) or x_train[i].__contains__(label.lower()) ):
						#print(i)
						y_train_occ.append(y_train[i]+"#"+label)
			else:
						y_train_occ.append(y_train[i])

	return y_train_occ

def tfidf(x_train, y_train, x_test):

	# standard NLP case
	tfidf= TfidfVectorizer(stop_words='english')
	tfidf.fit(x_train)
	x_train_tf = tfidf.transform(x_train)
	x_test_tf  = tfidf.transform(x_test)

	return x_train_tf, x_test_tf

def remove_symbols(text):

	new=[]
	for i in range(0,len(text)):
		 item=re.sub(r'[^a-zA-Z\']', ' ', text[i])
		 item= re.sub('\s+', ' ', item)
		 new.append(item.lower())

	return new


def remove_words(query,stopwords):

		results = []
		for _ in query:
				querywords = _.split()
				resultwords  = [word for word in querywords if word.lower() not in stopwords]
				result = ' '.join(resultwords)
				results.append(result)

		return results

def train_classifier(lea, x_train, y_train, x_test, y_test):

	# feed a classifier

	#lea=SVC(kernel='linear', C=100)

	#y_train_01=change_labels(y_train_bert)
	#y_test_01=change_labels(y_test)


	lea.fit(x_train, y_train)
	y_pred = lea.predict(x_test)

	f1=f1_score(y_pred, y_test)#, average = 'macro')
	
	print(np.round(f1,3))

	return y_pred


def read_embeddings(x,filename):

	data=pd.read_csv(filename)
	new_x=[]
	for i in range(0,len(data['Abstract'])):
		if (data['Abstract'][i] in x):
			new_x.append(data['Embedding'][i])

	return new_x

def detect_embedding(q, qx):
	flag = False
	for i in range(0, qx.shape[0]):
		if q == qx.iloc[i][0:1000]:
			flag = True
			break
	if flag:
		z = []
		t = qx.iloc[i][1].split('\n')
		for _ in t:
			for k in _.split():
				z.append(k)
		z.remove('[')
		z[-1] = z[-1][:-1]

		zz = []
		for _ in z:
			zz.append(float(_))

		w = np.array(zz)
	else:
		print('Error in detection')
	return w


def my_predictor_per_sentence(biobert, x_test, label_emb, threshold = 0.76):

	y_pred=[0 for i in range(len(x_test))]
	print(' th value that is examined: ', threshold)

	for i in range(0,len(x_test)):
		average_dist=0
		count=0
		text = x_test[i]
		tokens = nlp(text)
		for sent in tokens.sents:
			if len(sent.string.strip())>10 : #(len(sent)>5):
				#if (len(sent) == 59):
				#	continue
				print(len(sent))
				count+=1
				#print(sent.string.strip())
				dist1=1-distance.cosine(label_emb, np.array(biobert.sentence_vector(sent.string.strip())))
				average_dist+=dist1
				#print(dist1)
		average_dist=average_dist/count

		if average_dist > threshold: 
			y_pred[i]=1
		else:
			y_pred[i]=0

	return y_pred

def my_predictor_per_sentence_max(biobert, x_test, label_emb, threshold = 0.76):

	y_pred=[0 for i in range(len(x_test))]
	print(' th value that is examined: ', threshold)

	for i in range(0,len(x_test)):
		max1 = 0
		text = x_test[i]
		tokens = nlp(text)
		for sent in tokens.sents:
			if(len(sent)>5):
				if (len(sent) == 59):
					continue
				#print(len(sent))
				#print(sent.string.strip())
				dist1=1-distance.cosine(label_emb, np.array(biobert.sentence_vector(sent.string.strip())))
				if dist1 > max1:
					max1 = dist1
				#print(dist1)

		if max1 > threshold: 
			y_pred[i]=1
		else:
			y_pred[i]=0

	return y_pred

def my_predictor_per_sentence_max_loaded_embs(x, threshold = 0.77):

	df = {}
	for th in threshold:
		
		y_pred=[0 for i in range(len(x))]
		df[th] = []
		print(' th value that is examined: ', th)
		c = 0
		for i in x.keys():
			#print(c)
			#print(i)
			#print(i, max(x[i]))
			if len(x[i]) > 0 : 
				if  max(x[i]) > th: 
					y_pred[c]=1
				#else:
				#	y_pred[c]=0
			else:
				print(i)
			c+=1
		df[th] = y_pred

	return df

def my_predictor_save_old(biobert, x_test, label_emb, th = 0.77):
  
	y_pred=[0 for i in range(len(x_test))]

	#label_embedding1= np.array(biobert.sentence_vector(label1))#np.array(torch.stack(biobert.word_vector(label1)))[0]
	#label_embedding2= np.array(biobert.sentence_vector(label2))#np.array(torch.stack(biobert.word_vector(label2)))[0]
	#nlp = spacy.load('en')
	df1= {}
	#count=0

	for i in range(0,len(x_test)):
		average_dist=0
		text = x_test[i]
		tokens = nlp(text)
		max1=0
		df1['label1_instance' + str(i+1)] = []
		
		for sent in tokens.sents:
			if(len(sent)>5 and len(sent)!=59):
				dist1=1-distance.cosine(label_emb,np.array(biobert.sentence_vector(sent.string.strip())))
				if(dist1>=max1):
					 max1=dist1
			else:
				continue
			df1['label1_instance' + str(i+1)].append(dist1)
		
		if(max1>th):
			y_pred[i]=1
		else:
			y_pred[i]=0

	#count += 1

	return y_pred, df1
#%%

def my_predictor_save(biobert, x, label_emb):
  
	
	df = {}
	reject = []
	total_calls, total_sent = [] , []
	
	for i in range(0,len(x)):
		if i % 100 == 0:
			print(i, x[i][0:20])

		c, calls = 0, 0
		text = x[i]
		tokens = nlp(text)
		Max = 0
		
		df['label1_instance:' + str(i)] = []
		
		for sent in tokens.sents:
			
			c += 1
			if(len(sent.string.strip())>10):
				
				calls += 1
				dist = 1-distance.cosine(label_emb,np.array(biobert.sentence_vector(sent.string.strip())))
				
				if(dist >= Max):
					 Max = dist
				df['label1_instance:' + str(i)].append(dist)
				
			else:
				#print('Here')
				#reject.append(i)
				continue
		
		total_calls.append(calls)
		total_sent.append(c)

		
	#examines if all the sentences are empty
	for i in df.keys():
		if df[i] == []:
			print('Empty sentence: ', i)
			reject.append(i[ i.find(':') + 1 : ])
			del df[i]
	
	reject = list(set(reject))
	#print(reject)

	t = np.delete( np.array(x) , reject)
	
	t_new = []
	for i in list(t):
		t_new.append(str(i))

	return df, t_new, total_calls, total_sent, x, reject
	#return y_pred, df, reject, x
#%%
def augment_y_with_embeddings_max(biobert, label, label_embedding, x_train, y_train, command, threshold, qx = None):
 
	#label_embedding= np.array(torch.stack(biobert.word_vector(label)))
	#print('1: ',label_embedding.shape, '2: ', x_train[0].shape)
	y_all= {}
	reject_th = []


	for th in threshold:
		counter = 0

		#print('th: ', threshold)

		y_all[th] = []
		y = []
		if command == 1:
			pass

		elif command == 2:

				for i in range(0, len(x_train[th])):
					
					#command2
					#total_sim=(1-distance.cosine(label_embedding,x_train[i])) 
					#print('total sim: ' , total_sim)

					# without th
					if(x_train[th][i]>th):
							y.append( y_train[i]+"#"+label)
							counter += 1
					else:
							y.append( y_train[i] )
			 
				if counter == 0 or counter == len(x_train):
					reject_th.append(th)	


				y_all[th] = y

		else:
				return 'Error'
			
	return y_all, reject_th


def splitTextToNumber(string,number):
	words = string.split(" ")
	grouped_words = [' '.join(words[i: i + number]) for i in range(0, len(words)-(number-1), 1)]
	return grouped_words

#label_embedding=np.array(torch.stack(biobert.word_vector(label)))[0]

def state_of_the_art_predictor(biobert,x_test,th,cmax,label_embedding):
  
	y_pred=[0 for i in range(len(x_test))]
	#biobert=BiobertEmbedding()
	density = {}
	for i in range(0,len(x_test)):
		print('test instance: ' , i)
		density[i] = {}
		abstract=x_test[i]
		final_max=0
		for j in range(1,(cmax+1)):
			max_j=0
			#print( len( splitTextToNumber(abstract,j) ) )
			density[i][j] = []
			for sentence in splitTextToNumber(abstract,j):

				distj=1-distance.cosine(label_embedding,np.array(biobert.sentence_vector(sentence.strip())))
				density[i][j].append([distj])

				if (distj >=max_j):
					max_j=distj

			if(max_j>=final_max):
				final_max=max_j

		if(final_max>=th):
			y_pred[i]=1
		elif(final_max<th):
			y_pred[i]=0
		print(final_max)
	return y_pred , density

def save_results(mode, label, scenario, y_test_edited, predictions, time_execution, threshold = [], rest_information = []):

	accuracy, prec, rec, f1_macro, f1_weighted, exec_time = [], [], [], [], [], []

	if mode == 5:

		f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
		f1_weighted.append(f1_score(y_test_edited, predictions, average = 'weighted'))
		accuracy.append(acc(y_test_edited, predictions))
		prec.append(precision_score(y_test_edited, predictions))
		rec.append(recall_score(y_test_edited, predictions))
		exec_time.append(time_execution)

		df = pd.DataFrame(list(zip(f1_macro, f1_weighted, accuracy, prec, rec, exec_time)), columns =['f1_macro', 'f1_weighted', 'accuracy', 'prec', 'rec', 'execution_time(sec)']) 
		df.to_csv('SETN2020_LWS_results_' + label + '_' + scenario +  '.csv')

		return

	elif mode == 1:

		f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
		f1_weighted.append(f1_score(y_test_edited, predictions, average = 'weighted'))
		accuracy.append(acc(y_test_edited, predictions))
		prec.append(precision_score(y_test_edited, predictions))
		rec.append(recall_score(y_test_edited, predictions))
		exec_time.append(time_execution)
		
		df = pd.DataFrame(list(zip(f1_macro, f1_weighted, accuracy, prec, rec, exec_time)), columns =['f1_macro', 'f1_weighted', 'accuracy', 'prec', 'rec', 'execution_time(sec)']) 
		df.to_csv('SETN2020_DCbioSentenceMax_results_' + label + '_' + scenario + '.csv')

		return

	elif mode == 2:

		learner = rest_information[2]

		f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
		f1_weighted.append(f1_score(y_test_edited, predictions, average = 'weighted'))
		accuracy.append(acc(y_test_edited, predictions))
		prec.append(precision_score(y_test_edited, predictions))
		rec.append(recall_score(y_test_edited, predictions))
		exec_time.append(time_execution)
		tp, fp, fn, tn = confusion_matrix(y_test_edited, predictions).flatten()


		df = pd.DataFrame(list(zip(f1_macro, f1_weighted, accuracy, prec, rec, exec_time, [tp], [fp], [fn], [tn], [rest_information[0]], [rest_information[1]] ) ), columns =['f1_macro', 'f1_weighted', 'accuracy', 'prec', 'rec', 'execution_time(sec)', 'tp', 'fp', 'fn', 'tn', 'shape train data', 'shape test data']) 
		df.to_csv('SETN2020_WSL-baseline_results_' + label + '_' + scenario + '_' + learner + '.csv')

		return
				
	

	elif mode == 3 or mode == 4:
		
		space = rest_information[0]
		reject_th = rest_information[1]
		lea = rest_information[6]
		learner = rest_information[7]
		time_preprocess1 = rest_information[8]
		time_preprocess2 = rest_information[9]

		tps, fps, fns, tns = [], [], [], []

		for _ in space:
				
			if _ in reject_th:

				f1_macro.append(-1)
				f1_weighted.append(-1)
				accuracy.append(-1)
				prec.append(-1)
				rec.append(-1)
				exec_time.append(-1)
				continue


			start = time.time()


			if mode == 3:

				x_train_bert = rest_information[2]
				y_train_bert = rest_information[3]
				x_test_bert = rest_information [4]
				y_test_edited = rest_information[5]


				approach = 'WDCbio(bioBERT)'

				y_train_edited = change_labels(y_train_bert[_], label)
				predictions =  train_classifier(lea, x_train_bert, y_train_edited, x_test_bert, y_test_edited)
			
			
			elif mode == 4:

				x_train_new = rest_information[2]
				x_test_new = rest_information[3]
				y_train_mode4 = rest_information [4]
				y_test_edited = rest_information[5]

				approach = 'WDCbio(tfidf)'

				x_train_tf, x_test_tf = tfidf(x_train_new, y_train_mode4[_], x_test_new)
				y_train_edited =  change_labels(y_train_mode4[_], label)
				predictions =  train_classifier(lea, x_train_tf, y_train_edited, x_test_tf, y_test_edited)


			end = time.time()
			exec_time.append((end - start) + time_preprocess1 + time_preprocess2)
			print('time: ', exec_time[-1])

			f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
			f1_weighted.append(f1_score(y_test_edited, predictions, average = 'weighted'))
			accuracy.append(acc(y_test_edited, predictions))
			prec.append(precision_score(y_test_edited, predictions))
			rec.append(recall_score(y_test_edited, predictions))
			tp, fp, fn, tn = confusion_matrix(y_test_edited, predictions).flatten()
			tps.append(tp)
			fps.append(fp)
			fns.append(fn)
			tns.append(tn)


		df = pd.DataFrame(list(zip(f1_macro, f1_weighted, accuracy, prec, rec, exec_time, tps, fps, fns, tns)), columns =['f1_macro', 'f1_weighted', 'accuracy', 'prec', 'rec', 'execution_time(sec)', 'tp', 'fp', 'fn', 'tn']) 
		df.set_index(np.arange(0.73, 0.83, 0.01), inplace=True)
		df.to_csv('SETN2020_' + approach + '_results_' + label + '_' + scenario + '_' + learner + '.csv')

		return


	else:

		print('No available mode!!')
		return


def load_embeddings(label, selected_scenario, path):


		os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\bioBERT embeddings profile per sentence')

		with open('bioBERT_profile_per_sentence_' + label + '_' + selected_scenario + '.pickle', 'rb') as f:
				x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2 = pickle.load(f)
		f.close()
		os.chdir(path)

		return x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2

def load_embeddings_values(label, selected_scenario, path):


		os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\bioBERT embeddings')

		with open("bioBERTemb_" + selected_scenario + '_' + label + '.pickle' , 'rb') as f:
					x_train_bert, x_test_bert, time_preprocess1, x_train_new, x_test_new, y_train_new, y_test_new = pickle.load(f)
		f.close()
		os.chdir(path)

		return x_train_bert, x_test_bert, time_preprocess1, x_train_new, x_test_new, y_train_new, y_test_new 

#%%
##################################################################################################

def main(mesh, alg, scenario, path):

	#biobert=BiobertEmbedding()
	mesh_input = ["Biomineralization", "Chlorophyceae" , "Cytoglobin"]

	if mesh == 4:
		labels = mesh_input 
	else:
		labels = [mesh_input[mesh]]

	scenario_input = ['train_ratio_1_1_test_ratio',  'train_ratio_1_3_test_ratio']

	if scenario == 3:
		sc = scenario_input 
	else:
		sc = [scenario_input[scenario - 1]]

	print(scenario, sc)
	mode = alg

	for label in labels:
		
		results = []
		
		# read datasets
		
		for selected_scenario in sc:

			os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\raw data')
			print('Label: ', label, 'Scenario: ', selected_scenario)

			if selected_scenario == 'train_ratio_1_1_test_ratio':
				#pass
				x_train,y_train = prepare_X_Y("mesh_2018_" + label.lower() + ".txt")
				x_test,y_test   = prepare_X_Y("mesh_2019_" + label.lower()  + "_mixed.txt")

			else :
				#pass
				x_train,y_train = prepare_X_Y("mesh_2018_" + label.lower() + "1_to_3.txt")
				x_test,y_test   = prepare_X_Y("mesh_2019_" + label.lower()  + "_mixed.txt")
				#x_train,y_train = prepare_X_Y("new_" + label.lower() + "_2018_train.txt")

			# return to the main directory
			os.chdir(path)

			#x_test = x_test[0:2]
			#y_test = y_test[0:2]
			#x_train = x_train[0:20]
			#y_train = y_train[0:20]

			print("Train: %d, Test: %d" %(len(y_train), len(y_test)))

			#threshold = np.arange(0.65, 0.91, 0.01) #[0.77]#[0.7714534135333784] #np.arange(0.65, 0.91, 0.01)

			if mode == 6:
				
				biobert = BiobertEmbedding()

				print('\n****reduce time mode****\n')

				start_emb = time.time()

				label_emb= np.array(torch.stack(biobert.word_vector(label)))[0]
				df_train, x_train_new, total_calls_train, total_sentences_train, x_train_old, reject_train = my_predictor_save(biobert, x_train, label_emb)
				df_test,  x_test_new , total_calls_test , total_sentences_test , x_test_old , reject_test  = my_predictor_save(biobert, x_test,  label_emb)

				end_emb = time.time()

				os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\bioBERT embeddings profile per sentence')
				with open('bioBERT_profile_per_sentence_' + label + '_' + selected_scenario + '.pickle', 'wb') as f:
						pickle.dump([df_train, df_test, y_train, y_test, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sentences_train, total_calls_test, total_sentences_test] , (np.round(end_emb - start_emb,3))], f)
				f.close()
				os.chdir(path)
				continue


			elif mode == 7:
				# use Embeddings
				biobert = BiobertEmbedding()

				os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\bioBERT embeddings profile per sentence')

				with open('bioBERT_profile_per_sentence_' + label + '_' + selected_scenario + '.pickle', 'rb') as f:
					x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2 = pickle.load(f)
				f.close()


				start_emb = time.time()
				x_train_bert = get_embeddings(biobert, x_train_new)
				x_test_bert  = get_embeddings(biobert, x_test)
				end_emb = time.time()

				os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\bioBERT embeddings')
				with open("bioBERTemb_" + selected_scenario + '_' + label + '.pickle' , 'wb') as f:
					pickle.dump([x_train_bert, x_test_bert, (end_emb - start_emb), x_train_new, x_test_new, y_train_new, y_test_new], f,  protocol=pickle.HIGHEST_PROTOCOL)
				f.close()
				os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\bioBERT embeddings profile per sentence')

				print('Pickle was saved')
				os.chdir(path)

				continue


			elif mode == 1:
			
				# on the fly evaluation by cosine similarity

				biobert = BiobertEmbedding()
				th = 0.77
				start = time.time()

				x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2 = load_embeddings(label, selected_scenario, os.getcwd())

				y_test_edited = change_labels(y_test_new, label)

				label_emb= np.array(torch.stack(biobert.word_vector(label)))[0]
				df_test, x_test_new_export, total_calls_train, total_sent_train, x_test_old, reject_test =  my_predictor_save(biobert, x_test_new, label_emb)
				print(x_test_new_export[0][0:10], '\n', x_test_old[0][0:10], '\n', x_test_new[0][0:10])

				predictions = [0 for i in range(len(x_test_new_export))]
				c = 0
				for _ in df_test.keys():
					if max(df_test[_]) > th:
						predictions[c] = 1
					c+=1

				end = time.time()
				time_execution = (np.round(end - start,3))

				os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\DCbio-SentenceMax')
				save_results(mode, label, selected_scenario, y_test_edited, predictions, time_execution)

				print('Saving pickles for tuning stage... ')
				with open('SETN2020_DCbioSentenceMax_results_' + label + '_' + selected_scenario + '.pickle', 'wb') as f:
					pickle.dump([df_test, y_test_edited], f)
				f.close()

				os.chdir(path)

			
			elif mode == 2:
			
				# abstract occurence
				start = time.time()

				x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2 = load_embeddings(label, selected_scenario, os.getcwd())

				y_train_occ = occurence(x_train_new, y_train_new, label)
				x_train_tf, x_test_tf = tfidf(x_train_new, y_train_new, x_test_new)
				print(x_train_tf.shape)
				y_train_edited = change_labels(y_train_occ, label)
				y_test_edited =  change_labels(y_test_new,  label)
				
				lea = SVC(kernel='linear')
				#lea = LogisticRegression()

				learner = 'SVC'

				predictions =  train_classifier(lea, x_train_tf, y_train_edited, x_test_tf, y_test_edited)


				end = time.time()
				time_execution = (np.round(end - start,3))

				rest_information = [x_train_tf.shape, x_test_tf.shape, learner]

				os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\WSL-baseline')
				save_results(mode, label, selected_scenario, y_test_edited, predictions, time_execution, threshold = [],  rest_information = rest_information)



			
			elif mode == 3 or mode == 4:
				
				biobert = BiobertEmbedding()

				# use Embeddings
				
				th = [0.77]
				space = np.arange(0.73, 0.83, 0.01)

				lea = SVC(kernel = 'linear')
				learner = 'SVC_range'

				# here x_train_bert and x_test_bert are loaded, containing the embedding of the total instance (e.g. 1998 vectors of (768,) )
				x_train_bert, x_test_bert, time_preprocess1, x_train_new1, x_test_new1, y_train_new1, y_test_new1  = load_embeddings_values(label, selected_scenario, path)

				# here x_train_bert and x_test_bert are loaded, containing the embedding profile of the total instance into lists, as well as the computed y values fot th = 0.77
				x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2 = load_embeddings(label, selected_scenario, os.getcwd())

				
				#start = time.time()
				print(len(x_train_bert_profile) , len(x_test_bert_profile))
				
				x_train_bert1 = my_predictor_per_sentence_max_loaded_embs(x_train_bert_profile, threshold = space)
				x_test_bert1 = my_predictor_per_sentence_max_loaded_embs(x_test_bert_profile, threshold = space)

				label_emb = np.array(torch.stack(biobert.word_vector(label)))[0]

				y_test_edited =  change_labels(y_test_new, label)

				if mode == 3:
					# cosine similarity + classifier (embeddings transformed) 
					y_train_bert, reject_th = augment_y_with_embeddings_max(biobert, label, label_emb, x_train_bert1, y_train_new, 2, threshold = space)
					new_path = r'C:\Users\stam\Documents\git\Amulet-Setn\WDCbio(bioBERT)'
					rest_information = [space, reject_th, x_train_bert, y_train_bert, x_test_bert, y_test_edited, lea, learner, time_preprocess1, time_preprocess2]

				else:
					# cosine similarity + classifier (tfidf transformed) 
					y_train_mode4, reject_th = augment_y_with_embeddings_max(biobert, label, label_emb, x_train_bert1, y_train_new, 2, threshold = space)
					new_path = r'C:\Users\stam\Documents\git\Amulet-Setn\WDCbio(tfidf)'
					rest_information = [space, reject_th, x_train_new, x_test_new, y_train_mode4, y_test_edited, lea, learner, 0, time_preprocess2]



				print("rej ", reject_th)
		
				print('test: ', len(y_test))

				os.chdir(new_path)
				save_results(mode, label, selected_scenario, y_test_edited, [], 0, rest_information = rest_information)


			elif mode == 5:

				os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\bioBERT embeddings profile per sentence')
				with open('bioBERT_profile_per_sentence_' + label + '_' + selected_scenario + '.pickle', 'rb') as f:
					x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2 = pickle.load(f)
				f.close()
				os.chdir(path)

				biobert = BiobertEmbedding()

				start = time.time()

				label_emb= np.array(torch.stack(biobert.word_vector(label)))[0]
				predictions , density = state_of_the_art_predictor(biobert, x_test_new,0.77,3,label_emb)
				y_test_edited = change_labels(y_test_new, label)

				end = time.time()
				time_execution = np.round(end-start,2)

				os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\LWS')
				print('Saving density ... ')
				with open('density_' + label + '_' + selected_scenario + '.pickle', 'wb') as f:
						pickle.dump([density, predictions, y_test_edited, np.round(end-start,2)], f)
				f.close()

				save_results(mode, label, selected_scenario, y_test_edited, predictions, time_execution)

	return





if __name__ == "__main__":
	
	print('Welcome to AMULET-SETN repo!!')

	random.seed(24)
	
	mesh = int(input('You have to select among our labels: \n1. Biomineralization \n2. Chlorophyceae \n3. Cytoglobin \n4. all of them\n\nYour answer: ... '))
	alg = int(input('Choose which algorithm you want to run: \n1. DCbio(Sentence-max) \n2. Baseline of WSL \n3. WDCbio(bioBERT) \n4. WDCbio(tfidf) \n5. LWS\n6. Save Embeddings on Sentence Level\n7. Save Embeddings\nYour answer: ... '))

	scenario = int(input('Moreover, we need to knwo which sceraio based on ratio between train and test data you need to run: \n1. 1_1 \n2. 1_3\n3. both\n\nYour answer: ...'))

	
	main(mesh, alg, scenario, os.getcwd())
	print('End of scripting')