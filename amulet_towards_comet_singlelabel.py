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

def occurence(x_train, y_train):

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
def augment_y_with_embeddings_max(biobert, label_embedding, x_train, y_train, command, threshold, qx = None):
 
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

def save_results(mode, label, scenario, y_test_edited, predictions, time_execution, threshold = []):

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

	for th in threshold:
				
				start1 = time.time()
				
				if mode == 'mode1':
					
					#predictions = my_predictor(biobert, x_test, th)
					#label_emb = np.array(biobert.sentence_vector(label))
					label_emb= np.array(torch.stack(biobert.word_vector(label)))[0]
					#predictions = my_predictor_per_sentence(biobert,x_test, label_emb, th)
					#predictions, ddf1 = my_predictor_save(biobert, x_test, label_emb, th)
					#predictions, ddf1, x_train_new, total_calls_train, total_sent_train = my_predictor_save(biobert, x_test_new, label_emb, th)
					df_test, x_test_new_export, total_calls_train, total_sent_train, x_test_old, reject_test =  my_predictor_save(biobert, x_test_new, label_emb)

					print(x_test_new_export[0][0:10], '\n' , x_test_old[0][0:10], '\n', x_test_new[0][0:10])
					predictions = [0 for i in range(len(x_test_new_export))]
					c = 0
					for _ in df_test.keys():
						if max(df_test[_]) > th:
							predictions[c] = 1
						c+=1

					
					#y_test_edited = y_test_edited1


				elif mode == 'mode2':
				 
					predictions =  train_classifier(lea, x_train_tf, y_train_edited, x_test_tf, y_test_edited)
					#y_test_edited = y_test_edited2

				else:

						if mode == 'mode3':
									
									# cosine similarity + classifier (embeddings transformed) 
									#y_train_bert, reject_th = augment_y_with_embeddings(biobert, label, x_train_bert, y_train, 2, threshold = np.arange(0.65, 0.86, 0.01))
									y_train_bert, reject_th = augment_y_with_embeddings_max(biobert, label_emb, x_train_bert1, y_train_new, 2, threshold = threshold)#np.arange(0.65, 0.91, 0.01))
									#print(y_train_bert)
									#break

						elif mode == 'mode4':
									
									# cosine similarity + classifier (tfidf transformed) 
									#y_train_mode4, reject_th = augment_y_with_embeddings(biobert, label, x_train_bert, y_train, 2, threshold = np.arange(0.65, 0.86, 0.01))
									#print(x_train_bert1)
									y_train_mode4, reject_th = augment_y_with_embeddings_max(biobert, label_emb, x_train_bert1, y_train_new, 2, threshold = threshold)#np.arange(0.65, 0.91, 0.01))
									#print(y_train_mode4.keys())
									#x_train_tf, x_test_tf = tfidf(x_train, y_train_mode4, x_test)
									#y_train_edited =  change_labels(y_train_mode4)
						else:
							continue


						print("rej ",reject_th)
						#print(y_train_bert.keys(), y_train_bert[0.65], len(y_train_bert[0.65]))
						
						y_test_edited3 =  change_labels(y_test_new)
						print('test: ', len(y_test))
						#lea = SVC(kernel='linear' , C = 10 ,  probability = True)
						#lea = SVC(kernel = 'linear')
						lea = LinearSVC()
						#lea = RF()
						#lea = LogisticRegression()
						#lea = Grad()
						#lea = KNN(n_neighbors = 1)
						#lea = ADA()
						learner = 'linearSVC_range' #c_10_prob' #1nn' #RF' #Grad' #'SVC_tuned' # LR' 'linearSVC'

						for _ in threshold:#np.arange(0.70, 0.81, 0.01):
										
										if _ in reject_th:
											#print(set(change_labels(y_train_bert[_])))
						
											f1_macro.append(-1)
											f1_weighted.append(-1)
											accuracy.append(-1)
											prec.append(-1)
											rec.append(-1)
											exec_time.append(-1)
						
											continue
										start3 = time.time()

										if mode == 'mode3':
													
													y_train_edited = change_labels(y_train_bert[_])
													#print(y_train_bert[_] , y_train_edited)
													print(len(x_train_bert), len(y_train_edited), len(x_test_bert), len(y_test_edited3))
													#print(x_train_bert[0].shape, y_train_edited)
													predictions =  train_classifier(lea, x_train_bert, y_train_edited, x_test_bert, y_test_edited3)
													y_test_edited = y_test_edited3
													#print(y_train_edited, y_test_edited)
													#print('here')
													
										elif mode == 'mode4':
													
													#print(_, len(x_train), len(y_train_mode4[_]) , len(x_test))
													x_train_tf, x_test_tf = tfidf(x_train_new, y_train_mode4[_], x_test_new)
													y_train_edited =  change_labels(y_train_mode4[_])
													predictions =  train_classifier(lea, x_train_tf, y_train_edited, x_test_tf, y_test_edited3)
													y_test_edited = y_test_edited3

										f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
										f1_weighted.append(f1_score(y_test_edited, predictions, average = 'weighted'))
										accuracy.append(acc(y_test_edited, predictions))
										prec.append(precision_score(y_test_edited, predictions))
										rec.append(recall_score(y_test_edited, predictions))

										end = time.time()
										exec_time.append((end - start3) + time_preprocess1 + time_preprocess2)
										print('time: ', exec_time[-1])

				if mode == 'mode3' or mode == 'mode4':
					break
				# evaluation

				f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
				f1_weighted.append(f1_score(y_test_edited, predictions, average = 'weighted'))
				accuracy.append(acc(y_test_edited, predictions))
				prec.append(precision_score(y_test_edited, predictions))
				rec.append(recall_score(y_test_edited, predictions))

				end = time.time()
				exec_time.append((end - start1))
				print('time: ', exec_time[-1])
				#exp.log_metric('f1_macro', f1_macro[-1])
				#exp.log_metric('f1_weighted', f1_weighted[-1])
				#exp.log_metric('accuracy', accuracy[-1])
				#exp.log_metric('precision' , prec[-1])
				#exp.log_metric('recall' , rec[-1])
				#exp.log_metric('time (sec)' , exec_time[-1]) 
				if mode == 'mode2':
					break

	if mode != 'save_embs' :

		print('Saving stage...')
		if mode == 'mode2':
			df = pd.DataFrame(list(zip(f1_macro, f1_weighted, accuracy, prec, rec, exec_time, [x_train_tf.shape], [x_test_tf.shape])), columns =['f1_macro', 'f1_weighted', 'accuracy', 'prec', 'rec', 'execution_time(sec)', 'shape train data', 'shape test data']) 
		else:
			df = pd.DataFrame(list(zip(f1_macro, f1_weighted, accuracy, prec, rec, exec_time)), columns =['f1_macro', 'f1_weighted', 'accuracy', 'prec', 'rec', 'execution_time(sec)']) 

	
	if mode != 'mode1' and mode != 'furn':
	
		df.to_csv('single_results_' + label + '_' + sc + '_' + mode + '_' + learner + '.csv')

	else:

		df.to_csv('single_results_' + label + '_' + sc + '_' + mode +  '.csv')


	if mode == 'mode1':

		#df.to_csv('single_results_' + label + '_' + sc + '_' + mode +  '.csv')

		print('Saving pickles ... ')
		with open('results_' + label + '_' + sc + '_' + mode + '.pickle', 'wb') as f:
				pickle.dump([df_test, y_test_edited], f)
		f.close()

	return 
#%%
##################################################################################################

def main(mesh, alg, scenario, path):

	#biobert=BiobertEmbedding()
	mesh_input = ["Biomineralization", "Chlorophyceae" , "Cytoglobin"]

	if mesh == 4:
		labels = mesh_input 
	else:
		labels = [mesh_input[mesh]]

	scenario_input = ['train_ratio_1_1_test_ratio',  'train_ratio_1_4_test_ratio']

	if scenario == 3:
		sc = scenario_input 
	else:
		sc = [scenario_input[scenario - 1]]

	print(scenario, sc)
	#alg = 'furn'
	mode = alg

	for label in labels:
		
		results = []
		

		# read datasets
		
		#print(os.listdir(os.getcwd()))
		for selected_scenario in sc:

			os.chdir(r'C:\Users\stam\Documents\git\Amulet-Setn\raw data')
			print('Label: ', label, 'Scenario: ', selected_scenario)

			if selected_scenario == 'train_ratio_1_1_test_ratio':
				#pass
				x_train,y_train = prepare_X_Y("mesh_2018_" + label.lower() + ".txt")
				x_test,y_test   = prepare_X_Y("mesh_2019_" + label.lower()  + "_mixed.txt")

			else :
				#pass
				x_train,y_train = prepare_X_Y("mesh_2018_" + label.lower() + "1_to_4.txt")
				x_test,y_test   = prepare_X_Y("mesh_2019_" + label.lower()  + "_mixed.txt")

			# return to the main directory
			os.chdir(path)



			#x_test = x_test[0:2]
			#y_test = y_test[0:2]
			#x_train = x_train[0:20]
			#y_train = y_train[0:20]

			print("Train: %d, Test: %d" %(len(y_train), len(y_test)))

			threshold = np.arange(0.65, 0.91, 0.01) #[0.77]#[0.7714534135333784] #np.arange(0.65, 0.91, 0.01)

			#accuracy, prec, rec, f1_macro, f1_weighted, exec_time = [], [], [], [], [], []
			



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


			if mode == 1:
			
				# on the fly evaluation by cosine similarity

				with open('bioBERT_profile_per_sentence_' + label + '_' + selected_scenario + '.pickle', 'rb') as f:
					x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2 = pickle.load(f)
				f.close()

				y_test_edited = change_labels(y_test_new)
			
			elif mode == 'mode2':
			
				# abstract occurence
				start1 = time.time()


				with open('BERT_per_sentence_' + label +  '_' + sc + '_save_embsnew.pickle', 'rb') as f:
					#x_train_bert1, x_test_bert1, predictions_train, predictions_test, time_preprocess2 = pickle.load(f)
					#x_train_bert_profile, x_test_bert_profile, x_train_bert1_077, x_test_bert1_077, time_preprocess2 = pickle.load(f)
					#x_train_bert_profile, x_test_bert_profile, x_train_bert1_077, x_test_bert1_077, x_train_new, x_test_new, [total_calls_train, total_sent_train, total_calls_test, total_sent_test], time_preprocess2 = pickle.load(f)
					x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2 = pickle.load(f)

				f.close()

				y_train_occ = occurence(x_train_new, y_train_new)
				x_train_tf, x_test_tf = tfidf(x_train_new, y_train_new, x_test_new)
				print(x_train_tf.shape)
				y_train_edited = change_labels(y_train_occ)
				y_test_edited =  change_labels(y_test_new)
				
				#lea = SVC(kernel='linear')
				#lea = LinearSVC()
				#lea = RF(random_state = 23)
				lea = LogisticRegression()
				#lea = Grad()
				#lea = ADA()
				learner = 'LR' #Grad' #SVC' # LR' #'linearSVC'
			
			elif mode == 'mode3' or mode == 'mode4':
				
				# use Embeddings
				
				#x_train_bert = get_embeddings(biobert, x_train)
				#x_test_bert  = get_embeddings(biobert, x_test)

				# here x_train_bert and x_test_bert are loaded, containing the embedding of the total instance (e.g. 1998 vectors of (768,) )
				with open('BERTemb_' + sc +  '_' + label + 'new.pickle', 'rb') as f:
					#x_train_bert, x_test_bert, time_preprocess1 = pickle.load(f)
					x_train_bert, x_test_bert, time_preprocess1, x_train_new1, x_test_new1, y_train_new1, y_test_new1 = pickle.load(f)
				f.close()

				# here x_train_bert and x_test_bert are loaded, containing the embedding profile of the total instance into lists, as well as the computed y values fot th = 0.77

				with open('BERT_per_sentence_' + label +  '_' + sc + '_save_embsnew.pickle', 'rb') as f:
					#x_train_bert1, x_test_bert1, predictions_train, predictions_test, time_preprocess2 = pickle.load(f)
					#x_train_bert_profile, x_test_bert_profile, x_train_bert1_077, x_test_bert1_077, time_preprocess2 = pickle.load(f)
					#x_train_bert_profile, x_test_bert_profile, x_train_bert1_077, x_test_bert1_077, x_train_new, x_test_new, [total_calls_train, total_sent_train, total_calls_test, total_sent_test], time_preprocess2 = pickle.load(f)
					x_train_bert_profile, x_test_bert_profile, y_train_new, y_test_new, x_train_new, x_test_new, x_train_old, x_test_old, reject_train, reject_test, [total_calls_train, total_sent_train, total_calls_test, total_sent_test] , time_preprocess2 = pickle.load(f)

				f.close()
				
				#print(type(x_train_bert_profile))
				#print(x_train_bert_profile.keys())
				print(len(x_train_bert_profile) , len(x_test_bert_profile))
				x_train_bert1 = my_predictor_per_sentence_max_loaded_embs(x_train_bert_profile, threshold = threshold)
				x_test_bert1 = my_predictor_per_sentence_max_loaded_embs(x_test_bert_profile, threshold = threshold)

				label_emb= np.array(torch.stack(biobert.word_vector(label)))[0]
				#x_train_bert1 = my_predictor_per_sentence_max(biobert, x_train, label_emb, threshold )
				#x_test_bert1 = my_predictor_per_sentence_max(biobert, x_test, label_emb, threshold  )

				#print(len(x_train_bert), len(x_test_bert), len(x_train_bert1), len(x_test_bert1), x_train_bert[0].shape,)# x_train_bert1)
				#print('here1')
				#break

			#elif mode == 'mode4':
				# read Emdeddings
				#prepared_test_embeddings  = pd.read_csv('x_test_embedding_file.csv')
				#prepared_train_embeddings = pd.read_csv('x_train_embedding_file.csv')
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
	alg = int(input('Choose which algorithm you want to run: \n1. DCbio(Sentence-max) \n2. Baseline of WSL \n3. WDCbio(tfidf) \n4. WDCbio(bioBERT) \n5. LWS\n6. Save Embeddings on Sentence Level\n7. Save Embeddings\nYour answer: ... '))

	scenario = int(input('Moreover, we need to knwo which sceraio based on ratio between train and test data you need to run: \n1. 1_1 \n2. 1_4\n3. both\n\nYour answer: ...'))

	#labels = ["Biomineralization", "Chlorophyceae" , "Cytoglobin"]
	#labels = ["Chlorophyceae"]

	
	main(mesh, alg, scenario, os.getcwd())
	print('End of scripting')