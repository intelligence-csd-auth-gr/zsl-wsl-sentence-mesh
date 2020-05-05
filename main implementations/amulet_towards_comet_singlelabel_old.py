# Import comet_ml in the top of your file
from comet_ml import Experiment

exp = Experiment(api_key="FtKqgeF7bSvIiTKbMHZjvWaJ4",
												project_name="test_amulet", workspace="terry07")


#import os

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
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score as acc


from biobert_embedding.embedding import BiobertEmbedding

import time

import spacy

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

def change_labels(y_labels):

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
			if(len(sent)>5):
				if (len(sent) == 59):
					continue
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

def my_predictor_save(biobert, x_test, label_emb, th = 0.73):
  
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

def augment_y_with_embeddings_max(biobert, label_embedding, x_train, y_train, command, threshold, qx = None):
 
	#label_embedding= np.array(torch.stack(biobert.word_vector(label)))
	#print('1: ',label_embedding.shape, '2: ', x_train[0].shape)
	y_all= {}
	reject_th = []


	for th in threshold:
		counter = 0

		print('th: ', threshold)

		y_all[th] = []
		y = []
		if command == 1:
			pass

		elif command == 2:

				for i in range(0, len(x_train)):
					
					#command2
					#total_sim=(1-distance.cosine(label_embedding,x_train[i])) 
					#print('total sim: ' , total_sim)

				
					if(x_train[i]>th):
							y.append( y_train[i]+"#"+label )
							counter += 1
					else:
							y.append( y_train[i] )
			 
				
				y_all[th] = y

		else:
				return 'Error'
			
	return y_all, reject_th


def splitTextToNumber(string,number):
	words = string.split(" ")
	grouped_words = [' '.join(words[i: i + number]) for i in range(0, len(words)-(number-1), 1)]
	return grouped_words

#label_embedding=np.array(torch.stack(biobert.word_vector(label)))[0]

def state_of_the_art_predictor(x_test,th,cmax,label_embedding):
  
	y_pred=[0 for i in range(len(x_test))]
	#biobert=BiobertEmbedding()
	
	for i in range(0,len(x_test)):
		print('test instance: ' , i)
		abstract=x_test[i]
		final_max=0
		for j in range(1,(cmax+1)):
			max_j=0
			#print( len( splitTextToNumber(abstract,j) ) )
			for sentence in splitTextToNumber(abstract,j):

				distj=1-distance.cosine(label_embedding,np.array(biobert.sentence_vector(sentence.strip())))
				
				if (distj >=max_j):
					max_j=distj

			if(max_j>=final_max):
				final_max=max_j

		if(final_max>=th):
			y_pred[i]=1
		elif(final_max<th):
			y_pred[i]=0
		print(final_max)
	return y_pred


random.seed(24)
labels = ["Biomineralization" , "Chlorophyceae" , "Cytoglobin"]
#labels = [ "Chlorophyceae" , "Cytoglobin"]

biobert=BiobertEmbedding()

for label in labels:
	#label = labels[0]
	results = []
	mode = 'furn'


	# main step for initial data per label
	#scenario = ['train_ratio_1_1_test_ratio_pure']
	scenario = ['train_ratio_1_1_test_ratio_mixed']
	#scenario = ['train_ratio_1_4_test_ratio_pure']
	#scenario = ['train_ratio_1_4_test_ratio_mixed']
	#scenario = ['train_ratio_1_1_test_ratio_pure', 'train_ratio_1_1_test_ratio_mixed', 'train_ratio_1_4_test_ratio_pure', 'train_ratio_1_4_test_ratio_mixed']
	#scenario = ['train_ratio_1_4_test_ratio_mixed',  'train_ratio_1_1_test_ratio_mixed']


	for sc in scenario:
		print(label, sc)
		if sc == 'train_ratio_1_1_test_ratio_pure':

			x_train,y_train = prepare_X_Y("mesh_2018_" + label.lower() + ".txt")
			x_test,y_test   = prepare_X_Y("mesh_2019_" + label.lower() + ".txt")

		elif sc == 'train_ratio_1_1_test_ratio_mixed':

			x_train,y_train = prepare_X_Y("mesh_2018_" + label.lower() + ".txt")
			x_test,y_test   = prepare_X_Y("mesh_2019_" + label.lower()  + "_mixed.txt")


		elif sc == 'train_ratio_1_4_test_ratio_pure':

			x_train,y_train = prepare_X_Y("mesh_2018_" + label.lower() + "1_to_4.txt")
			x_test,y_test   = prepare_X_Y("mesh_2019_" + label.lower()  + ".txt")

		elif sc == 'train_ratio_1_4_test_ratio_mixed':

			x_train,y_train = prepare_X_Y("mesh_2018_" + label.lower() + "1_to_4.txt")
			x_test,y_test   = prepare_X_Y("mesh_2019_" + label.lower()  + "_mixed.txt")
			#mesh_2019_cytoglobin_mixed

		#x_test = x_test[0:2]
		#y_test = y_test[0:2]
		#x_train = x_train[0:20]
		#y_train = y_train[0:20]

		print("Train: %d, Test: %d" %(len(y_train), len(y_test)))

		threshold = [0.77]#np.arange(0.65, 0.86, 0.01)

		accuracy, prec, rec, f1_macro, f1_weighted = [], [], [], [], []
		exec_time = []

		if mode == 'mode1':
			# on the fly evaluation by cosine similarity
			y_test_edited1 = change_labels(y_test)
		elif mode == 'mode2':
			# abstract occurence
			y_train_occ = occurence(x_train, y_train)
			x_train_tf, x_test_tf = tfidf(x_train, y_train, x_test)
			print(x_train_tf.shape)
			y_train_edited = change_labels(y_train_occ)
			y_test_edited2 =  change_labels(y_test)
			lea = SVC(kernel='linear')
			#lea = RF(random_state = 23)
		elif mode == 'mode3' or mode == 'mode4':
			# use Embeddings
			x_train_bert = get_embeddings(biobert, x_train)
			x_test_bert  = get_embeddings(biobert, x_test)

			#label_emb= np.array(torch.stack(biobert.word_vector(label)))[0]
			#x_train_bert1 = my_predictor_per_sentence_max(biobert, x_train, label_emb, 0.77)
			#x_test_bert1 = my_predictor_per_sentence_max(biobert, x_test, label_emb, 0.77)

			#print(len(x_train_bert), len(x_test_bert), len(x_train_bert1), len(x_test_bert1), x_train_bert[0].shape, x_train_bert1)
			#print('here1')
			#break

		#elif mode == 'mode4':
			# read Emdeddings
			#prepared_test_embeddings  = pd.read_csv('x_test_embedding_file.csv')
			#prepared_train_embeddings = pd.read_csv('x_train_embedding_file.csv')
		elif mode == 'furn':

			start1 = time.time()
			label_emb= np.array(torch.stack(biobert.word_vector(label)))[0]
			predictions=state_of_the_art_predictor(x_test,0.77,3,label_emb)
			y_test_edited = change_labels(y_test)


			f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
			f1_weighted.append(f1_score(y_test_edited, predictions, average = 'weighted'))
			accuracy.append(acc(y_test_edited, predictions))
			prec.append(precision_score(y_test_edited, predictions))
			rec.append(recall_score(y_test_edited, predictions))
			end1 = time.time()
			exec_time.append((end1 - start1))
			
			#continue
			threshold = []

		for th in threshold:
			
			start = time.time()
			
			if mode == 'mode1':
				
				#predictions = my_predictor(biobert, x_test, th)
				#label_emb = np.array(biobert.sentence_vector(label))
				label_emb= np.array(torch.stack(biobert.word_vector(label)))[0]
				#predictions = my_predictor_per_sentence(biobert,x_test, label_emb, th)
				predictions, ddf1 = my_predictor_save(biobert, x_test, label_emb, th)
				y_test_edited = y_test_edited1


			elif mode == 'mode2':
			 
				predictions =  train_classifier(lea, x_train_tf, y_train_edited, x_test_tf, y_test_edited2)
				y_test_edited = y_test_edited2

			else:

					if mode == 'mode3':
								# cosine similarity + classifier (embeddings transformed) 
								y_train_bert, reject_th = augment_y_with_embeddings(biobert, label, x_train_bert, y_train, 2, threshold = np.arange(0.65, 0.86, 0.01))
								#y_train_bert, reject_th = augment_y_with_embeddings_max(biobert, label_emb, x_train_bert1, y_train, 2, threshold = [0.77])
								#print(y_train_bert)
								#break
					elif mode == 'mode4':
								# cosine similarity + classifier (tfidf transformed) 
								y_train_mode4, reject_th = augment_y_with_embeddings(biobert, label, x_train_bert, y_train, 2, threshold = np.arange(0.65, 0.86, 0.01))
								#y_train_mode4, reject_th = augment_y_with_embeddings_max(biobert, label_emb, x_train_bert1, y_train, 2, threshold = [0.77])

								#x_train_tf, x_test_tf = tfidf(x_train, y_train_mode4, x_test)
								#y_train_edited =  change_labels(y_train_mode4)
					else:
						continue


					print("rej ",reject_th)
					#print(y_train_bert.keys(), y_train_bert[0.65], len(y_train_bert[0.65]))
					
					y_test_edited3 =  change_labels(y_test)
					lea = SVC(kernel='linear')

					for _ in np.arange(0.65, 0.86, 0.01):
									
									if _ in reject_th:
										#print(set(change_labels(y_train_bert[_])))
					
										f1_macro.append(-1)
										f1_weighted.append(-1)
										accuracy.append(-1)
										prec.append(-1)
										rec.append(-1)
										exec_time.append(-1)
					
										continue
									#start3 = time.time()

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
												x_train_tf, x_test_tf = tfidf(x_train, y_train_mode4[_], x_test)
												y_train_edited =  change_labels(y_train_mode4[_])
												predictions =  train_classifier(lea, x_train_tf, y_train_edited, x_test_tf, y_test_edited3)
												y_test_edited = y_test_edited3

									f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
									f1_weighted.append(f1_score(y_test_edited, predictions, average = 'weighted'))
									accuracy.append(acc(y_test_edited, predictions))
									prec.append(precision_score(y_test_edited, predictions))
									rec.append(recall_score(y_test_edited, predictions))

									end3 = time.time()
									exec_time.append((end3 - start))
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
			exec_time.append((end - start))
			print('time: ', exec_time[-1])
			#exp.log_metric('f1_macro', f1_macro[-1])
			#exp.log_metric('f1_weighted', f1_weighted[-1])
			#exp.log_metric('accuracy', accuracy[-1])
			#exp.log_metric('precision' , prec[-1])
			#exp.log_metric('recall' , rec[-1])
			#exp.log_metric('time (sec)' , exec_time[-1]) 
			if mode == 'mode2':
				break


		print('Saving stage...')
		if mode == 'mode2':
			df = pd.DataFrame(list(zip(f1_macro, f1_weighted, accuracy, prec, rec, exec_time, [x_train_tf.shape], [x_test_tf.shape])), columns =['f1_macro', 'f1_weighted', 'accuracy', 'prec', 'rec', 'execution_time(sec)', 'shape train data', 'shape test data']) 
		else:
			df = pd.DataFrame(list(zip(f1_macro, f1_weighted, accuracy, prec, rec, exec_time)), columns =['f1_macro', 'f1_weighted', 'accuracy', 'prec', 'rec', 'execution_time(sec)']) 

		
		if True:#mode != 'furn':
			df.to_csv('single_results_' + label + '_' + sc + '_' + mode + '.csv')
		
		#params = {'thresold': threshold}
		#exp.log_parameters(params)

		if mode == 'mode1':
			print('Saving pickles ... ')
			with open('results_' + label + '_' + sc + '_' + mode + '.pickle', 'wb') as f:
					pickle.dump([ddf1, y_test_edited], f)
			f.close()