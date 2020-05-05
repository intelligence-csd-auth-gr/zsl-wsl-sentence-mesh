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

import numpy as np
import pandas as pd
from scipy.spatial import distance

from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score,hamming_loss,accuracy_score

import spacy
spacy.load('en_core_web_sm')
nlp = spacy.load("en_core_web_sm") #nlp = spacy.load('en')

from biobert_embedding.embedding import BiobertEmbedding

import time

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

####
def my_predictor_multi(biobert, x_test, label1, label2, th = 0.73):
  
	th1 = th + 0.01
	y_pred=[0 for i in range(len(x_test))]
	label_embedding1= np.array(torch.stack(biobert.word_vector(label1)))[0]
	label_embedding2= np.array(torch.stack(biobert.word_vector(label2)))[0]
	#label_embedding1= np.array(biobert.sentence_vector(label1))#np.array(torch.stack(biobert.word_vector(label1)))[0]
	#label_embedding2= np.array(biobert.sentence_vector(label2))#np.array(torch.stack(biobert.word_vector(label2)))[0]
	#nlp = spacy.load('en')
	df1, df2= {} , {}
	#count=0

	for i in range(0,len(x_test)):
		average_dist=0
		text = x_test[i]
		tokens = nlp(text)
		max1=0
		max2=0
		df1['label1_instance' + str(i+1)] = []
		df2['label2_instance' + str(i+1)] = []
		
		for sent in tokens.sents:
			if(len(sent)>5 and len(sent)!=59):
				dist1=1-distance.cosine(label_embedding1,np.array(biobert.sentence_vector(sent.string.strip())))
				dist2=1-distance.cosine(label_embedding2,np.array(biobert.sentence_vector(sent.string.strip())))
				if(dist1>=max1):
					 max1=dist1
				if(dist2>=max2):
					max2=dist2
			df1['label1_instance' + str(i+1)].append(dist1)
			df2['label2_instance' + str(i+1)].append(dist2)
		
		if(max1>th and max2<th1):
			y_pred[i]=[1,0]
		elif(max1<th and max2>th1):
			y_pred[i]=[0,1]
		elif(max1>th and max2>th1):
			y_pred[i]=[1,1]
		elif(max1<th and max2<th1):
			y_pred[i]=[0,0]

	#count += 1

	return y_pred, df1, df2

def my_predictor_multi_simple(biobert, x_test, label1, label2, th = 0.73):

	y_pred=[0 for i in range(len(x_test))]
	for i in range(0,len(x_test)):
		if(((1-distance.cosine(np.array(biobert.sentence_vector(label1)),
									  np.array(biobert.sentence_vector(x_test[i][:1000]))))>th) and 
		   (1-distance.cosine(np.array(biobert.sentence_vector(label2)),
									  np.array(biobert.sentence_vector(x_test[i][:1000]))))>th):
		  y_pred[i]=[1,1]
		elif(((1-distance.cosine(np.array(biobert.sentence_vector(label1)),
									  np.array(biobert.sentence_vector(x_test[i][:1000]))))>th) and 
		   (1-distance.cosine(np.array(biobert.sentence_vector(label2)),
									  np.array(biobert.sentence_vector(x_test[i][:1000]))))<=th):
		  y_pred[i]=[1,0]
		elif(((1-distance.cosine(np.array(biobert.sentence_vector(label1)),
									  np.array(biobert.sentence_vector(x_test[i][:1000]))))<=th) and 
			 (1-distance.cosine(np.array(biobert.sentence_vector(label2)),
									  np.array(biobert.sentence_vector(x_test[i][:1000]))))>th):
		  y_pred[i]=[0,1]
		else:
		  y_pred[i]=[0,0]

	return y_pred

def change_labels_multi(y_labels, label1, label2):
	new_y=[]
	for y in y_labels:
		if(y.__contains__(label2) and  y.__contains__(label1)):
			new_y.append([1,1])
		elif(y.__contains__(label1) and not y.__contains__(label2)):
			new_y.append([1,0])
		elif(y.__contains__(label2) and not y.__contains__(label1)):
			new_y.append([0,1])
		else:
			new_y.append([0,0])
	return new_y


random.seed(24)
biobert=BiobertEmbedding()

labels = ['neuroglobin_cytoglobin']
label1 = 'Cytoglobin'
label2 = 'Neuroglobin'

for label in labels:
	#label = labels[0]
	results = []
	mode = 'mode1'


	# main step for initial data per label
	scenario = ['train_ratio_1_2_test_ratio_multilabel_mixed']#, 'train_ratio_1_4_test_ratio_multilabel_pure']


	for sc in scenario:
		#print(label, sc)
		if sc == 'train_ratio_1_4_test_ratio_multilabel_pure':

			x_train,y_train=prepare_X_Y("mesh_2018_cytoglobin1_to_4.txt")
			x_test,y_test=prepare_X_Y("mesh_2019_neuroglobin_cytoglobin_mixed.txt")

		elif sc == 'train_ratio_1_2_test_ratio_multilabel_mixed':

			x_train,y_train=prepare_X_Y("mesh_2018_cytoglobin1_to_4.txt")
			x_test,y_test=prepare_X_Y("mesh_2019_neuroglobin_cytoglobin_mixed.txt")


		#x_test = x_test[0:8]
		#y_test = y_test[0:8]
		#x_train = x_train[0:20]
		#y_train = y_train[0:20]

		print("Train: %d, Test: %d" %(len(y_train), len(y_test)))

		threshold = np.arange(0.73, 0.75, 0.01)
		threshold = [0.73]

		accuracy, prec, rec, f1_macro, f1_weighted = [], [], [], [], []
		exec_time = []

		if mode == 'mode1':
			# on the fly evaluation by cosine similarity
			y_test_edited1 = change_labels_multi(y_test, label1, label2)
		elif mode == 'mode2':
			# abstract occurence
			y_train_occ = occurence(x_train, y_train)
			x_train_tf, x_test_tf = tfidf(x_train, y_train, x_test)
			print(x_train_tf.shape)
			y_train_edited = change_labels(y_train_occ)
			y_test_edited2 =  change_labels(y_test)
			lea = SVC(kernel='linear', C=100)
		elif mode == 'mode3' or mode == 'mode4':
			# use Embeddings
			x_train_bert = get_embeddings(biobert, x_train)
			x_test_bert  = get_embeddings(biobert, x_test)

		#elif mode == 'mode4':
			# read Emdeddings
			#prepared_test_embeddings  = pd.read_csv('x_test_embedding_file.csv')
			#prepared_train_embeddings = pd.read_csv('x_train_embedding_file.csv')
	 



		for th in threshold:
			
			start = time.time()
			
			if mode == 'mode1':
				
				predictions, ddf1, ddf2 = my_predictor_multi(biobert,x_test, label1, label2, th)
				y_test_edited = y_test_edited1

			elif mode == 'mode2':
			 
				predictions =  train_classifier(lea, x_train_tf, y_train_edited, x_test_tf, y_test_edited2)
				y_test_edited = y_test_edited2

			else:

					if mode == 'mode3':
								# cosine similarity + classifier (embeddings transformed) 
								y_train_bert, reject_th = augment_y_with_embeddings(biobert, label, x_train_bert, y_train, 2, threshold = np.arange(0.73, 0.84, 0.01))
					elif mode == 'mode4':
								# cosine similarity + classifier (tfidf transformed) 
								y_train_mode4, reject_th = augment_y_with_embeddings(biobert, label, x_train_bert, y_train, 2, threshold = np.arange(0.73, 0.84, 0.01))
								#x_train_tf, x_test_tf = tfidf(x_train, y_train_mode4, x_test)
								#y_train_edited =  change_labels(y_train_mode4)


					print("rej ",reject_th)
					#print(y_train_bert.keys(), y_train_bert[0.65], len(y_train_bert[0.65]))
					
					y_test_edited3 =  change_labels(y_test)
					lea = SVC(kernel='linear', C=100)

					for _ in np.arange(0.73, 0.75, 0.01):
									
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
												predictions =  train_classifier(lea, x_train_bert, y_train_edited, x_test_bert, y_test_edited3)
												y_test_edited = y_test_edited3
									elif mode == 'mode4':
												x_train_tf, x_test_tf = tfidf(x_train, y_train_mode4[_], x_test)
												y_train_edited =  change_labels(y_train_mode4[_])
												predictions =  train_classifier(lea, x_train_tf, y_train_edited, x_test_tf, y_test_edited3)
												y_test_edited = y_test_edited3

									f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
									f1_weighted.append(f1_score(y_test_edited, predictions, average = 'weighted'))
									accuracy.append(acc(y_test_edited, predictions))
									prec.append(precision_score(y_test_edited, predictions, average = 'macro'))
									rec.append(recall_score(y_test_edited, predictions, average = 'macro'))

									end3 = time.time()
									exec_time.append((end3 - start3))
									print('time: ', exec_time[-1])

			if mode == 'mode3' or mode == 'mode4':
				break
			# evaluation
			print(y_test_edited , predictions)
			y_test_edited, predictions = np.array(y_test_edited), np.array(predictions)
			f1_macro.append(f1_score(y_test_edited, predictions, average = 'macro'))
			f1_weighted.append(f1_score(y_test_edited, predictions, average = 'weighted'))
			accuracy.append(acc(y_test_edited, predictions))
			prec.append(precision_score(y_test_edited, predictions, average = 'macro'))
			rec.append(recall_score(y_test_edited, predictions, average = 'macro'))

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


		if mode == 'mode2':
			df = pd.DataFrame(list(zip(f1_macro, f1_weighted, accuracy, prec, rec, exec_time, [x_train_tf.shape], [x_test_tf.shape])), columns =['f1_macro', 'f1_weighted', 'accuracy', 'prec', 'rec', 'execution_time(sec)', 'shape train data', 'shape test data']) 
		else:
			df = pd.DataFrame(list(zip(f1_macro, f1_weighted, accuracy, prec, rec, exec_time)), columns =['f1_macro', 'f1_weighted', 'accuracy', 'prec', 'rec', 'execution_time(sec)']) 
			#df = pd.DataFrame(list(zip(f1_macro, exec_time)), columns =['f1_macro', 'execution_time(sec)']) 

		df.to_csv('results_multilabel' + label + '_' + sc + '_' + mode + '.csv')
		
		#params = {'thresold': threshold}
		#exp.log_parameters(params)


		import pickle
		print('Saving pickles ... ')
		with open('mode1_multilabel_biobert.pickle', 'wb') as f:
		        pickle.dump([ddf1, ddf2, y_test_edited], f)
		f.close()

		#files.download('results_multilabel' + label + '_' + sc + '_' + mode + '.pickle')