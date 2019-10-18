import nltk
import pandas as pd
import tensorflow as tf
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import re
from tensorflow.keras import regularizers
from tensorflow import keras
import h5py
import Tkinter as tk
from Tkinter import *
from PIL import ImageTk, Image
import os
import tkFont

def showEmoji(msg):
	if msg == "sadness":
		panel.configure(image = SadEmotionImage, bg="black")
	elif msg == "happiness":
		panel.configure(image = HappyEmotionImage, bg="black")
	elif msg == "worry":
		panel.configure(image = WorryEmotionImage, bg="black")
	elif msg == "anger":
		panel.configure(image = AngryEmotionImage, bg="black")
	else:
		panel.configure(image = NeutralEmotionImage, bg="black")
	panel.place(x=330, y=150)

def finalFunction():
	data = pd.read_csv('train_data.csv',encoding='ISO-8859-1')
	data = data.drop(data[data.sentiment == 'boredom'].index)
	data = data.drop(data[data.sentiment == 'enthusiasm'].index)
	data = data.drop(data[data.sentiment == 'empty'].index)
	data = data.drop(data[data.sentiment == 'fun'].index)
	data = data.drop(data[data.sentiment == 'relief'].index)
	data = data.drop(data[data.sentiment == 'surprise'].index)
	data = data.drop(data[data.sentiment == 'love'].index)
	data = data.drop(data[data.sentiment == 'hate'].index)

	data_frame_train=data

	type(data_frame_train)

	data_frame_train

	data_train = data_frame_train[['sentiment','content']]

	data_train

	data_train['content'] = data_train['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

	data_train['content'] = data_train['content'].str.replace('[^\w\s]',' ')

	stop = stopwords.words('english')
	data_train['content'] = data_train['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

	max_features = 10000
	MAX_SEQUENCE_LENGTH = 15
	tokenizer = Tokenizer(num_words=max_features, split=' ')
	tokenizer.fit_on_texts(data_train['content'].values)
	word_index = tokenizer.word_index
	X = tokenizer.texts_to_sequences(data_train['content'].values)
	X = pad_sequences(X,maxlen=MAX_SEQUENCE_LENGTH)

	X.shape

	Y = data_train['sentiment'].to_numpy()

	np.unique(Y)

	classes=['anger', 'happiness', 'neutral', 'sadness', 'worry']

	dict={'anger':0}
	i=-1
	for clas in classes:
	  i+=1
	  dict.update({clas:i})

	y=np.zeros((Y.shape[0],5),dtype=int)
	for i in range(Y.shape[0]):
	  y[i][dict[Y[i]]]=1

	X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.1,random_state=42)

	def load_embeddings():
	    embeddings_index = {}
	    f = open('./glove.twitter.27B.100d.txt','r')
	    for line in f:
	        values = line.split()
	        word = values[0]
	        coefs = np.asarray(values[1:],dtype='float32')
	        embeddings_index[word] = coefs
	    f.close()
	    print('Found %s word vectors' %len(embeddings_index))
	    return embeddings_index

	embedding_index = load_embeddings()

	embedding_dim = 100
	embedding_matrix = np.zeros((max_features,embedding_dim))
	for word,i in word_index.items():
	    if i<max_features:
	        embedding_vector = embedding_index.get(word)
	        if embedding_vector is not None:
	            embedding_matrix[i] = embedding_vector


	loaded_model = tf.keras.models.load_model("sentimental_analysisv5.h5")

	def preprocessing(dataframe):
	    dataframe['SentimentText'] = dataframe['SentimentText'].apply(lambda x: " ".join(x.lower() for x in x.split()))
	    dataframe['SentimentText'] = dataframe['SentimentText'].str.replace('[^\w\s]',' ')
	    stop = stopwords.words('english')
	    dataframe['SentimentText'] = dataframe['SentimentText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
	    return dataframe

	def prepare(dataframe):
	    max_features = 10000
	    MAX_SEQUENCE_LENGTH = 15
	    tokenizer = Tokenizer(num_words=max_features, split=' ')
	    tokenizer.fit_on_texts(dataframe['SentimentText'].values)
	    word_index = tokenizer.word_index
	    X = tokenizer.texts_to_sequences(dataframe['SentimentText'].values)
	    X = pad_sequences(X,maxlen=MAX_SEQUENCE_LENGTH)
	    return X

	Y = loaded_model.predict_classes(X[10].reshape(1,MAX_SEQUENCE_LENGTH))

	input_text = InputText.get()

	x_unseen = {'SentimentText':[input_text]}
	x_unseen = pd.DataFrame(x_unseen)
	x_unseen = preprocessing(x_unseen)
	x_unseen = prepare(x_unseen)
	Y = loaded_model.predict_classes(x_unseen)

	temp_result = ""

	for name, age in dict.items():
		if age == Y[0]:
			temp_result += name

	print(dict)
	print(temp_result)

	showEmoji(temp_result)

master = tk.Tk()
master.geometry("800x500")

InputTextLabel = Label(master, text = "Enter Message: ", bg="black", fg="white").place(x = 30,y = 25)  
InputText = Entry(master, width="70")
InputText.place(x = 150, y = 25)  
checkEmotion = Button(master, text = "Check Emotion",activebackground = "black", activeforeground = "white", bg="white", command=finalFunction).place(x = 330, y = 65)

panel = tk.Label(master)
SadImagePath = "/home/nagadiapreet/Desktop/SDP/FrontEnd/emoji/sad.png"
SadEmotionImage = Image.open(SadImagePath)
SadEmotionImage = SadEmotionImage.resize((150, 150), Image.BILINEAR)
SadEmotionImage = ImageTk.PhotoImage(SadEmotionImage)

HappyImagePath = "/home/nagadiapreet/Desktop/SDP/FrontEnd/emoji/happy.png"
HappyEmotionImage = Image.open(HappyImagePath)
HappyEmotionImage = HappyEmotionImage.resize((150, 150), Image.BILINEAR)
HappyEmotionImage = ImageTk.PhotoImage(HappyEmotionImage)

AngryImagePath = "/home/nagadiapreet/Desktop/SDP/FrontEnd/emoji/Angry.png"
AngryEmotionImage = Image.open(AngryImagePath)
AngryEmotionImage = AngryEmotionImage.resize((150, 150), Image.BILINEAR)
AngryEmotionImage = ImageTk.PhotoImage(AngryEmotionImage)

NeutralImagePath = "/home/nagadiapreet/Desktop/SDP/FrontEnd/emoji/neutral.png"
NeutralEmotionImage = Image.open(NeutralImagePath)
NeutralEmotionImage = NeutralEmotionImage.resize((150, 150), Image.BILINEAR)
NeutralEmotionImage = ImageTk.PhotoImage(NeutralEmotionImage)

WorryImagePath = "/home/nagadiapreet/Desktop/SDP/FrontEnd/emoji/worry.png"
WorryEmotionImage = Image.open(WorryImagePath)
WorryEmotionImage = WorryEmotionImage.resize((150, 150), Image.BILINEAR)
WorryEmotionImage = ImageTk.PhotoImage(WorryEmotionImage)

master.title("AI Therapist")
master.configure(background='black')

master.mainloop()