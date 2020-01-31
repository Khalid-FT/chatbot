# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import re
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import pandas as pd


# preprocess query question
def preProccessQuery(sentence):
    # tokenize the pattern - split words into array
    sentence = sentence.split()
    # stem each word - create short form for word
    sentence = [ ps.stem(word) for word in sentence]
    return sentence

# vectorize query question
def vectQuery(sentence, words):
    # tokenize the pattern
    sentence = preProccessQuery(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    vect = [0]*len(words) 
    for s in sentence:
        for i,w in enumerate(words): # exemple [(0, «manger»), (1, «dormir»)] donc (i, «w»)
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                vect[i] = 1    
    return(np.array(vect))


# predict sentence
def predict(sentence, model , classes , words):
    # filter out predictions below a threshold
    vect = vectQuery(sentence, words) 
    res = model.predict(np.array([vect]))[0] 
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD] # exemple si on a deux classes : exemple [("class 0", 0.5), ("class 1", 0.90)] donc (i, «w»)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True) 
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# get response
def getResponse(predict, intents):
    tag = predict[0]['intent'] 
    list_of_intents = intents['intents'] 
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses']) 
            break
    return result

# app response
def chatbot_response(sentence, model , classes , words , intents):
    pred = predict(sentence, model , classes , words)
    res = getResponse(pred, intents)
    return res
