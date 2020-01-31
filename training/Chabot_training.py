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
from keras.layers import Dense , Dropout

data = open('intents.json').read()
intents = json.loads(data)

# Apply on all the rows
corps = []
words = []
classes = []
documents = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
       pattern = re.sub('[^A-Za-z]' , ' ', pattern )
       pattern = pattern.lower()
       pattern = pattern.split()
       #add documents in the corpus
       documents.append((pattern, intent['tag']))
       #pattern =[word for word in pattern if not word in set(stopwords.words('english'))]
       pattern = [ ps.stem(word) for word in pattern]
       words.extend(pattern)
       pattern = ' '.join(pattern)
       corps.append(pattern)
       # add to our classes list
       if intent['tag'] not in classes:
            classes.append(intent['tag'])

# remove duplicate
words = list(set(words))

# extract  x train
train_x = []
for sentence in corps:
    bag = []
    tokens = sentence.split()
    for w in words:
       bag.append(1) if w in tokens else bag.append(0)
    train_x.append(bag)

# extract y train            
train_y = []
# create an empty array for our output
labels = [0] * len(classes)
# training set, bag of words for each sentence
for  doc in documents:
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    row = list(labels)
    row[classes.index(doc[1])] = 1
    train_y.append(row)

train_x_size = len(train_x[0])
train_y_size = len(train_y[0])

     
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


'''
# Create a Bag Of words 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
train_x = cv.fit_transform(corps).toarray()
'''


'''
# Split Data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_x,train_y,test_size = 0.2 , random_state =42)
'''

# ANN
# Initialising the ANN
classifier = Sequential()
# addin input layer
classifier.add(Dense( input_dim=train_x_size  , init='uniform' , activation='relu' , output_dim=64))
classifier.add(Dropout(p=0.1))
# Adding the first hidden layer
classifier.add(Dense(init='uniform' , output_dim=64, activation='relu'))
classifier.add(Dropout(p=0.1))
# Adding the output layer
classifier.add(Dense(init='uniform' , activation='softmax' ,  output_dim=train_y_size, ))
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
hist = classifier.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) 
classifier.save('chatbot_model.h5', hist)

