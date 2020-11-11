
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:46:34 2020

@author: egc
"""
'''
MODEL-4- CNN-LSTM 

Feature Extraction-
         word2vec embedding
         
Classication-
        Input layer- Embedding layer Relu Activation
                   - covolution layer 32 filters and Relu Activation 
                   - dropout layer 0.5
                   - max-pooling layer -2 pooling
                   - LSTM layer 
        Output Layer- 1 Dense layer with sigmoid
        
Classification Report-

                precision    recall  f1-score   support

         0.0       0.78      0.93      0.85       517
         1.0       0.85      0.59      0.70       343

    accuracy                           0.80       860
   macro avg       0.81      0.76      0.77       860
weighted avg       0.81      0.80      0.79       860

'''


'''
Library imports
'''
from keras.models import Sequential
from sklearn import metrics
from keras.layers import Dense,Dropout
import re
from contractions import CONTRACTION_MAP
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd #for data manipulation and analysis
import numpy as np #for large and multi-dimensional arrays
import warnings #Ignoring unnecessory warnings
warnings.filterwarnings("ignore")  


'''
read files
''' 
tsv_file_test='./Dataset/testset-levela.tsv'
tsv_file_train='./Dataset/olid-training-v1.0.tsv'
test_lab='./Dataset/labels-levela.csv'
test=pd.read_table(tsv_file_test,sep='\t')
train=pd.read_table(tsv_file_train,sep='\t')
test_l=pd.read_csv(test_lab)

'''
drop unused feature and records
'''
train.drop(["id","subtask_b", "subtask_c"],axis=1, inplace = True)
test.drop(["id"],axis=1, inplace = True)
test_l.drop(["id"],axis=1, inplace = True)
#collecting nan index
drop_list=[]
for i in range(0,train.shape[0]):
    j=train.values[i][1]
    if pd.isna(j):
        drop_list.append(i)
#droping records
train=train.drop(train.index[[drop_list]],axis=0)
train = train.reset_index(drop=True)
'''
basic preprocessing steps
'''

'''#1 Lower case'''

train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['tweet'].head() 
test['tweet'] = test['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
test['tweet'].head()

'''#2 Removing Punctuation'''

train['tweet'] = train['tweet'].str.replace('[^\w\s]',' ')
train['tweet'].head()
test['tweet'] = test['tweet'].str.replace('[^\w\s]',' ')
test['tweet'].head()

''''#3 Removal of Stop Words'''
#import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['tweet'].head()
test['tweet'] = test['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
test['tweet'].head()

''' To split Hashtag Words '''
import wordninja
train['tweet']=train['tweet'].apply(lambda x:' '.join(wordninja.split(str(x))))
test['tweet']=test['tweet'].apply(lambda x:' '.join(wordninja.split(str(x))))


'''#4  Spelling correction'''
# * pip install textblob  * in anaconda prompt

#from textblob import TextBlob
#train['tweet']=train['tweet'].apply(lambda x: str(TextBlob(x).correct()))
#train['tweet'].head()
#test['tweet']=test['tweet'].apply(lambda x: str(TextBlob(x).correct()))
#test['tweet'].head()

'''#5 Lemmatization'''

#import nltk
#nltk.download('wordnet')

from textblob import Word
train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['tweet'].head()
test['tweet'] = test['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
test['tweet'].head()
'''6#  Your Cleaning Process'''
def remove_numbers(tweet): 
        result = re.sub(r'\d+', '', tweet) 
        return result 
def remove_user(tweet):
        res= tweet.replace("user", "")
        return res
def remove_url(tweet):
        res= tweet.replace("url", "")
        return res 

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
def convert_symbol(tweet):
        res= tweet.replace("’", "'")
        return res

train['tweet']= train['tweet'].apply(expand_contractions)
test['tweet']= test['tweet'].apply(expand_contractions)    
# Apply the method to whole series of tweet column 
train['tweet']= train['tweet'].apply(convert_symbol)
test['tweet']= test['tweet'].apply(convert_symbol)
# Apply the method to whole series of tweet column 
train['tweet']= train['tweet'].apply(remove_numbers)
test['tweet']= test['tweet'].apply(remove_numbers)
# Apply the method to each of the tweet instance 
train['tweet']= train['tweet'].apply(remove_user)
test['tweet']= test['tweet'].apply(remove_user)
# Apply the method to each of the tweet instance 
train['tweet']= train['tweet'].apply(remove_url)
test['tweet']= test['tweet'].apply(remove_url)

'''
Advance Text Processing
'''
'''
###Encode lable
'''
from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
train_l=train.iloc[:,1]
train_l = Encoder.fit_transform(train_l)
test_l = Encoder.fit_transform(test_l)
sd=np.concatenate((train['tweet'], test['tweet']), axis=0)

#vectorization
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(sd)

X_train = vectorizer.transform(train['tweet'])
X_test  = vectorizer.transform(test['tweet'])

#sampling
from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
X_train, train_l= undersample.fit_resample(X_train, train_l)

#X_train=X_train.toarray()
#X_test=X_test.toarray()
#X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
#X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

'''model creation'''
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3,input_shape=(X_train.shape[1],1),activation='relu'))
#model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add((LSTM(16)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#history=model.fit(X_train, train_l, validation_data=(X_test, test_l),epochs=20, batch_size=10)

''' load json and create model '''
from keras.models import model_from_json

json_file = open('model_neural_network_4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_neural_network_4.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
predicted = loaded_model.predict(X_test)
predicted_r=np.round(predicted)
print(metrics.classification_report(predicted_r, test_l))
