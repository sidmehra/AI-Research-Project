# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:46:34 2020
MODEL-5- Bert Model

Feature Extraction-
         bert embedding
         
Classication-
        Input layer- Bert Embedding layer
                   - dense layer 256 filters  with Relu
        Output Layer- 1 Dense layer with sigmoid
        
Classification Report-
    precision    recall  f1-score   support

         0.0       0.87      0.91      0.89       593
         1.0       0.78      0.70      0.74       267

    accuracy                           0.85       860
   macro avg       0.83      0.81      0.82       860
weighted avg       0.84      0.85      0.85       860




"""

'''
pip install tensorflow_hub
pip install bert-for-tf2
pip install sentencepiece
'''
import tensorflow as tf
from sklearn import metrics
import pandas as pd
import tensorflow_hub as hub
import re
import numpy as np
from bert import tokenization
from tensorflow.keras.models import Model
from tqdm import tqdm
from tensorflow.keras import backend as K

# Initialize session
sess = tf.Session()

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


'''6#  Cleaning Process'''
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

class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="mean",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


# Build model
def build_model(max_seq_length):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    dense = tf.keras.layers.Dense(256, activation="relu")(bert_output)
    pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)
    
'''

###### process starts here
'''

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = 128

# Create datasets (Only take up to max_seq_length words for memory)
train_text = train['tweet'].tolist()
train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_l.tolist()

test_text = test['tweet'].tolist()
test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_l.tolist()


model = build_model(max_seq_length)

# Instantiate variables
initialize_vars(sess)

# Load BERT Word Embeddings 
test_input_ids=np.load("te_input_ids_l (1).npy")
test_input_masks=np.load("te_input_masks_l (1).npy")
test_segment_ids=np.load("te_input_segments_l (1).npy")
train_input_ids=np.load("tr_input_ids_l.npy")
train_input_masks=np.load("tr_input_masks_l.npy")
train_segment_ids=np.load("tr_input_segments_l.npy")
#model.fit(
#    [train_input_ids, train_input_masks, train_segment_ids],
#    train_label,
#    validation_data=(
#        [test_input_ids, test_input_masks, test_segment_ids],
#        test_label,
#    ),
#    epochs=5,
#    batch_size=32,
#)
#model.predict([test_input_ids, test_input_masks, test_segment_ids])

''' load json and create model '''
from keras.models import model_from_json

json_file = open('model_neural_network_bert.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_neural_network_bert.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
predicted = loaded_model.predict(test_input_ids)
predicted_r=np.round(predicted)
print(metrics.classification_report(predicted_r, test_l))
    
