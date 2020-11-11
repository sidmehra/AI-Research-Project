# -*- coding: utf-8 -*-
'''
This python file will create bert embedding
input:loaded the preprocessed data
output:bert-embedding saved as npy
'''

#code taken from bert documentation
'''!pip install tensorflow==2.0
!pip install tensorflow_hub
!pip install bert-for-tf2
!pip install sentencepiece
'''

import tensorflow as tf
import tensorflow_hub as hub
print("TF version: ", tf.__version__)
print("Hub version: ", hub.__version__)

import tensorflow_hub as hub
import tensorflow as tf
import bert
FullTokenizer = bert.bert_tokenization.FullTokenizer
from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow
import math

max_seq_length = 128  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

# See BERT paper: https://arxiv.org/pdf/1810.04805.pdf
# And BERT implementation convert_single_example() at https://github.com/google-research/bert/blob/master/run_classifier.py

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids



vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

from google.colab import files
uploaded = files.upload()

# Loading preprocessed data
import io
import numpy as np
train=np.load(io.BytesIO(uploaded['train.npy']),allow_pickle=True)
test=np.load(io.BytesIO(uploaded['test.npy']),allow_pickle=True)
train_l=np.load(io.BytesIO(uploaded['train_l.npy']),allow_pickle=True)
test_l=np.load(io.BytesIO(uploaded['test_l.npy']),allow_pickle=True)

import pandas as pd
train=pd.DataFrame(train)
tr=train.iloc[:,0]
test=pd.DataFrame(test)
te=test.iloc[:,0]
len(test)

train_text = tr.tolist()
train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_l.tolist()

test_text = te.tolist()
test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_l.tolist()

tr_input_ids_l=[]
tr_input_masks_l=[]
tr_input_segments_l=[]
for i in range(len(tr)):
  stokens = tokenizer.tokenize(tr[i])
  stokens = ["[CLS]"] + stokens + ["[SEP]"]
  input_ids = get_ids(stokens, tokenizer, max_seq_length)
  input_masks = get_masks(stokens, max_seq_length)
  input_segments = get_segments(stokens, max_seq_length)
  tr_input_ids_l.append(input_ids)
  tr_input_masks_l.append(input_masks)
  tr_input_segments_l.append(input_segments)
te_input_ids_l=[]
te_input_masks_l=[]
te_input_segments_l=[]
for j in range(len(te)):
  stokens = tokenizer.tokenize(te[j])
  stokens = ["[CLS]"] + stokens + ["[SEP]"]
  input_ids = get_ids(stokens, tokenizer, max_seq_length)
  input_masks = get_masks(stokens, max_seq_length)
  input_segments = get_segments(stokens, max_seq_length)
  te_input_ids_l.append(input_ids)
  te_input_masks_l.append(input_masks)
  te_input_segments_l.append(input_segments)
  np.save("tr_input_ids_l.npy",tr_input_ids_l)
  np.save("te_input_ids_l.npy",te_input_ids_l)
  np.save("te_input_masks_l.npy",te_input_masks_l)
  np.save("tr_input_masks_l.npy",tr_input_masks_l)
  np.save("te_input_segments_l.npy",te_input_segments_l)
  np.save("tr_input_segments_l.npy",tr_input_segments_l)

from google.colab import files
files.download('/content/te_input_segments_l.npy')