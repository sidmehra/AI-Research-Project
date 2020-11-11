# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:45:02 2020

@author: sid
"""

# Import the basic libraries 
import pandas as pd # For data manipulation and analysis
import numpy as np # For large and multi-dimensional arrays
import warnings # Ignoring unnecessory warnings
warnings.filterwarnings("ignore")  
import seaborn as sns 
import matplotlib.pyplot as plt 

# Some more libraries 
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Import the cleaning and pre-processing libraries 
import re
import string 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from wordsegment import load, clean, segment
from contractions import CONTRACTION_MAP
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder

def taskA_data_train():
    """
    This function returns the training data for sub-task A
    """
    # Loading the training data into the dataframe 
    train_data= pd.read_table('./dataset_files/training_taskABC.tsv', sep='\t')
    # Removing the columns of subtask A and subtask B 
    train_data= train_data.drop(['id', 'subtask_b', 'subtask_c'], axis = 1) 
    # Rename the column name of our target variable 
    train_data= train_data.rename(columns = {'subtask_a':'label_taskA'})
    # Collecting nan index
    drop_list=[]
    for i in range(0,train_data.shape[0]):
        j=train_data.values[i][1]
        if pd.isna(j):
                drop_list.append(i)
    # Droping records
    train_data = train_data.drop(train_data.index[[drop_list]],axis=0)
    train_data = train_data.reset_index(drop=True)
    
    return train_data

def taskA_data_test():
    """
    This function returns the test data for the sub-task A 
    """
    # Loading the feature test data into a data-frame
    feature_data= pd.read_table('./dataset_files/test_taskA.tsv', sep='\t')
    # Loading the label test data into a data-frame 
    label_data= pd.read_csv('./dataset_files/labels_taskA.csv')
    # Merge the 2 data frames to get a single one 
    test_data= pd.merge(feature_data, label_data, on='id')
    # Rename the column name of our target variable 
    test_data= test_data.rename(columns = {'label':'label_taskA'})
    # Remove the id column from the data frame 
    test_data= test_data.drop(['id'], axis=1)
    
    return test_data 

#---------------------FUNCTIONS FOR CLEANING AND PRE-PROCESS THE TRAIN/TEST DATA-------------------------------

def lowercasing(data):
    """
    Returns the data-frame with lower-cased twitter instances
    """
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return data

def apostrophe_conversion(data):
    """
    Converts the wrong apostrophe symbol to the correct representation
    """
    def convert_symbol(tweet):
        res= tweet.replace("’", "'")
        return res
    # Apply the method to whole series of tweet column 
    data['tweet']= data['tweet'].apply(convert_symbol)
    return data 
    
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

def contraction_expansion(data):
    """
    Returns the data-frame with the expanded contractions 
    """
    data['tweet']= data['tweet'].apply(expand_contractions)
    return data 

def punctuation_removal(data):
    """
    Remove all the punctuation symbols and special characters 
    """
    def remove_punctuation(tweet):
        result = tweet.translate(str.maketrans("","", string.punctuation))
        return result
    # Apply the method to whole series of tweet column 
    data['tweet']= data['tweet'].apply(remove_punctuation)
    return data 

def emoticons_removal(data):
    """
    Remove all the emoticons from the text 
    """
    data['tweet'] = data['tweet'].str.replace('[^\w\s]',' ')
    return data 
    
def numbers_removal(data):
    """
    Removes the numbers from the text 
    """
    def remove_numbers(tweet): 
        result = re.sub(r'\d+', '', tweet) 
        return result 
    # Apply the method to whole series of tweet column 
    data['tweet']= data['tweet'].apply(remove_numbers)
    return data

def usermentions_removal(data):
    """
    Removes the user mentions from each of the tweet instance
    """
    def remove_user(tweet):
        res= tweet.replace("user", "")
        return res
    # Apply the method to each of the tweet instance 
    data['tweet']= data['tweet'].apply(remove_user)
    return data

def urlmentions_removal(data):
    """
    Removes the user mentions from each of the tweet instance
    """
    def remove_url(tweet):
        res= tweet.replace("url", "")
        return res
    # Apply the method to each of the tweet instance 
    data['tweet']= data['tweet'].apply(remove_url)
    return data

def stopwords_removal(data):
    """
    Removes the stopwords from each of the tweet instance 
    """
    stop = stopwords.words('english')
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return data

def lemmatize(data):
    """
    Lemmatize the words in the data
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    data['tweet'] = data['tweet'].apply(lambda x: " ".join([wordnet_lemmatizer.lemmatize(word) for word in x.split()]))
    return data 

def stemming(data):
    """
    Stemming of the inflected words
    """
    ps = PorterStemmer()
    ss= SnowballStemmer("english")
    data['tweet'] = data['tweet'].apply(lambda x: " ".join([ss.stem(word) for word in x.split()]))
    return data 

def whitespaces_removal(data):
    """
    Remove the extra whitespaces 
    """
    def remove_whitespace(tweet): 
        res= " ".join(tweet.split()) 
        return res 
    # Apply the method to whole series of the tweet column 
    data['tweet']= data['tweet'].apply(remove_whitespace)
    return data 

def full_text(data):
    # Make a single list of individual tweets 
    tweets_list= [tweet for tweet in data.tweet]
    # Join all the elements of the list to get a single string 
    single_string = " ".join(tweets_list)
    return single_string

def clean_preprocess(data):
    """
    Clean and Pre-process the train/test data 
    """
    # Convert all the twitter instances into lower case 
    data= lowercasing(data)
    
    # Convert the ’(wrong apostrophe) symbol into '(right apostrophe) symbol
    data= apostrophe_conversion(data)
    
    # Expand the contractions 
    data= contraction_expansion(data)
    
    # Remove punctuations and special characters 
    data= punctuation_removal(data)
    
    # Remove the emoticons 
    data= emoticons_removal(data)
    
    # Remove the numbers from the text 
    data= numbers_removal(data)

    # Remove the user mentions from the text 
    data= usermentions_removal(data)

    # Remove the url mentions from the text 
    data= urlmentions_removal(data)
    
    # Remove the stopwords from the text 
    data= stopwords_removal(data)
    
    # Perform the lemmatisation of the words
    #data= lemmatize(data)
    
    # Perform the stemming of the words
    #data= stemming(data)
    
    # Remove extra whitespaces 
    data= whitespaces_removal(data)

    # Get the whole text as single string 
    single_string= full_text(data)
    
    # Encode the labels (target column)
    # Offensive tweet is encoded as 1 
    # Non-Offensive tweet is encoded as 0
    label_encoder = preprocessing.LabelEncoder() 
    data['label_taskA']= label_encoder.fit_transform(data['label_taskA']) 
    
    return data 

def word_cloud(data):
    """
    Display the Word-Cloud of the tweet data
    """
    # Get all the rows where label is OFFENSIVE (1)
    NOT_data = data[data.label_taskA.eq(0)]
    text = " ".join(review for review in NOT_data.tweet)
    print ("There are {} words in the combination of all review.".format(len(text)))
    stopwords = set(STOPWORDS)
    #stopwords.update(['people','even','know'])
    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    # Save the image in the img folder:
    wordcloud.to_file("word_cloud.png")
    

#--------------------------- FUNCTION FOR EXTRACTING THE FEATURES FROM TEXT DATA----------------------

def feature_extraction(train_data, test_data, model):
  """
   Perform the feature extraction using various models 
     1. UNIGRAM-----------------------> (1, 1)
     2. BIGRAM------------------------> (2, 2)
     3. TRIGRAM-----------------------> (3, 3)
     4. UNIGRAM + BIGRAM--------------> (1, 2) 
     5. BIGRAM + TRIGRAM--------------> (2, 3)
     6. UNIGRAM + BIGRAM + TRIGRAM----> (1, 3) 
  """
  if model == 'BOW':
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer= CountVectorizer(stop_words= 'english', ngram_range=(1,3))
  elif model== 'TF_IDF':
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer= TfidfVectorizer(stop_words='english', ngram_range=(1,1))
  
  # Encode the training feature data 
  X_train= vectorizer.fit_transform(train_data['tweet'])
  # Encode the test feature data 
  X_test= vectorizer.transform(test_data['tweet'])
  # Get all the feature names 
  feature_names= vectorizer.get_feature_names()
  # How many features are there 
  feature_column_size= len(feature_names)
    
  return X_train, X_test, feature_column_size

def word_embeddings(embedding, train_data, test_data):
    """ 
    1. Train Word2Vec for our own dataset
    2. Return the word vectors for train and test set 
    """
    if embedding=='Word2Vec':
        from gensim.models import Word2Vec 
        def word2vec(data):
            tweets_list = []
            for tweet in data['tweet'].values: 
                tweets=[]
                for word in tweet.split():
                    tweets.append(word)
                tweets_list.append(tweets)
            word2vec_model = Word2Vec(tweets_list, min_count=5, size=200, workers=4)
            # List for storing the vectors for each tweet 
            tweet_vectors = []
            for tweet in tweets_list:
                tweet_vector = np.zeros(50)
                count_words =0
                for word in tweet:
                    try:
                        tweet_vector += word2vec_model[word] 
                        count_words += 1
                    except:
                        pass
                tweet_vector / count_words
                tweet_vectors.append(tweet_vector)
            tweet_vectors_df= pd.DataFrame(tweet_vectors)
            tweet_vectors_df= tweet_vectors_df.fillna(0)
            return tweet_vectors_df
       
    X_train= word2vec(train_data)
    X_test= word2vec(test_data)
        
    return X_train, X_test
           
        
def get_stats(label_data_train):
    # How many data points for each class are present ? 
    # To check whether the dataset is balanced or imbalanced 
    unique, counts = np.unique(label_data_train, return_counts=True)
    stats= dict(zip(unique, counts))
    # Visualization of the distribution 
    return stats

#------------------------- FUNCTIONS FOR OVER-SAMPLING AND UNDERSAMPLING------------------------

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
def perform_oversampling(method, X_train, y_train):
    """
    Perform the Over-sampling on the training data
    """
    if method== 'SMOTE':
        sampler= SMOTE(random_state=0)
    elif method== 'RANDOM':
        sampler= RandomOverSampler(random_state=0)
    elif method== 'ADASYN':
        sampler= ADASYN(random_state=0)
    # Rebalance the training dataset 
    X_train, y_train= sampler.fit_sample(X_train, y_train)
    print('{} Oversampling performed \n'.format(method))
    return X_train, y_train 
    

from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks, NeighbourhoodCleaningRule, NearMiss
def perform_undersampling(method, X_train, y_train):
    """
    Perform the Under-sampling on the training data
    """
    if method== 'RANDOM':
        sampler = RandomUnderSampler(random_state=0)
    elif method== 'CLUSTER':
        sampler= ClusterCentroids(random_state=0)
    elif method== 'TOMEK':
        sampler= TomekLinks(random_state=0)
    elif method== 'NEIGH':
        sampler= NeighbourhoodCleaningRule(random_state=0)
    elif method== 'NEAR':
        sampler= NearMiss(random_state=0)
    # Rebalance the training dataset 
    X_train, y_train= sampler.fit_sample(X_train, y_train)
    print('{} Undersampling performed \n'.format(method))
    return X_train, y_train 
    

#---------------------------------BEGINNING OF OUR MAIN PROGRAM------------------------------
    
# Get the training data 
train_data= taskA_data_train()

# Get the test data 
test_data= taskA_data_test()

# Clean and Pre-process the train data 
train_data= clean_preprocess(train_data)

# Get the word-cloud of the cleaned training data 
#word_cloud(train_data)

# Clean and Pre-process the test data 
test_data= clean_preprocess(test_data)

#-------------------------------------- PERFORM FEATURE EXTRACTION------------------------------------------

# Feature extraction using Bag of Words / Tf-idf  / Bi-gram / Tri-gram 
model= 'BOW'
X_train, X_test, feature_column_size= feature_extraction(train_data, test_data, model)

# Feature Extraction using Word Embeddings 
#embedding= 'BOW'
#X_train, X_test= word_embeddings(embedding, train_data, test_data)


# Get the train and the test labels 
y_train= train_data['label_taskA'] 
y_test= test_data['label_taskA'] 

# Get the statistics about the distribution of the training data 
stats= get_stats(y_train)
print("Training data distribution before balancing the data")
print(stats)
print()

#--------------------------PERFORM OVER-SAMPLING/ UNDER-SAMPLING------------------------------

# Perform the over-sampling 
# Perform the oversampling technique on the training data 
# method= 'RANDOM'
# X_train, y_train= perform_oversampling(method, X_train, y_train)

# Perform the undersampling technique on the training data 
method= 'RANDOM'
X_train, y_train= perform_undersampling(method, X_train, y_train)

# Again check the data distribution after Sampling technique
stats= get_stats(y_train)
print("Training data distribution after balancing the data")
print(stats)
print()


#--------------------------Build, Train and Evaluate range of Machine Learning Models----------------------------

# Build range of machine learning models with default Hyperparameters
# Building the LOGISTIC REGRESSION model 
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state=0)
# Building the SVM model 
from sklearn.svm import SVC
svm = SVC(random_state=0)
# Building the NAIVE BAYES model 
from sklearn.naive_bayes import MultinomialNB
bayes = MultinomialNB()
# Building the DECISION TREE model
from sklearn.tree import DecisionTreeClassifier
decisionTree = DecisionTreeClassifier(random_state=0)
# Building the RANDOM FOREST model 
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(random_state=0)
# Building the GRADIENT BOOSTING CLASSIFIER 
from sklearn.ensemble import GradientBoostingClassifier
gradientBoosting = GradientBoostingClassifier(random_state=0)
# Building the ADA BOOST CLASSIFIER 
from sklearn.ensemble import AdaBoostClassifier
adaBoost = AdaBoostClassifier(random_state=0)

# Import the evaluation metrics 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


model_names = ['Logistic Regression', 'Naive Bayes', 'SVM', 'Decision tree', 'Random forest', 'Gradient Boosting', 'Ada Boost']
default_models = [logistic, bayes, svm, decisionTree, randomForest, gradientBoosting, adaBoost]


for model_name, default_model in zip(model_names, default_models):
    default_model = default_model.fit(X_train, y_train)
    predictions= default_model.predict(X_test)
    report = classification_report(y_test, predictions)
    print("The CLASSIFICATION REPORT for default {} is: {}\n".format(model_name, report))
    print("The CONFUSION MATRIX of {} is:".format(model_name))
    cm= confusion_matrix(y_test, predictions)
    print(cm)
    nonOffensive_instances= cm[0, 0] + cm[0, 1]
    offensive_instances= cm[1, 0] + cm[1, 1]
    non_class_acc= cm[0, 0]/nonOffensive_instances
    off_class_acc= cm[1, 1]/offensive_instances
    print("The non-offensive class accuracy is",non_class_acc)
    print("The offensive class accuracy is",off_class_acc)
    print("----------------------------------------------------------------------------------------")
    print("\n")

