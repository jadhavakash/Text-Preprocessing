####-------------------Text Preprocessing in python----------------------------------------------####
#    Version   :     1.1
#    Author    :     Ravi PAUL
#    Date      :     18th Sept.2019
#
# we will cover following topics :
# 1. Basic feature extraction using text data
#     Number of words
#     Number of characters
#     Average word length
#     Number of stopwords
#     Number of special characters
#     Number of numerics
#     Number of uppercase words
# 2. Basic Text Pre-processing of text data
#     Lower casing
#     Punctuation removal
#     Stopwords removal
#     Frequent words removal
#     Rare words removal
#     Spelling correction
#     Tokenization
#     Stemming
#     Lemmatization
# 3. Advance Text Processing
#     N-grams
#     Term Frequency
#     Inverse Document Frequency
#     Term Frequency-Inverse Document Frequency (TF-IDF)
#     Bag of Words
#     Sentiment Analysis
#     Word Embedding
# 4. Performing Sentiment Analysis using Text Classification
# 5. Evaluation Metric: The metric used for evaluating the performance of classification model would be F1-Score.
#     The metric can be understood as -
#     True Positives(TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes.
#     True Negatives(TN) -
#     False Positives(FP)
#     False Negatives(FN)
#     Precision = TP / TP + FP
#     Recall = TP / TP + FN
#     F1 Score = 2 * (Recall * Precision) / (Recall + Precision)


    ######## ---------------------------------------------------------------------########
import numpy as np
import pandas as pd
import seaborn as sns
import nltk


train = pd.read_csv('C:/Training Classes/Python Programing/train_tweets.csv')
train.shape
train.columns
# 1.1 Number of Words
train['word_count'] = train['tweet'].apply(lambda x: len(str(x).split(" ")))
train[['tweet','word_count']].head()

# 1.2 Number of characters
train['char_count'] = train['tweet'].str.len() ## this also includes spaces
train[['tweet','char_count']].head()


# Average Word Length
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

train['avg_word'] = train['tweet'].apply(lambda x: avg_word(x))
train[['tweet','avg_word']].head()


# Number of stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')

train['stopwords'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
train[['tweet','stopwords']].head()


#  Number of special characters
train['hastags'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['tweet','hastags']].head()

#Number of numerics
train['numerics'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['tweet','numerics']].head()

#Number of Uppercase words
train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['tweet','upper']].head()


# 2. Basic Pre-processing
# Lower case

train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['tweet'].head()

#Removing Punctuation
train['tweet'] = train['tweet'].str.replace('[^\w\s]','')
train['tweet'].head()

#Remove numbers
import re
input_str = 'Box A contains 3 red and 5 white balls, while Box B contains 4 red and 2 blue balls.'
#input_str = train['tweet'][0]
result = re.sub(r'\d', '', input_str)
print(result)

# Removal of Stop Words
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['tweet'].head()

#White spaces removal
input_str = " \t a string example\t "
input_str = input_str.strip()
input_str

#Common word removal
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]
freq
freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()

#Rare words removal
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
freq
freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()

#Spelling correction
from textblob import TextBlob
train['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))

#Tokenization
TextBlob(train['tweet'][1]).words

#Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

# Lemmatization
from textblob import Word
train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['tweet'].head()

#. Advance Text Processing
# N-grams
# N-grams are the combination of multiple words used together. Ngrams with N=1 are called unigrams. Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used.
# Unigrams do not usually contain as much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the language structure,
# like what letter or word is likely to follow the given one. The longer the n-gram (the higher the n), the more context you have to work with. Optimum length really
# depends on the application – if your n-grams are too short, you may fail to capture important differences. On the other hand, if they are too long, you may fail
# to capture the “general knowledge” and only stick to particular cases.

TextBlob(train['tweet'][0]).ngrams(2)

# Term frequency Term frequency is simply the ratio of the count of a word present in a sentence, to the length of the sentence. Therefore, we can generalize term frequency as:
#  TF = (Number of times term T appears in the particular row) / (number of terms in that row)

tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1

#Inverse Document Frequency : The intuition behind inverse document frequency (IDF) is that a word is not of much use to us if it’s appearing in all the documents.
    # Therefore, the IDF of each word is the log of the ratio of the total number of rows to the number of rows in which that word is present.
    # IDF = log(N/n), where, N is the total number of rows and n is the number of rows in which the word was present.

for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['tweet'].str.contains(word)])))

tf1

# Term Frequency – Inverse Document Frequency (TF-IDF)
tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1

#We don’t have to calculate TF and IDF every time beforehand and then multiply it to obtain TF-IDF. Instead, sklearn has a separate function to directly obtain it:
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(train['tweet'])

train_vect


#Bag of Words
# Bag of Words (BoW) refers to the representation of text which describes the presence of words within the text data.
# The intuition behind this is that two similar text fields will contain similar kind of words, and will therefore have a similar bag of words.

from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(train['tweet'])
train_bow


#Sentiment Analysis
#   our problem was to detect the sentiment of the tweet. So, before applying any ML/DL models (which can have a separate feature
#   detecting the sentiment using the textblob library), let’s check the sentiment of the first few tweets.

train['tweet'][:5].apply(lambda x: TextBlob(x).sentiment)

# Above, we can see that it returns a tuple representing polarity and subjectivity of each tweet. Here, we only extract polarity as it indicates the sentiment
# as value nearer to 1 means a positive sentiment and values nearer to -1 means a negative sentiment. This can also work as a feature for building a machine learning model.

train['sentiment'] = train['tweet'].apply(lambda x: TextBlob(x).sentiment[0] )
train[['tweet','sentiment']].head()


