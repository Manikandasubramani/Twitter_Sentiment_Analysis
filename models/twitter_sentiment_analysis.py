# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import re
import string
import pickle

# importing data
train_df = pd.read_csv('../dataset/train.csv')
test_df = pd.read_csv('../dataset/test.csv')

# Checking for missing values
train_df.isnull().sum().sort_values(ascending=False)
test_df.isnull().sum().sort_values(ascending=False)

# Dropping null values
train_df = train_df.dropna()
test_df = test_df.dropna()

# Seperate Label
label = train_df['selected_text'].values
train_df = train_df.drop(['selected_text'], axis=1)

# Drop the irrevelant parameter
train_df = train_df.drop(['textID'], axis=1)
test_df = test_df.drop(['textID'], axis=1)

# print(train_df.head())
# print(test_df.head())

# Combine dataset
df = pd.concat([train_df, test_df], axis=0).reset_index()
print('There are {} rows and {} columns in train'.format(train_df.shape[0],train_df.shape[1]))
print('There are {} rows and {} columns in test'.format(test_df.shape[0],test_df.shape[1]))
print('There are {} rows and {} columns in total'.format(df.shape[0],df.shape[1]))

# Save the size of train and test dataset
train_size = train_df.shape[0]
test_size = test_df.shape[0]

# remove URL's / HTMLS / Emojis / email / numbers / duplicate / Punctuations
df.loc[:, 'text'] = df.loc[:, 'text'].apply(lambda x: re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", x))
df.loc[:, 'text'] = df.loc[:, 'text'].apply(lambda x: re.sub(r'<.*?>', "", x))
df.loc[:, 'text'] = df.loc[:, 'text'].apply(lambda x: re.sub(re.compile(r"["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE), "", x))

df.loc[:, 'text'] = df.loc[:, 'text'].apply(lambda x: re.sub(r'[0-9]', "", x))
df.loc[:, 'text'] = df.loc[:, 'text'].apply(lambda x: re.sub(r'(\w)\1+', "", x))
df.loc[:, 'text'] = df.loc[:, 'text'].apply(lambda x: x.translate(str.maketrans(dict.fromkeys(string.punctuation))))

# Remove the tags in comments
df.loc[:, 'text'] = df.loc[:, 'text'].apply(lambda x: re.sub(re.compile(r'[@|#][^\s]+'), "", x))

# Removing the stop words using stemming
stemmer  = SnowballStemmer('english')
stopword = stopwords.words('english')

df.loc[:, 'text'] = df.loc[:, 'text'].apply(lambda x:' '.join([stemmer.stem(i) for i in x.split() if i not in stopword]))

map = {'positive': 1, 'negative':0, 'neutral': 2}
df['sentiment'] = df['sentiment'].map(map)

# train and test split
X_train, X_test, Y_train, Y_test = train_test_split(df['text'].values, df['sentiment'].values, test_size=0.21, random_state=42)

train_data = pd.DataFrame({'text':X_train, 'sentiment':Y_train})
test_data = pd.DataFrame({'text':X_test, 'sentiment':Y_test})

# initializing the TFIDF vectorizer
vectorizer = TfidfVectorizer(min_df = 0.0005, max_features = 100000)

train_vectors = vectorizer.fit_transform(train_data['text'])
test_vectors = vectorizer.transform(test_data['text'])

# SVM Classification
classifier = svm.SVC(kernel='linear', C = 1.0)
classifier.fit(train_vectors, train_data['sentiment'])
prediction = classifier.predict(test_vectors)
print(prediction)
print(classifier.score(test_vectors,test_data['sentiment']))

#classification report
report = classification_report(test_data['sentiment'], prediction, output_dict=True)
print('positive:', report['1']['recall'])
print('negative:', report['0']['recall'])
print('neutral:', report['2']['recall'])

pickle.dump(vectorizer, open('../models/transform.pkl', 'wb'))
pickle.dump(classifier, open('../models/model.pkl', 'wb'))
