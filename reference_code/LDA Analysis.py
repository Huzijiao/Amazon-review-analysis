'''
pip install PyDrive
pip install gensim
pip install pyldavis
python - m spacy download en'''

import gzip
import json
import os
import re
from itertools import chain
from string import punctuation

import gensim
import jieba
import matplotlib .pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim import corpora
from google.colab import auth , drive
from nltk import FreqDist , ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn. feature_extraction .text import CountVectorizer , TfidfVectorizer
from sklearn. linear_model import LogisticRegression
from sklearn. model_selection import train_test_split

import pyLDAvis
import pyLDAvis.gensim
import seaborn as sns
import spacy
from oauth2client .client import GoogleCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from wordcloud import WordCloud

drive.mount('/content/drive')

nltk.download('stopwords')

pd. set_option ("display. max_colwidth ", 200)

%matplotlib inline

hair_dryer = pd.read_csv('/content/drive/My Drive/data/ new_hair_dryer .csv')
microwave = pd.read_csv('/content/drive/My Drive/data/ new_microwave .csv')
pacifier = pd. read_csv('/content/drive/My Drive/data/ new_pacifier .csv')
hair_dryer1 = pd.read_csv('/content/drive/My Drive/ classify_byrating / hair_dryer /rating1.csv')
hair_dryer2 = pd.read_csv('/content/drive/My Drive/ classify_byrating / hair_dryer /rating2.csv')
hair_dryer3 = pd.read_csv('/content/drive/My Drive/ classify_byrating / hair_dryer /rating3.csv')
hair_dryer4 = pd.read_csv('/content/drive/My Drive/ classify_byrating / hair_dryer /rating4.csv')
hair_dryer5 = pd.read_csv('/content/drive/My Drive/ classify_byrating / hair_dryer /rating5.csv')
microwave1 = pd.read_csv('/content/drive/My Drive/ classify_byrating / microwave /rating1.csv')
microwave2 = pd.read_csv('/content/drive/My Drive/ classify_byrating / microwave /rating2.csv')
microwave3 = pd.read_csv('/content/drive/My Drive/ classify_byrating / microwave /rating3.csv')
microwave4 = pd.read_csv('/content/drive/My Drive/ classify_byrating / microwave /rating4.csv')
microwave5 = pd.read_csv('/content/drive/My Drive/ classify_byrating / microwave /rating5.csv')
pacifier1 = pd.read_csv('/content/drive/My Drive/ classify_byrating /pacifier/rating1.csv')
pacifier2 = pd.read_csv('/content/drive/My Drive/ classify_byrating /pacifier/rating2.csv')
pacifier3 = pd.read_csv('/content/drive/My Drive/ classify_byrating /pacifier/rating3.csv')
pacifier4 = pd.read_csv('/content/drive/My Drive/ classify_byrating /pacifier/rating4.csv')
pacifier5 = pd.read_csv('/content/drive/My Drive/ classify_byrating /pacifier/rating5.csv')

add_punc =' {}() %^ >.^ -=&#@'
add_punc = add_punc+ punctuation
h_head1 = hair_dryer1 . review_headline .tolist ()
m_head1 = microwave1 . review_headline .tolist ()
p_head1 = pacifier1 . review_headline .astype(str).tolist ()
h_head2 = hair_dryer2 . review_headline .tolist ()
m_head2 = microwave2 . review_headline .tolist ()
p_head2 = pacifier2 . review_headline .astype(str).tolist ()
h_head3 = hair_dryer3 . review_headline .tolist ()
m_head3 = microwave3 . review_headline .tolist ()
p_head3 = pacifier3 . review_headline .astype(str).tolist ()
h_head4 = hair_dryer4 . review_headline .tolist ()
m_head4 = microwave4 . review_headline .tolist ()
p_head4 = pacifier4 . review_headline .astype(str).tolist ()
h_head5 = hair_dryer5 . review_headline .tolist ()
m_head5 = microwave5 . review_headline .tolist ()
p_head5 = pacifier5 . review_headline .astype(str).tolist ()

h_body1 = hair_dryer1 . review_body .tolist ()
m_body1 = microwave1 . review_body .tolist ()
p_body1 = pacifier1 . review_body .astype(str).tolist ()
 h_body2 = hair_dryer2 . review_body .tolist ()
m_body2 = microwave2 . review_body .tolist ()
p_body2 = pacifier2 . review_body .astype(str).tolist ()
h_body3 = hair_dryer3 . review_body .tolist ()
m_body3 = microwave3 . review_body .tolist ()
p_body3 = pacifier3 . review_body .astype(str).tolist ()
h_body4 = hair_dryer4 . review_body .tolist ()
m_body4 = microwave4 . review_body .tolist ()
p_body4 = pacifier4 . review_body .astype(str).tolist ()
h_body5 = hair_dryer5 . review_body .tolist ()
m_body5 = microwave5 . review_body .tolist ()
p_body5 = pacifier5 . review_body .astype(str).tolist ()


def freq_words (x, filepath , terms =30):
all_words ='.join ([ text for text in x])
all_words = all_words .split ()

fdist = FreqDist( all_words)
words_df = pd. DataFrame(
{'word': list(fdist.keys ()),'count': list(fdist.values ())})

# selecting top 20 most frequent words
d = words_df.nlargest(columns="count", n=terms)
plt.figure(figsize =(20 , 5))
ax = sns.barplot(data=d, x="word", y="count")
ax.set(ylabel='Count')
 plt.show ()
plt.savefig(filepath)
hair_dryer ['review_body'] = hair_dryer ['review_body'].str.replace("n\'t", " not")

 # remove unwanted characters , numbers and symbols
hair_dryer ['review_body'] = hair_dryer ['review_body'].str.replace([^a-zA -Z#]", " ")
stop_words = stopwords .words('english')

# function to remove stopwords


def remove_stopwords (rev):
rev_new = " ".join ([i for i in rev if i not in stop_words ])
return rev_new


# remove short words (length < 3)
hair_dryer ['review_body'] = hair_dryer ['review_body'].apply(
lambda x:'.join ([w for w in x.split () if len(w) > 2]))

# remove stopwords from the text
reviews = [ remove_stopwords (r.split ()) for r in hair_dryer ['review_body']]

# make entire text lowercase
reviews = [r.lower () for r in reviews]

nlp = spacy.load('en', disable =['parser','ner'])


def lemmatization (texts , tags =['NOUN','ADJ']):
    output = []
    for sent in texts:
    doc = nlp(" ".join(sent))
    output.append ([ token.lemma_ for token in doc if token.pos_ in tags ])
    return output


tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split ())
reviews_2 = lemmatization ( tokenized_reviews )
reviews_3 = []
for i in range(len( reviews_2 )):
    reviews_3 .append(''.join( reviews_2 [i]))
freq_words (reviews_3 ,'/content/drive/My Drive/ classify_byrating / main_word_h', 35)
dictionary = corpora. Dictionary ( reviews_2)
doc_term_matrix = [ dictionary .doc2bow(rev) for rev in reviews_2]
LDA = gensim.models.ldamodel.LdaModel
lda_model = LDA(corpus=doc_term_matrix ,
id2word=dictionary ,
num_topics =7,
random_state =100 ,
chunksize =1000 ,
passes =50)
lda_model . print_topics ()
 # Visualize the topics
pyLDAvis. enable_notebook ()
vis = pyLDAvis.gensim.prepare(lda_model , doc_term_matrix , dictionary )
vis