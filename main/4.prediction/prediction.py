import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
import re
import spacy
import numpy 
from numpy import dot
from numpy.linalg import norm
nlp = spacy.load('en_core_web_sm')
import pickle

# reading the stop words list with pickle
with open ('stop_words.ob', 'rb') as fp:
    domain_stop_word = pickle.load(fp)

file_path = "diseases_with_description.csv"
all_chapters = pd.read_csv(file_path)

cv = CountVectorizer(stop_words="english")
cv_tfidf = TfidfVectorizer(stop_words="english")

X = cv.fit_transform(list(all_chapters.loc[:, 'Description' ]))
X_tfidf = cv_tfidf.fit_transform(list(all_chapters.loc[:, 'Description' ]))

df_cv = pd.DataFrame(X.toarray() , columns=cv.get_feature_names())
df_tfidf = pd.DataFrame(X_tfidf.toarray() , columns=cv_tfidf.get_feature_names())

cosine = lambda v1 , v2 : dot(v1 , v2) / (norm(v1) * norm(v2))

new_text = ["dizziness loss of balance  vomiting tinnitus of hearing in the high frequency range in one ear difficulty focusing your eyes "]
new_text_cv = cv.transform(new_text).toarray()[0]
new_text_tfidf = cv_tfidf.transform(new_text).toarray()[0]

for chpter_number in range(int(all_chapters.shape[0])):
    print(f"This is chpter number : {chpter_number} ")
    print(f"Cosin cv :    { cosine( df_cv.iloc[chpter_number]  , new_text_cv )} ")
    print(f"Cosin TFIDF : { cosine( df_tfidf.iloc[chpter_number]  , new_text_tfidf) } ")

    
