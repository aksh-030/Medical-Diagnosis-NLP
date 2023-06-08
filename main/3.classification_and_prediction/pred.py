import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
from numpy import dot
from numpy.linalg import norm

# reading the stop words list with pickle
with open ('stop_words.ob', 'rb') as fp:
    domain_stop_word = pickle.load(fp)

# read data file
file_path = 'diseases_with_description.csv'
df = pd.read_csv(file_path)
#print(df.head())

def clean_text_func(text):
    
    """ this function clean & pre-process the data  """

    text = str(text)
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    final_text = ""
    for x in text.split():
        if x not in domain_stop_word:
            final_text = final_text + x  +" "
    return final_text

df['Description'] = df['Description'].apply(lambda x: clean_text_func(x))
df.head()

# WORDS EMBEDDING
cv = CountVectorizer(stop_words="english")
cv_tfidf = TfidfVectorizer(stop_words="english")

X = cv.fit_transform(list(df.loc[:, 'Description' ]))
X_tfidf = cv_tfidf.fit_transform(list(df.loc[:, 'Description' ]))


df_cv = pd.DataFrame(X.toarray() , columns=cv.get_feature_names_out())
df_tfidf = pd.DataFrame(X_tfidf.toarray() , columns=cv_tfidf.get_feature_names_out())

#print(df_cv.shape)
cosine = lambda v1 , v2 : dot(v1 , v2) / (norm(v1) * norm(v2))

# Cosine Similarity
new_text = [input('Detail your symptoms:\n')]
new_text_cv = cv.transform(new_text).toarray()[0]
new_text_tfidf = cv_tfidf.transform(new_text).toarray()[0]

for chpter_number in range(int(df.shape[0])):
    print(f"This is chapter number : {chpter_number} ")
    print(f"Cosin cv :    { cosine( df_cv.iloc[chpter_number]  , new_text_cv )} ")
    print(f"Cosin TFIDF : { cosine( df_tfidf.iloc[chpter_number]  , new_text_tfidf) } ")

# Implementing Logical Regression
#print(df.columns)

X_train = df.Description
y_train = df.D_Name

cv1 = CountVectorizer()
X_train_cv1 = cv1.fit_transform(X_train)
pd_cv1 = pd.DataFrame(X_train_cv1.toarray(), columns=cv1.get_feature_names_out())

lr = LogisticRegression()
lr.fit(X_train_cv1, y_train)

X_test = new_text
cleaned_text = clean_text_func(X_test)

X_test_cv3  = cv1.transform([cleaned_text])
y_pred_cv3 = lr.predict(X_test_cv3)
print("The patient is suggested to visit a ", y_pred_cv3)
