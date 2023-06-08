# Imports
import pandas
import numpy 
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from gensim import corpora, models, similarities, matutils
from corextopic import corextopic

df = pandas.read_csv('corpus_dataframe.csv')

# Stop words pickle file
with open ('stop_words.ob', 'rb') as fp:
    stop_words = pickle.load(fp)
print(df.columns)

# Declare a list that is to be converted into a column
target = ['Cardiologist', 'Ophthalmologist','ENT doctor', 'Dermatologist',
          'Medical Geneticist', 'Ob-Gyn', 'Pulmonologist',
          'Neurologist', 'Gastroenterologist', 'Orthopedist']
df['Target'] = pandas.Series(target)

# WORD EMBEDDING
# Create a CountVectorizer for parsing/counting words
count_vectorizer = CountVectorizer(stop_words=stop_words)
doc_word_cv = count_vectorizer.fit_transform(df['corpus_values'])

print(pandas.DataFrame(doc_word_cv.toarray(), index=df['Target'], columns = count_vectorizer.get_feature_names_out()).head())

# Create a TfidfVectorizer for parsing/counting words
tfidf = TfidfVectorizer(stop_words=stop_words)
doc_word_tfidf = tfidf.fit_transform(df['corpus_values'])

print(pandas.DataFrame(doc_word_tfidf.toarray(), index=df['Target'], columns = tfidf.get_feature_names_out()).head())

# TOPIC MODELING: LDA
corpus = matutils.Sparse2Corpus(doc_word_cv)
id2word = dict((v, k) for k, v in count_vectorizer.vocabulary_.items())

# Create lda model
lda = models.LdaModel(corpus=corpus, num_topics=10, id2word=id2word, passes=5)
lda.print_topics(10)

# PERFORMING COREX
words = list(numpy.asarray(count_vectorizer.get_feature_names_out()))
topic_model = corextopic.Corex(n_hidden=10, words=words, seed=1)
topic_model.fit(doc_word_cv, words=words, docs=df['corpus_values'])

topics = topic_model.get_topics()
for n,topic in enumerate(topics):
    topic_words,_,_ = zip(*topic)
    print('{}: '.format(n) + ','.join(topic_words))

# TOPIC MODELING : LSA
lsa = TruncatedSVD(21)
doc_topic = lsa.fit_transform(doc_word_cv)
print(lsa.explained_variance_ratio_)

topic_word = pandas.DataFrame(lsa.components_.round(10),
             index = ['component'+str(i) for i in range(10)],
             columns = count_vectorizer.get_feature_names_out())

print(topic_word)

tem_list = [] 
def display_topics(model, feature_names, no_top_words, topic_names=None):
    
    for ix, topic in enumerate(model.components_):
        inner_tem_list = []
       
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
            
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        inner_tem_list.append(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        tem_list.append(inner_tem_list)

result1 = display_topics(lsa, count_vectorizer.get_feature_names_out(), 10)

#tem_list
final_dic = {}
final_dic["Cardiologist"] = tem_list[0]
final_dic["Ophthalmologist"] = tem_list[1]
final_dic["ENT Doctor"] = tem_list[2]
final_dic["Dermatologist"] = tem_list[3]
final_dic["Medical Geneticist"] = tem_list[4]
final_dic["Ob-Gyn"] = tem_list[5]
final_dic["Pulmonologist"] = tem_list[6]
final_dic["Neurologist"] = tem_list[7]
final_dic["Gastroenterologist"] = tem_list[8]
final_dic["Orthopedist"] = tem_list[9]

print(final_dic)

tem_df = pandas.DataFrame.from_dict(final_dic, orient ='index') 
print(tem_df)

# Declare a list that is to be converted into a column
d_name = ['Cardiologist', 'Ophthalmologist','ENT doctor', 'Dermatologist',
          'Medical Geneticist', 'Ob-Gyn', 'Pulmonologist',
          'Neurologist', 'Gastroenterologist', 'Orthopedist']
 
# Using 'ch_no' as the column name and equating it to the list
tem_df['D_Name'] = d_name
tem_df = tem_df.rename(columns={0: 'Description'})
print(tem_df)

# create csv
tem_df.to_csv('diseases_with_description.csv', index=False)

