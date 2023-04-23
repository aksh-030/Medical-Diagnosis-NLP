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
target = ['CardioVascular', 'Ear', 'Nose', 'Throat']
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
lda = models.LdaModel(corpus=corpus, num_topics=4, id2word=id2word, passes=5)
lda.print_topics(4)

# PERFORMING COREX
words = list(numpy.asarray(count_vectorizer.get_feature_names_out()))
topic_model = corextopic.Corex(n_hidden=4, words=words, seed=1)
topic_model.fit(doc_word_cv, words=words, docs=df['corpus_values'])

topics = topic_model.get_topics()
for n,topic in enumerate(topics):
    topic_words,_,_ = zip(*topic)
    print('{}: '.format(n) + ','.join(topic_words))

# TOPIC MODELING : LSA
lsa = TruncatedSVD(4)
doc_topic = lsa.fit_transform(doc_word_cv)
print(lsa.explained_variance_ratio_)

topic_word = pandas.DataFrame(lsa.components_.round(4),
             index = ['component'+str(i) for i in range(4)],
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

result1 = display_topics(lsa, count_vectorizer.get_feature_names_out(), 20)

tem_list
final_dic = {}
final_dic["Heart"] = tem_list[0]
final_dic["Ear"] = tem_list[1]
final_dic["Nose"] = tem_list[2]
final_dic["Throat"] = tem_list[3]

print(final_dic)

tem_df = pandas.DataFrame.from_dict(final_dic, orient ='index') 
print(tem_df)

# Declare a list that is to be converted into a column
d_name = ['CardioVascular', 'Ear', 'Nose', 'Throat']
 
# Using 'ch_no' as the column name and equating it to the list
tem_df['D_Name'] = d_name
tem_df = tem_df.rename(columns={0: 'Description'})
print(tem_df)

# create csv
tem_df.to_csv('diseases_with_description.csv', index=False)

