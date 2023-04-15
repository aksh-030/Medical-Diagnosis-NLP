# Data read/write
import PyPDF2
import numpy as np
import pandas as pd

# Data Visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Function to convert from list to str
def listToString(s): 
    
    # initialize an empty string
    str1 = " " 
    
    # return string  
    return (str1.join(s))

def get_freq_word(TEXT_list):
    """This method get count the frequency of each word in the passed parameter"""
    Vocab_list = ' '.join(TEXT_list)
    temp_dict = Counter(Vocab_list.split()).items()
    freq_df = pd.DataFrame(temp_dict)
    freq_df = freq_df.sort_values(1, ascending = False)
    return freq_df

# specify file path 
path = "GL-MDM.pdf"

# open file and read it through PyPDF2 packages
ff = open( path ,"rb")
pdfReader = PyPDF2.PdfReader(ff)

# print the total pages number
print(len(pdfReader.pages))
total_page = len(pdfReader.pages)

list_from_pdf =[]
article = " "

# loop through the pages and assign/save the page content in list and str
for page_num  in np.arange(0 , total_page, dtype=int):
    #print(type(page_num))
    page_number=page_num.item()
    #print(type(page_number))
    pageObj = pdfReader.pages[page_number]
    extract = pageObj.extract_text().split("\n")
    article += listToString(extract)
    list_from_pdf.append(extract)

# save the book content in dataframe
df = pd.DataFrame([article], columns=['string_values'])

# DATA VIZ
All_text = list(df.string_values)
All_text = " ".join(All_text)
#word_cloud2 = WordCloud(collocations = False, background_color = 'white').generate(All_text)
#plt.figure(figsize=(15,10))
#plt.imshow(word_cloud2, interpolation='bilinear')
#plt.axis("off")
#plt.show()

# plotting common words
#freq_words = get_freq_word(df.string_values)
#freq_words.columns = ['WORD', 'COUNT']

#plt.figure(figsize=(15,10))
#plt.bar(freq_words.WORD[:20], freq_words.COUNT[:20], color ='#ceeffa',
 #       width = 0.4)
#plt.xticks(rotation=90 ,fontsize = 15)
#plt.yticks(fontsize = 15)
#plt.xlabel("Words" , fontsize = 25)
#plt.ylabel("Repetition", fontsize = 25)
#plt.title("Most Frequent Word", fontsize = 30)
#plt.show();

# Import necessary modules
import ssl
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
ssl._create_default_https_context = ssl._create_unverified_context

import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# Import WordNetLemmatizer, and stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# import spcay framework
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.symbols import amod

# Import Counter
from collections import Counter

# Tokenize the article: tokens
tokens = word_tokenize(All_text)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
#print(bow_simple.most_common(10))

# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()]

# Remove all stop words: no_stops
no_stops = [t for t in alpha_only if t not in stopwords.words('english')]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# Create the bag-of-words: bow
bow = Counter(lemmatized)

# Print the 200 most common tokens
#print(bow.most_common(200))

# SpaCy to DEFINE ADJECTIVES
df['spacy_doc'] = list(nlp.pipe(df.string_values))

doc_adj = [token.text.lower() for doc in df.spacy_doc for token in doc if token.pos_=='ADJ']
doc_noun = [token.text.lower() for doc in df.spacy_doc for token in doc if token.pos_=='NOUN']

# common adjectives
Counter(doc_adj).most_common(10)

# common nouns
Counter(doc_noun).most_common(10)

# save the adjective words in list
adj_list = list(set(doc_adj))

print(len(adj_list))

# defined stop words in topic domain
  
stop_words = ["patient", "may", "disease", "cause", "treatment", "also", "symptom", "usually", "sign",
                "diagnosis", "result", "pain", "include", "pressure", "lung", "pulmonary", "respiratory",
                "chest", "fluid", "complication", "change", "blood", "infection", "therapy", "prevent",
                "acute", "care", "child", "level", "air", "use", "severe", "help", "used", "exercise",
                "normal", "incidence", "pneumonia","tissue", "show", "chronic", "failure", "cast", "increased",
                "monitor", "hypoxemia", "produce", "edema", "increase", "space", "occurs", "cough", "alveolar", 
                "heart", "pathophysiology", "sputum", "provide", "decreased", "pneumothorax", "test", "special",
                "tube", "condition", "common", "surgery","secretion", "fibrosis", "disorder", "pa", "area", "form",
                "cell", "skin", "drainage", "tb", "year", "commonly", "check", "teach", "rest", "watch", "encourage", 
                "underlying", "consideration", "et", "early", "hour", "family", "need", "effusion", "body", "drug", "support", 
                "rate", "syndrome", "requires", "inflammation", "abg", "side", "infant", "however", "upper", "cor", "pulmonale",
                 "ventilator", "mechanical", "breath", "maintain" , "foot", "day", "bed", "parent", "especially", "fever", "culture",
                'system', 'within', 'factor', 'amount', 'death', 'movement', 'progress', 'volume', 'one', 'stage', 'report',
                'avoid', 'respiration', 'trauma', 'occur', 'atelectasis', 'hand', 'includes', 'weight', 'tendon', 'hypertension', 
                'le', 'time', 'lead', 'damage', 'causing', 'require', 'activity', 'injury', 'risk', 'mm', 'measure', 'examination',
                'nerve', 'stress', 'make', 'al', 'see', 'decrease', 'age', 'hg''case', 'month', 'coughing', 'develops', 'formation', 
                'without', 'site', 'every', 'reduce', 'relieve', 'effect','percussion', 'ordered', 'develop', 'affect', 'loss', 'flow',
                'lesion', 'technique', 'exposure', 'gas', 'finding', 'procedure', 'begin', 'wall', 'immediately', 'type', 'response', 
                'position', 'needed', 'administer', 'control', 'ass', 'increasing', 'although', 'tell', 'output', 'give', 'analysis',
                'history', 'often' ,'week', 'home', 'perform','function', 'typically', 'frequently', 'adult', 'indicate', 'administration',
                'explain', 'using', 'suggest', 'called', 'center', 'head', 'people', 'resulting', 'including', 'period', 'feature'
                   ]

# marge (adjective & topic domain stop words) list
new_stopwords = adj_list+stop_words

stpwrd = stopwords.words('english')

print(f' adjective words = {len(adj_list)}')
print(f' topic domain stop words = {len(stop_words)}')
print(f' marge adj_list & stop_words =  {len(new_stopwords)}')
print(f' english stop words = {len(stpwrd)}')

stpwrd.extend(new_stopwords)

print(f'after marge all of the stop words = {len(stpwrd)}')

# Remove all stop words: no_stops
no_stops01 = [t for t in lemmatized if t not in stpwrd ]

# Create the bag-of-words: bow
bow = Counter(no_stops01)

# Print the 100 most common tokens
#print(bow.most_common(100))

# PLOT CLOUD AFTER PREPROCESS
All_text = " ".join(no_stops01)
word_cloud2 = WordCloud(collocations = False, background_color = 'white').generate(All_text)
plt.figure(figsize=(15,10))
plt.imshow(word_cloud2, interpolation='bilinear')
plt.axis("off")
plt.show()
# most freq words
freq_words = get_freq_word(no_stops01)
freq_words.columns = ['WORD', 'COUNT']

plt.figure(figsize=(15,10))
plt.bar(freq_words.WORD[:20], freq_words.COUNT[:20], color ='#ceeffa',
        width = 0.4)
plt.xticks(rotation=90 ,fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("Words" , fontsize = 25)
plt.ylabel("Repetition", fontsize = 25)
plt.title("Most Frequent Word", fontsize = 30)

# SAVE DATAFRAME INTO CSV FILE
corpus = " ".join(no_stops01)
string = corpus
df01 = pd.DataFrame([string], columns=['string_values'])

df01.to_csv('dataframe.csv', index=False)

# save stop words list in pickle file, to use it later
import pickle
with open('stop_words.ob', 'wb') as fp:
    pickle.dump(stpwrd, fp)
plt.show();
