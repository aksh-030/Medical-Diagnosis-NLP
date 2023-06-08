# Imports
import PyPDF2
import numpy
import pandas
import os
import csv
import ssl
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
ssl._create_default_https_context = ssl._create_unverified_context
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.symbols import amod
from collections import Counter

# Functions
def listToString(s): 
    string = " "  
    return (string.join(s))

# Create csv
with open('corpus_dataframe.csv', 'w') as file: 
   pass 

# Stop words
sw_list=stopwords.words('english')
sw_domain = ["patient", "may", "disease", "cause", "duration", "treatment", "also", "symptom", "usually", "sign", "diagnosis",
             "include", "pulmonary", "respiratory", "complication", "change", "infection", "therapy", "prevent", "acute",
             "care", "child", "level", "air", "use", "severe", "help", "used", "exercise", "normal", "incidence", "pneumonia",
             "tissue", "show", "chronic", "failure", "cast", "increased", "monitor", "produce", "increase", "space", "occurs",
             "alveolar", "pathophysiology", "sputum", "provide", "decreased", "pneumothorax", "test", "special", "tube", "condition",
             "common", "surgery", "fibrosis", "disorder", "pa", "area", "form", "cell", "drainage", "tb", "year", "commonly",
             "check", "teach", "rest", "watch", "encourage", "underlying", "consideration", "et", "early", "hour", "family", "need",
             "effusion", "body", "drug", "support", "rate", "syndrome", "requires", "abg", "side", "infant", "however", "upper",
             "cor", "pulmonale", "ventilator", "mechanical", "breath", "maintain" , "foot", "day", "bed", "parent", "especially",
             "culture", "system", "within", "factor", "amount", "death", "movement", "progress", "volume", "one", "stage", "report",
             "avoid", "respiration", "trauma", "occur", "atelectasis", "hand", "includes", "weight", "tendon", "le", "time", "lead",
             "damage", "causing", "require", "activity", "injury", "risk", "mm", "measure", "examination", "nerve", "stress", "make",
             "al", "see", "decrease", "age", "hg", "case", "month", "develops", "formation", "without", "site", "every", "reduce",
             "relieve", "effect", "percussion", "ordered", "develop", "affect", "loss", "flow", "lesion", "technique", "exposure",
             "gas", "finding", "procedure", "begin", "wall", "immediately", "type", "response", "position", "needed", "administer",
             "control", "increasing", "although", "tell", "output", "give", "analysis", "history", "often", "week", "home",
             "perform", "function", "typically", "frequently", "adult", "indicate", "administration", "explain", "using", "suggest",
             "called", "center", "head", "people", "resulting", "including", "period", "feature", "result", "environment", "fail",
             "quality", "outcome", "tool", "question", "identify", "appropriate", "cause", "description", "classic", "ascertain",
             "benefit", "potential", "thraten", "life", "send", "set", "remember", "active", "establish", "assess", "guide",
             "professional", "title", "signs", "symptoms", "treatment", "diagnosis", "test", "indicate", "situation",
             "edition", "copyright"]

# File extraction
folder_with_pdfs = 'D:\\Projects\\Medical-Diagnosis-NLP-main\\textbook_source\\signs_and_symptoms'
linesOfFiles = []
for pdf_file in os.listdir(folder_with_pdfs):
    if pdf_file.endswith('.pdf'):
        file_path = os.path.join(folder_with_pdfs, pdf_file)
        file = open( file_path ,"rb")
        reader = PyPDF2.PdfReader(file)
        total_page = len(reader.pages)

        list_from_pdf =[]
        article = " "

        for page_num  in numpy.arange(0 , total_page, dtype=int):
            page_number=page_num.item()
            pageObj = reader.pages[page_number]
            extract = pageObj.extract_text().split("\n")
            article += listToString(extract)
            list_from_pdf.append(extract)

        df = pandas.DataFrame([article], columns=['strings'])

        All_text = list(df.strings)
        All_text = " ".join(All_text)

        # Tokenize
        tokens = word_tokenize(All_text)
        tokens_lower = [w.lower() for w in tokens]
        tokens_alpha = [w for w in tokens_lower if w.isalpha()]

        # Stop Word Removal
        no_stops = [w for w in tokens_alpha if w not in sw_list]

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(w) for w in no_stops]

        # adjectives
        df['spacy_doc'] = list(nlp.pipe(df.strings))
        doc_adj = [token.text.lower() for doc in df.spacy_doc for token in doc if token.pos_=='ADJ']
        adj_list = list(set(doc_adj))

        # stop words including topic domain & adj
        sw_list.extend( adj_list + sw_domain )
        tokens_nostop = [t for t in lemmatized if t not in sw_list ]

        # Dataframe to CSV
        corpus = " ".join(tokens_nostop)
        df_corpus = pandas.DataFrame([corpus], columns=['corpus_values'])
        df_corpus.to_csv('corpus_dataframe.csv', mode='a', index=False, header=False)

# Save stop words
import pickle
with open('stop_words.ob', 'wb') as fp:
    pickle.dump(sw_list, fp)
