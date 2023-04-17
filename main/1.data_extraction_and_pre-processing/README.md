# Data Extraction and Pre-processing

Data is extracted and processed from handbook in the following steps:
  + Extract strings of text from each page using PyPdf2
  + Tokenize extracted strings
  + Remove stop words using NLTK
  + Lemmatize the output
  + Again remove stopwords, including adjectives and domain-specific stopwords
  + Save dataframe as a csv file
