# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:12:58 2022

@author: henry
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np

## Get data from API
path ="http://api.agpaldataservice.ca/AgPalDataAPI/api/v2/query?format=xml&type=file"
data = pd.read_xml(path)
data.desc = data.desc.fillna('None')

## Extract urls
extracted_url = []

# The HTML string to be cleaned
for html_string in data.url:

  # Create a BeautifulSoup object
  soup = BeautifulSoup(html_string, 'html.parser')

  # Find all 'a' elements (which represent hyperlinks)
  links = soup.find_all('a')

  # Extract the 'href' attribute from each element
  for link in links:
      href = link['href']
      extracted_url.append(href)

data.url = extracted_url

## Remove html tags
def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result

data['shortDesc']=data['shortDesc'].apply(lambda cw : remove_tags(cw))
data['desc'] = data['desc'].astype(str).apply(lambda cw : remove_tags(cw))
data['desc_new']= np.where(data['desc']!='None', data['desc'], data['shortDesc'])

## Remove stop words
import nltk
# Get stop words
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Get stop words 
en_stop = set(nltk.corpus.stopwords.words('english'))
print(en_stop)


## Preprocess data 
"""
Data pre-processing
"""

# Lemmatization
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()

# Text cleaning function for gensim fastText word embeddings in python
def process_text(document):
     
        # Remove extra white space from text
        document = re.sub(r'\s+', ' ', document, flags=re.I)
         
        # Remove all the special characters from text
        document = re.sub(r'\W', ' ', str(document))
 
        # Remove all single characters from text
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
 
        # Converting to Lowercase
        document = document.lower()
 
        # Word tokenization       
        tokens = document.split()

        # Lemmatization using NLTK
        lemma_txt = [stemmer.lemmatize(word) for word in tokens]
        
        # Remove stop words
        lemma_no_stop_txt = [word for word in lemma_txt if word not in en_stop]
        
        # Drop words 
        tokens = [word for word in tokens if len(word) > 3]
                 
        clean_txt = ' '.join(lemma_no_stop_txt)
 
        return clean_txt
    
some_sent = data.desc_new

from tqdm import tqdm
print(tqdm(some_sent))

clean_corpus = [process_text(sentence) for sentence in tqdm(some_sent) if sentence.strip() !='']
clean_corpus_lst = [d.split() for d in clean_corpus]
 
#word_tokenizer = nltk.WordPunctTokenizer()
#word_tokens = [word_tokenizer.tokenize(sent) for sent in tqdm(clean_corpus)]

#print("\n\nword_tokens: ",word_tokens)
#print("\nclean_corpus_lst: ",clean_corpus_lst)