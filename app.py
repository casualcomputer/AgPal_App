import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pickle
import torch
import pandas as pd 
import json
import requests
import re

# https://curlconverter.com/
# https://www.convertcsv.com/json-to-csv.htm

## Remove html tags
def remove_tags(string):
    result = re.sub('<.*?>','',string).strip()
    return result


def search_key_generator(keyword_list):
    search_string = ''
    n_keyword = len(keyword_list)
    counter = 0 
    if len(keyword_list) > 0 and len(keyword_list[0]):
      for keyword in keyword_list:
        if counter < n_keyword-1:
          search_string += 'keyword:{' + keyword +'} '
        else:
          search_string += 'keyword:{' + keyword +'}' 
        counter +=1
      return search_string

#search_keys = search_key_generator(["Ontario","Funding"])

def agpal_search_results(keyword_list, top_k):
  search_keys = search_key_generator(keyword_list)
  headers = {
      'authority': 'prodapi.agpal.ca',
      'accept': 'application/json, text/plain, */*',
      'accept-language': 'en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7',
      'content-type': 'multipart/form-data; boundary=----WebKitFormBoundary7qAY07ZLedVRqMnX',
      'origin': 'https://agpal.ca',
      'referer': 'https://agpal.ca/',
      'sec-ch-ua': '"Not?A_Brand";v="8", "Chromium";v="108", "Microsoft Edge";v="108"',
      'sec-ch-ua-mobile': '?0',
      'sec-ch-ua-platform': '"Windows"',
      'sec-fetch-dest': 'empty',
      'sec-fetch-mode': 'cors',
      'sec-fetch-site': 'same-site',
      'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.42',
  }
  if search_keys!= None:
    data = f'------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="queryaction"\r\n\r\nfieldquery\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="fq"\r\n\r\n{search_keys}\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="lang"\r\n\r\nen\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="pageSize"\r\n\r\n25\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="sort"\r\n\r\nscore:desc\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="sort"\r\n\r\nsort2:asc\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX--\r\n'
    response = requests.post('https://prodapi.agpal.ca/api/v1/search', headers=headers, data=data)
    json_doc = list(json.loads(json.dumps(response.json())).values())[1] #relevant search results
    agpal_results = []
    for i in json_doc:
      agpal_results.append([i.get("ID"),i.get("title"), i.get("shortDesc")])
    out_df = pd.DataFrame(agpal_results, columns = ['ID', 'title',"shortDesc"]).head(top_k)
    out_df['shortDesc']=out_df['shortDesc'].apply(lambda cw : remove_tags(cw))
    return out_df


#agpal_search_results([])
#agpal_search_results([""])
#agpal_search_results(["Ontario","Funding"])

# title and description

st.set_page_config(layout="wide")
st.write("""# AgPal+ dummy
                
                """)

queries = st.text_input("Search something:", "agri food")

with open('data/corpus_embeddings_sbert.p', 'rb') as fp:
    corpus_embeddings = pickle.load(fp)
    
with open('data/embedder.p', 'rb') as fp:
    embedder = pickle.load(fp)

data =  pd.read_csv("data/data.csv")
    

query_embeddings = embedder.encode([queries], convert_to_tensor=True)
query_embeddings = util.normalize_embeddings(query_embeddings)

hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)

def relevant_results(query_index):
  return data.loc[pd.DataFrame(hits[query_index])['corpus_id'].to_numpy(),]


#get search results 
our_model_results = relevant_results(0)[["id","title","shortDesc"]]
agpal_results = agpal_search_results(queries.split(),10)

#get symmetric difference
x = set(our_model_results["id"])
print(x)
y = set(agpal_results["ID"])
print(y)

k = x.intersection(y)
print(k)
z = x.difference(y)

how_to_text = "Current solution: we represent the user search query and agpal listings as normalized vectors using SBERT word embeddings. I return the top 10 listings that are the closest to the user search query as measured by dot-product."
improvement_text = " Improvement: When user types in long setences our tool works better. But we need more work on short phrases, such as 'Ontario funding'... What could be helpful: fuzzy matching, tuning SBERT with labelled data, look into Solr implementation, define a ranking function; Agpal's user experience can be improved quite easily, exact match then semantic match; make agpal split on user experience."
skepticism_txt = "Skepticism: to be honest, I don't think untuned large models matcing on similarity scores ALONE would be helpful at all (e.g. listings from irrelevant provinces)."


st.markdown(how_to_text)
st.markdown(improvement_text)
st.markdown(skepticism_txt)

data_container = st.container()

with data_container:
    table,plot = st.columns(2  )
    with table:
        st.header('Our Results:')
        st.dataframe(our_model_results)
    with plot:
        st.header('Agpal Results')
        st.dataframe(agpal_results)


st.markdown('Common results: '+str(k))