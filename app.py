import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pickle
import pandas as pd 
import json
import requests
import torch
import re
import pytz
from datetime import datetime
import numpy as np
import streamlit as st 
from pandas import DataFrame
import ast
import json
from gspread_pandas import Spread,Client
from google.oauth2 import service_account

#from st_aggrid import AgGrid, GridUpdateMode
#from st_aggrid.grid_options_builder import GridOptionsBuilder

# https://curlconverter.com/
# https://www.convertcsv.com/json-to-csv.htm

st.set_page_config(layout="wide")

#@st.cache(allow_output_mutation=True)
def initialize():
    
    with open('data/corpus_embeddings_sbert.p', 'rb') as fp:
        corpus_embeddings = pickle.load(fp)
        
    with open('data/embedder.p', 'rb') as fp:
        embedder = pickle.load(fp)
    
    data =  pd.read_csv("data/data.csv")
    return corpus_embeddings,embedder,data


corpus_embeddings,embedder,data = initialize()

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
    dat = f'------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="queryaction"\r\n\r\nfieldquery\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="fq"\r\n\r\n{search_keys}\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="lang"\r\n\r\nen\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="pageSize"\r\n\r\n25\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="sort"\r\n\r\nscore:desc\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX\r\nContent-Disposition: form-data; name="sort"\r\n\r\nsort2:asc\r\n------WebKitFormBoundary7qAY07ZLedVRqMnX--\r\n'
    response = requests.post('https://prodapi.agpal.ca/api/v1/search', headers=headers, data=dat)
    json_doc = list(json.loads(json.dumps(response.json())).values())[1] #relevant search results
    agpal_results = []
    for i in json_doc:
      agpal_results.append([i.get("ID"),i.get("title"), i.get("shortDesc")])
    out_df = pd.DataFrame(agpal_results, columns = ['id', 'title',"shortDesc"]).head(top_k)  
    data['index'] = data.index
    merged = out_df.merge(data,on='id', how="left")
    merged = pd.DataFrame(merged[merged.shortDesc_x.notnull()])[["id","title_x","shortDesc_x","index"]]
    merged.columns = ["id","title","shortDesc","index"]
    merged["index"] = merged["index"].astype(str).replace(r'\.\d+$', '', regex=True)
    merged['shortDesc']=merged['shortDesc'].apply(lambda cw : remove_tags(cw)) #remove html tags and trim spaces
    merged.set_index("index", inplace = True) 
    return merged 

#agpal_search_results([])
#agpal_search_results([""])
#agpal_search_results(["Ontario","Funding"],10)
#agpal_search_results(["agri","food"],10)


# UI 
# title and description
st.write("""# AgPal+ dummy
                
                """)

queries = st.text_input("Search something:", "ontario funding")
if len(queries) == 0: #handle empty user strings
    queries  = "ontario funding"
    
query_embeddings = embedder.encode([queries], convert_to_tensor=True)
query_embeddings = util.normalize_embeddings(query_embeddings)

hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)

def relevant_results(query_index):
  return data.loc[pd.DataFrame(hits[query_index])['corpus_id'].to_numpy(),]


#get search results
num_results = 10 
our_model_results = relevant_results(0)[["id","title","shortDesc"]]
agpal_results = agpal_search_results(queries.split(),num_results)

#get symmetric difference
x = set(our_model_results["id"])
#print(x)
y = set(agpal_results["id"])
#print(y)

x_y_intersect = x.intersection(y)
x_index = set(our_model_results.index)
y_index = set(agpal_results.index)
x_y_index_union = x_index.union(y_index)
x_y_index_union = set([int(x) for x in x_y_index_union if x!='nan'])

# Show some intro
how_to_text = "Current solution: we represent the user search query and agpal listings as normalized vectors using SBERT word embeddings. I return the top 10 AgPal     listings that are the closest to the user search query as measured by dot-product."
improvement_text = " Improvement: When user types in long setences our tool works better. But we need more work on short phrases, such as 'Ontario funding'... What could be helpful: fuzzy matching, tuning SBERT with labelled data, look into Solr implementation, define a ranking function; Agpal's user search query can be improved quite easily, exact match then semantic match; make agpal split on user search query."
skepticism_txt = "Skepticism: to be honest, I don't think untuned large models matcing on similarity scores ALONE would be helpful at all (e.g. listings from irrelevant provinces)."

#st.markdown(how_to_text)
#st.markdown(improvement_text)
#st.markdown(skepticism_txt)

data_container = st.container()
mutiselect_container = st.container()

with data_container:
    
    sbert_table,agpal_table = st.columns(2)
    
    with sbert_table:
        st.header('Our Results')
        st.dataframe(our_model_results) #.assign(hack='').set_index('hack')    
        
            
    with agpal_table:
        st.header('Agpal Results')
        st.dataframe(agpal_results)
        
        
with mutiselect_container:
    col1,col2,col3 = st.columns([4,1,5]) 
    
    with col1:
        selected_indices = st.multiselect('Enter bad match indices from the above searches: ', x_y_index_union)
        
    with col3:
        st.write("") #placeholder for alignment
        st.write("") #placeholder for alignment
        button_sent = st.button("Submit!")

# print common matches (our solution vs. agpal)
if len(x_y_intersect) >0:
    st.markdown("Our search matched with AgPal's search: " +str(x_y_intersect))
else:
    st.markdown('Our search results did not match with those of the AgPal.')


#collect user feedback
feedback_colnames = ["id","title"]
bad_results =data.loc[set(selected_indices)][feedback_colnames]
z = set(bad_results['id']) #id's of bad listings (indexed)


num_na = sum(np.array(agpal_results.index == 'nan'))#number of unindexed agpal posts 
num_valid = 10-num_na

if button_sent and bad_results.shape[0]==0:
    st.write("Number of indexed agpal results: ",num_valid)
    st.write("Accuracy ratio: ", 1)
    
if button_sent and bad_results.shape[0]>0:

    #performance metrics calculation
    x_minus_z = x.difference(z)
    y_minus_z = y.difference(z)
    
    if num_valid ==0:
        accuracy_ratio = 'None of the results fetched was indexed. Re-download data now.'
    else:
        
        accuracy_ratio = (1.00*len(x_minus_z)/10)/((len(y_minus_z)-num_na)/num_valid)
    
    benchmark = accuracy_ratio-1
    
    
    if benchmark >= 0:
        st.write("Our model is ","{0:.0%}".format(benchmark), ' better!' )
    else: 
        st.write("Our model is ","{0:.0%}".format(-benchmark), ' worse...' )
    
    # submit feedback to build training data 
    curr_time = str(datetime.now(pytz.utc))
    
    #bad results identified by the user
    bad_results['create_timestamp'] =  curr_time
    bad_results['match_quality'] = 'bad'
    
    #acceptable results 
    okay_results_our_model = our_model_results.loc[set(our_model_results.index) - set(selected_indices)][feedback_colnames]
    okay_results_our_model['create_timestamp'] =  curr_time
    
    okay_results_agpal = agpal_results.loc[set(agpal_results.index) - set(selected_indices)][feedback_colnames]
    okay_results_agpal['create_timestamp'] =  curr_time

    okay_results = pd.concat([okay_results_our_model, okay_results_agpal])
    okay_results['match_quality'] = 'okay'
    
    all_results_df = pd.concat([bad_results, okay_results])
    all_results_df["user_search_string"]= queries
    #probably want to add another indicator for data source
    
    #used https://stackoverflow.com/questions/64817724/pandas-df-to-json-with-duplicate-keys
    all_results_json = all_results_df.set_index('create_timestamp').groupby('create_timestamp').apply(lambda x: x.to_dict('r')).to_dict() #pd.DataFrame.to_json(all_results_df)
    
    st.write("(IN DEVELOPMENT) Uploading your feedback:") 
    #st.write(all_results_json) #visualize your feedback
 
    
    #append the json recorder in gcp storage or some stuff
    # Create a Google Authentication connection object
    # tutorial: https://www.notion.so/avra-youtube/Streamlit-Google-Sheets-Automation-76c67cfa6d784b2eba195cc454a4dbaa
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    
    service_account_info = json.load(open('agpal_plus_8d2114c78f0a.json'))
    credentials = service_account.Credentials.from_service_account_info(
                   service_account_info, scopes = scope)
    
    
    client = Client(scope=scope,creds=credentials)
    
    spreadsheetname = "agpal_feedback_db"
    spread = Spread(spreadsheetname,client = client)
    
    sh = client.open(spreadsheetname)
    worksheet_list = sh.worksheets()
    
    # Functions 
    @st.cache()
    # Get our worksheet names
    def worksheet_names():
        sheet_names = []   
        for sheet in worksheet_list:
            sheet_names.append(sheet.title)  
        return sheet_names
    
    # Get the sheet as dataframe
    def load_the_spreadsheet(spreadsheetname):
        worksheet = sh.worksheet(spreadsheetname)
        df = DataFrame(worksheet.get_all_records())
        return df
    
    # Update to Sheet
    def update_the_spreadsheet(spreadsheetname,dataframe):
        spread.df_to_sheet(dataframe,sheet = spreadsheetname,index = False)
        st.write('Success! Your feedback has been uploaded to this GoogleSheet: ',spread.url)
        
    df = load_the_spreadsheet('searches')
    new_df = df.append(all_results_df,ignore_index=True)
    update_the_spreadsheet("searches",new_df)