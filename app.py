from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from usingllm import GPT
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone_text.sparse import BM25Encoder
import nltk
nltk.download('punkt_tab')
sparse_encoder = BM25Encoder()

"""
!! use the previous response (?)
"""

pc = Pinecone(api_key="e6890c15-db55-4a6f-95e1-0810d25641bf")
model = SentenceTransformer('all-MiniLM-L6-v2')
data = pd.read_csv('planet_organic.csv')

llm_model = 'gpt-3.5-turbo' # gpt-3.5-turbo' or gpt-4o or
llm = GPT(llm_model)

app = Flask(__name__, static_folder='build', static_url_path='/')
CORS(app)

HYBRID = True

DP_THRESHOLD = 0.5 # dot product threshold

ID_COL = 'id'
NAME_COL = 'name'
PRICE_COL = 'price'
INFO_COL = 'info'
URL_COL = 'url'

if HYBRID:
    index = pc.Index('food-products-hybrid')
else:
    index = pc.Index('food-products')

chat_history = []

# train the sparse encoder
def fit_sparse_encoder(data):
    corpus = []
    for _, row in data.iterrows():
        try:
            entry = row[INFO_COL].strip().replace('\n\n' ,'\n').replace('\n', ' ')
        except:
            continue
        corpus.append(entry)
    sparse_encoder.fit(corpus)
fit_sparse_encoder(data)

def send_query(input):
    input = list(model.encode(input))
    input = [float(v) for v in input]
    results = index.query(
        namespace='product-info',
        vector=input,
        top_k=4,
        include_values=False
    )
    chunk_ids = [entry['id'] for entry in results.matches]
    product_ids = [int(id)//10000 for id in chunk_ids]
    return product_ids

def send_hybrid_query(input):
    dense_input = list(model.encode(input))
    dense_input = [float(v) for v in dense_input]
    sparse_input = sparse_encoder.encode_queries(input)
    results = index.query(
        vector=dense_input,
        sparse_vector=sparse_input,
        top_k=4,
        include_values=False
    )
    chunk_ids = [entry['id'] for entry in results.matches]
    product_ids = [int(id)//10000 for id in chunk_ids]
    return product_ids

def get_info(id):
    row = data.loc[data[ID_COL] == id]
    row = f'PRODUCT NAME: {row[NAME_COL].values[0]}, PRICE: {row[PRICE_COL].values[0]}, URL: {row[URL_COL].values[0]}\nPRODUCT INFO: {row[INFO_COL].values[0]}'
    return row

def get_cosine_score(id, input):
    row = data.loc[data[ID_COL] == id]
    product_info = f'Name: {row[NAME_COL].values[0]}, price: {row[PRICE_COL].values[0]}, Info: {row[INFO_COL].values[0]}'
    p = model.encode(product_info)
    input = model.encode(input)
    score = cosine_similarity(p.reshape(1, -1), input.reshape(1, -1))
    return score[0][0]

prev_response = 'Hi, how can I help you today?'

#@app.route('/')
#def serve_frontend():
#    return send_from_directory(app.static_folder, 'index.html')

@app.route('/')
def serve_frontend():
    return send_file('build/index.html')

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global prev_response, chat_history
    request_data = request.json
    user_message = request_data.get('message', '')
    chat_history.append(user_message)
    joined_history = '\n'.join(chat_history)
    #pre_input = f"Create a concise one-line prompt to feed into a vector similarity search to get a relevant product for this query. Strip all unnecessary conversations and words such as 'I am looking for...', 'Search for' etc. and only keep useful keywords. If reference is made to multiple different products, only create a query based on the last product mentioned in the queries:\nSequence of queries: {joined_history}\n{user_message}"
    system_prompt = "You are tasked with creating a concise, one-line query for a vector similarity search to find the most relevant product for a user's request. Follow these rules: \n(1) Strip unnecessary words like 'I am looking for...' or 'Search for,' keeping only essential keywords. DO NOT USE WORDS SUCH AS 'SEARCH FOR'. \n(2) If reference is made to multiple different products, only create a query based on the last product mentioned in the queries, for example the sequence ['chicken', 'pasta'] would result in 'pasta' since it is the last product mentioned. The sequence ['I am loking for a cutting board', 'something cheaper?'] would result in a query like 'cheap cutting board' since this combines the relevant parts of the specific product that the user is looking for. \n(4) If the last query is not product-related or irrelevant or nonsensical (e.g., 'How are you?' or 'What's your name?' or 'abcdefsle'), respond only with 'irrelevant' and nothing else. Note that single words such as 'chicken', 'soy sauce' are relevant."
    prompt = f"Here is the sequence of user queries, separated by newlines (most recent come last): \n{joined_history}"
    chat_history = chat_history[-3:]
    search_query = llm.write_message(prompt, system_prompt)

    if search_query == 'irrelevant':
        system_prompt = f"Answer as a helpful ecommerce store assistant in one or two short sentences. Skip intro words such as 'Hi', 'Hello!', etc. Only answer questions related to your products (your online store sells food products). DO NOT MAKE UP ANSWERS - you can make small talk, but ask the user to rephrase their question if any specific information (such as the product link, price etc.) is needed. Respond normally to the user's small talk (eg. Hello, How can you help me) but DO NOT engage in any irrelevant conversation other than selling your products."
        prompt = f"User input: {user_message}"
        response = llm.write_message(prompt, system_prompt)

        img, link = None, None
    
    
    else:

        if HYBRID:
            ids = send_hybrid_query(search_query)
        else:
            ids = send_query(search_query)

        # solid up to here
        product_results = [get_info(id) for id in ids]
        joined_results = '\n'.join(product_results)
        system_prompt = f"Answer as a helpful ecommerce store assistant in one or two short sentences. Only answer questions related to your food products. Only pick the most relevant product from the list. Don't make up answers - if you don't have the information, ask the user to rephrase their question. Remember that you are assisting the user on a website, so do not make mention of the physical store eg. 'the fruit section'. If the URL is passed to you, include the link to the product with this syntax: **[Name of product](url-link-to-product)**. If the user's message is not relevant to any of the products mentioned, say that you don't have the information and ask the user to rephrase their question, and DO NOT use any of the product information in your answer."
        prompt = f"Respond to this user query: {user_message} \n\nRespond based on these product results, separated by newlines: \n{joined_results}"# \n\nYour previous response was: {prev_response}"
        response = llm.write_message(prompt, system_prompt)

        scores = []
        for id in ids:
            score = get_cosine_score(id, response)
            scores.append(score)
        sorted_ids = [id for id, s in sorted(zip(ids, scores), key = lambda x: x[1], reverse=True)]
        chosen_id = sorted_ids[0]
        
        dp_score = get_cosine_score(chosen_id, response)
        if dp_score < DP_THRESHOLD:
            img, link = None, None
        else:
            img = f'images/{chosen_id}.png'
            link = data.loc[data[ID_COL] == chosen_id][URL_COL].values[0]
    
    prev_response = response

    return jsonify({
        "response": response,
        "image": img,
        "url": link
    })

if __name__ == '__main__':
    app.run()