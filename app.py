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
sparse_encoder = BM25Encoder()
#input_classifier = pipeline("text-classification", model="edncodeismad/bert_rag_classifier")
input_classifier = pipeline("text-classification", model="luzalbaposse/HelloBERT")

"""
updated to give multiple recommendations (make number dynamical)

fix input classifier
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
    data = pd.read_csv('planet_organic.csv')
    row = data.loc[data[ID_COL] == id]
    row = f'PRODUCT NAME: {row[NAME_COL].values[0]}, PRICE: {row[PRICE_COL].values[0]}, \nPRODUCT INFO: {row[INFO_COL].values[0]}'
    return row

def classify_input(input):
    input_res = input_classifier(input)[0]
    if input_res['label'] == 'greeting':
        return False
    else:
        return True

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
    global prev_response
    request_data = request.json
    user_message = request_data.get('message', '')

    perform_search = classify_input(user_message)

    if perform_search:
        pre_input = f"Create a concise one-line prompt to feed into a vector similarity search to get a relevant product for this query. Strip all unnecessary conversations and words such as 'I am looking for...' and only keep useful keywords. If reference is made to multiple different products, only create a query based on the last product mentioned in the queries:\nSequence of queries: {'\n'.join(chat_history)}\n{user_message}"
        chat_history.append(user_message)
        search_query = llm.write_message(pre_input)

        if HYBRID:
            ids = send_hybrid_query(search_query)
        else:
            ids = send_query(search_query)

        # use cosine similarity here (from db.py)

        product_results = [get_info(id) for id in ids]
        
        prompt = f"Instructions: Always answer as a helpful shop assistant in one or two short sentences. Include the URL link for the product if it is provided in this prompt. Select the most relevant product from the list. If none of the product information is significantly relevant to the user input, do not answer the question and ignore all the product information, and ask the user to rephrase their question.\n\nUser input: {user_message}\n\nProduct information: {'\n'.join(product_results)}"
        prompt = f"Instructions: Answer as a helpful shop assistant in one or two short sentences. Only choose the most relevant product from the list. Only answer questions related to your products (you sell food products). If none of the products listed here are relevant to the query, ignore the products and ask the user to rephrase their question.\n\nUser input: {user_message}\n\nYour previous response: {prev_response}\n\nProduct information: {'\n'.join(product_results)}"
        #prompt = f"Instructions: Answer as a helpful shop assistant in one or two short sentences. Only answer questions related to your products (you sell food products). If none of the products listed here are relevant to the query, ignore the products and ask the user to rephrase their question.\n\nUser input: {user_message}. User search: {search_query}\n\nChoose from the following products: {'\n'.join(product_results)}"
        response = llm.write_message(prompt)


        # CHECK IF THE OUTPUT IS RELEVANT TO ANY GIVEN PRODUCT BEFORE PASSING IMAGE AND URL
        scores = []
        for id in ids:
            score = get_cosine_score(id, response)
            scores.append(score)
        sorted_ids = [id for id, s in sorted(zip(ids, scores), key = lambda x: x[1], reverse=True)]
        chosen_id = sorted_ids[0]
        #chosen_id2 = sorted_ids[1]
        
        dp_score = get_cosine_score(chosen_id, response)
        if dp_score < DP_THRESHOLD:
            img, link = None, None
        else:
            img = f'images/{chosen_id}.png'
            link = data.loc[data[ID_COL] == chosen_id][URL_COL].values[0]
    
    else:
        prompt = f"Instructions: Answer as a helpful shop assistant in one or two short sentences. Only answer questions related to your products (you sell food products). Don't make up answers - if you don't have the information, ask the user to rephrase their question.\n\nUser input: {user_message}."
        response = llm.write_message(prompt)

        img, link = None, None

    prev_response = response

    return jsonify({
        "response": response,
        "image": img,
        "url": link
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)