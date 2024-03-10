# -*- coding: utf-8 -*-
"""vectordb_chromadb.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1utitzIUqL0zF2mf2KWDFvLsSnMYelL94
"""

!pip install openai

from openai import OpenAI

import requests
import json

# Define your OpenAI API key
OPENAI_API_KEY = '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'

# Define the endpoint URL
url = 'https://api.openai.com/v1/embeddings'

# Define the headers
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {OPENAI_API_KEY}'
}

# Define the data payload
data = {
    'input': 'Your text string goes here',
    'model': 'text-embedding-3-small'
}

# Convert data to JSON format
data_json = json.dumps(data)

# Make the API call using requests.post()
response = requests.post(url, headers=headers, data=data_json)

# Check the response
if response.status_code == 200:
    embedding = response.json()
    print(embedding)
    with open('embedding.json', 'w') as f:
        json.dump(embedding, f, indent=4)
    print("Embedding written to embedding.json successfully.")
else:
    print("Error:", response.text)

# Define the list of sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I enjoy reading books on a rainy day.",
    "Artificial intelligence is shaping the future.",
    "The sun sets beautifully over the horizon.",
    "Coding is like solving puzzles."
]

# Initialize list to store embeddings
embeddings = []

# Loop through each sentence and get embedding
for sentence in sentences:
    # Define the data payload
    data = {
        'input': sentence,
        'model': 'text-embedding-3-small'
    }

    # Convert data to JSON format
    data_json = json.dumps(data)

    # Make the API call using requests.post()
    response = requests.post(url, headers=headers, data=data_json)

    # Check the response
    if response.status_code == 200:
        embedding = response.json()['data'][0]['embedding']
        embeddings.append(embedding)
    else:
        print("Error:", response.text)

# Print or do further processing with the embeddings list
print(embeddings)

!pip install chromadb

import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_openai_embedded_collection")

# Prepare the collection data as a list of dictionaries

metadata_list = [{"source": "my_source"} for _ in range(len(embeddings))]
id_list = [f"id{i}" for i in range(1, len(embeddings) + 1)]

collection.add(
    embeddings=embeddings,
    documents=sentences,
    metadatas=metadata_list,
    ids=id_list
)

# Define the input sentence
input_sentence = "future is bright"

# Define the data payload
data = {
    'input': input_sentence,
    'model': 'text-embedding-3-small'
}

# Convert data to JSON format
data_json = json.dumps(data)

# Make the API call using requests.post()
response = requests.post(url, headers=headers, data=data_json)

# Initialize the variable to store the embedding
embedding = None

# Check the response
if response.status_code == 200:
    embedding = response.json()['data'][0]['embedding']
else:
    print("Error:", response.text)

# Now you can use the 'embedding' variable to query the collection
query_texts = [embedding]

results = collection.query(
    query_embeddings=query_texts,
    n_results=1
)

results

collection2 = chroma_client.create_collection(name="my_default_embedded_collection")

# Prepare the collection data as a list of dictionaries

metadata_list = [{"source": "my_source"} for _ in range(len(embeddings))]
id_list = [f"id{i}" for i in range(1, len(embeddings) + 1)]

collection2.add(
    documents=sentences,
    metadatas=metadata_list,
    ids=id_list
)

results = collection2.query(
    query_texts=["future is bright"],
    n_results=1
)

results