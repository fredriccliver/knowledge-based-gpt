import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import nltk
import os
import openai
from dotenv import load_dotenv
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load .env
load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Get the current working directory
cwd = os.getcwd()

# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# all models: https://www.sbert.net/docs/pretrained_models.html

custom_path = os.path.join(cwd, 'nltk_data')
if not os.path.exists(custom_path):
    os.makedirs(custom_path)
    # Specify the custom path
    nltk.data.path.append(os.path.join(cwd, 'nltk_data'))
    # Now download the data
    nltk.download('punkt')


def read_and_split(file_name):
    with open(file_name, 'r') as file:
        text = file.read()
    sentences = nltk.tokenize.sent_tokenize(text)
    return sentences


file_name = 'information.txt'  # replace with your file name
sentences = read_and_split(file_name)
# print(sentences)


# Generate embeddings for each sentence in the list
embeddings = model.encode(sentences)

# Get the dimension of the embeddings
d = embeddings.shape[1]

# Build the index with the correct dimension
index = faiss.IndexFlatL2(d)

# Add embeddings to the Faiss index
index.add(embeddings)

# Check if a query was passed as a command-line argument
if len(sys.argv) > 1:
    # Given a query sentence from the command-line arguments
    query = sys.argv[1]
else:
    print('Please provide a query as a command-line argument.')
    sys.exit()


# Given a query sentence
# query = "언제 이 사건이 일어났어?"


# Get the query vector
query_vector = model.encode([query])


# Search the Faiss index
# Returns distances and indices of k-nearest neighbors
D, I = index.search(query_vector, k=10)

# Loop through the list of indices
print("\n\n##### searched #####")
summary = ""
for i in I[0]:
    summary += sentences[i] + "\n"
print(summary)


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. You are helping a user find information about the fact that you can find in this summary.\n\nSummary: " + summary + "\n\n"},
        {"role": "user", "content": query}
    ],
    max_tokens=600
)

print("\n\n##### Answered #####")
print(response.choices[0].message.content)
