import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import nltk
import os
import openai
from dotenv import load_dotenv
import sys
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load .env
load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Get the current working directory
cwd = os.getcwd()

# Sentence Transformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

custom_path = os.path.join(cwd, 'nltk_data')
if not os.path.exists(custom_path):
    os.makedirs(custom_path)
    nltk.data.path.append(os.path.join(cwd, 'nltk_data'))
    nltk.download('punkt')


def read_and_split(file_name):
    with open(file_name, 'r') as file:
        text = file.read()
    sentences = nltk.tokenize.sent_tokenize(text)
    return sentences


def build_or_load_index(index_file, embeddings):
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        faiss.write_index(index, index_file)
    return index


file_name = 'information.txt'  # replace with your file name
sentences = read_and_split(file_name)
embeddings = model.encode(sentences)


custom_path = os.path.join(cwd, 'conversation')
if not os.path.exists(custom_path):
    os.makedirs(custom_path)

index_file = './conversation/faiss.index'
index = build_or_load_index(index_file, embeddings)

if len(sys.argv) > 1:
    query = sys.argv[1]
else:
    print('Please provide a query as a command-line argument.')
    sys.exit()

query_vector = model.encode([query])
D, I = index.search(query_vector, k=10)

# print("\n\n##### searched #####")
summary = ""
for i in I[0]:
    summary += sentences[i] + "\n"
# print(summary)

messages_file = './conversation/messages.pkl'
if os.path.exists(messages_file):
    with open(messages_file, 'rb') as f:
        messages = pickle.load(f)
        messages.extend([
            {"role": "user", "content": query}
        ])
else:
    messages = [
        {"role": "system", "content": "This is casual conversation with a specific topic. You must reply as shortly. Now you can use these additional background knowledge to talk with. \n\nBackground knowledge: " + summary + "\n\n"},
        {"role": "user", "content": query}
    ]


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=300
)

# print("\n\n##### Answered #####")
print(response.choices[0].message.content)

messages.extend([
    {"role": "assistant", "content": response.choices[0].message.content}
])

# Save the messages for the next run
with open(messages_file, 'wb') as f:
    pickle.dump(messages, f)
