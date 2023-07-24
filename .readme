# Project: Knowledge-Based GPT
Utilizes a vector database and OpenAI API.

# How to Use:

## Preparation
Install necessary packages with `pip install -r requirements.txt`.
Create a `/.env` file and set your OpenAI key as follows: `OPENAI_API_KEY = [OPENAI KEY]`

## Knowledge Preparation
Add the information you want to use into `information.txt`. For instance, you can scrape an entire Wikipedia page and add it.

## One-Time Chat
Run `python embedding_test.py "Your question"` For example: `python embedding_test.py "Explain about Russia's invasion in Ukraine"`
The required embedding model will automatically be downloaded to `/nltk_data`. There is no need for you to manually manage this.

## Continuous Chat
Run `python embedding_conversation.py "Your question"` For example: `python embedding_conversation.py "Let's talk about Russia's invasion in Ukraine"`

## Caution
A `faiss.index` file will automatically be created in the `/conversation` directory. If you modify the contents of `information.txt`, make sure to delete the `/conversation/faiss.index` file beforehand.
The dialogue of the conversation is stored in `/conversation/messages.pkl`. If you want to reset the conversation history, delete this file.




# Comparison

## Original ChatGPT
![ChatGPT](./images/original%20chatgpt.png)

## knowledge based ChatGPT
![ChatGPT](./images/informed%20chatgpt.png)